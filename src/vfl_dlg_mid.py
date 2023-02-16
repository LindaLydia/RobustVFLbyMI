import logging
import pprint
import time

import tensorflow as tf

import torch
import torch.nn.functional as F
import numpy as np

from models.vision import *
from utils import *

from marvell_model import (
    update_all_norm_leak_auc,
    update_all_cosine_leak_auc,
    KL_gradient_perturb
)
import marvell_shared_values as shared_var

# BOTTLENECK_SCALE = 25
BOTTLENECK_SCALE = 1

tf.compat.v1.enable_eager_execution() 


class LabelLeakage(object):
    def __init__(self, args):
        '''
        :param args:  contains all the necessary parameters
        '''
        self.dataset = args.dataset
        self.model = args.model
        self.num_exp = args.num_exp
        self.epochs = args.epochs
        self.lr = args.lr
        self.early_stop = args.early_stop
        self.early_stop_param = args.early_stop_param
        self.device = args.device
        self.batch_size_list = args.batch_size_list
        self.num_class_list = args.num_class_list
        self.dst = args.dst
        self.exp_res_dir = args.exp_res_dir
        self.exp_res_path = args.exp_res_path
        self.apply_trainable_layer = args.apply_trainable_layer
        self.apply_laplace = args.apply_laplace
        self.apply_gaussian = args.apply_gaussian
        self.dp_strength = args.dp_strength
        self.apply_grad_spar = args.apply_grad_spar
        self.grad_spars = args.grad_spars
        self.apply_encoder = args.apply_encoder
        # self.apply_random_encoder = args.apply_random_encoder
        # self.apply_adversarial_encoder = args.apply_adversarial_encoder
        self.ae_lambda = args.ae_lambda
        self.encoder = args.encoder
        self.apply_marvell = args.apply_marvell
        self.marvell_s = args.marvell_s
        # self.apply_ppdl = args.apply_ppdl
        # self.ppdl_theta_u = args.ppdl_theta_u
        # self.apply_gc = args.apply_gc
        # self.gc_preserved_percent = args.gc_preserved_percent
        # self.apply_lap_noise = args.apply_lap_noise
        # self.noise_scale = args.noise_scale
        self.apply_discrete_gradients = args.apply_discrete_gradients
        self.discrete_gradients_bins = args.discrete_gradients_bins
        self.discrete_gradients_bound = args.discrete_gradients_bound
        self.apply_mi = args.apply_mi
        self.mi_loss_lambda = args.mi_loss_lambda
        self.apply_mid = args.apply_mid
        self.mid_tau = args.mid_tau
        self.mid_loss_lambda = args.mid_loss_lambda
        self.mid_model = None
        self.mid_enlarge_model = None
        self.apply_distance_correlation = args.apply_distance_correlation
        self.distance_correlation_lambda = args.distance_correlation_lambda
        self.apply_grad_perturb = args.apply_grad_perturb
        self.perturb_epsilon = args.perturb_epsilon
        self.apply_RRwithPrior = args.apply_RRwithPrior
        self.RRwithPrior_epsilon = args.RRwithPrior_epsilon        
        # self.show_param()

    def show_param(self):
        print(f'********** config dict **********')
        pprint.pprint(self.__dict__)

    def calc_label_recovery_rate(self, dummy_label, gt_label):
        success = torch.sum(torch.argmax(dummy_label, dim=-1) == torch.argmax(gt_label, dim=-1)).item()
        total = dummy_label.shape[0]
        return success / total

    def get_random_softmax_onehot_label(self, gt_onehot_label):
        _random = torch.randn(gt_onehot_label.size()).to(self.device)
        for i in range(len(gt_onehot_label)):
            max_index, = torch.where(_random[i] == _random[i].max())
            max_label, = torch.where(gt_onehot_label[i] == gt_onehot_label[i].max())
            while len(max_index) > 1:
                temp = torch.randn(gt_onehot_label[i].size()).to(self.device)
                # temp = torch.randn(gt_onehot_label[i].size())
                max_index, = torch.where(temp == temp.max())
                _random[i] = temp.clone()
            assert(len(max_label)==1)
            max_index = max_index.item()
            max_label = max_label.item()
            if max_index != max_label:
                temp = _random[i][int(max_index)].clone()
                _random[i][int(max_index)] = _random[i][int(max_label)].clone()
                _random[i][int(max_label)] = temp.clone()
            _random[i] = F.softmax(_random[i], dim=-1)
        return self.encoder(_random)

    def train(self):
        '''
        execute the label inference algorithm
        :return: recovery rate
        '''

        print(f"Running on %s{torch.cuda.current_device()}" % self.device)
        if self.dataset == 'nuswide':
            all_nuswide_labels = []
            for line in os.listdir('../../../share_dataset/NUS_WIDE/Groundtruth/AllLabels'):
                all_nuswide_labels.append(line.split('_')[1][:-4])
        for batch_size in self.batch_size_list:
            for num_classes in self.num_class_list:
                if self.apply_mid:
                    self.mid_model = MID_layer(num_classes*BOTTLENECK_SCALE, num_classes).to(self.device)
                    self.mid_enlarge_model = MID_enlarge_layer(num_classes,num_classes*2*BOTTLENECK_SCALE).to(self.device)
                classes = [None] * num_classes
                gt_equal_probability = torch.from_numpy(np.array([1/num_classes]*num_classes)).to(self.device)
                print("gt_equal_probability:", gt_equal_probability)
                if self.dataset == 'cifar20':
                    classes = random.sample(list(range(20)), num_classes)
                    all_data, all_label = get_class_i(self.dst, classes)
                elif self.dataset == 'cifar100':
                    # if apply the defense, we only use cifar20
                    # if self.apply_laplace or self.apply_gaussian or self.apply_grad_spar:
                    #     classes = [i for i in range(num_classes)]
                    # else:
                        # classes = random.sample(list(range(20)), num_classes)
                    classes = random.sample(list(range(100)), num_classes)
                    all_data, all_label = get_class_i(self.dst, classes)
                elif self.dataset == 'mnist':
                    classes = random.sample(list(range(10)), num_classes)
                    all_data, all_label = get_class_i(self.dst, classes)
                elif self.dataset == 'nuswide':
                    classes = random.sample(all_nuswide_labels, num_classes)
                    x_image, x_text, Y = get_labeled_data('../../../share_dataset/NUS_WIDE', classes, None, 'Train')
                elif self.dataset == 'cifar10':
                    classes = random.sample(list(range(10)), num_classes)
                    all_data, all_label = get_class_i(self.dst, classes)

                recovery_rate_history = []
                for i_run in range(1, self.num_exp + 1):
                    start_time = time.time()
                    # randomly sample
                    if self.dataset == 'mnist' or self.dataset == 'cifar100' or self.dataset == 'cifar20' or self.dataset == 'cifar10':
                        gt_data = []
                        gt_label = []
                        for i in range(0, batch_size):
                            sample_idx = torch.randint(len(all_data), size=(1,)).item()
                            gt_data.append(all_data[sample_idx])
                            gt_label.append(all_label[sample_idx])
                        gt_data = torch.stack(gt_data).to(self.device)
                        half_size = list(gt_data.size())[-1] // 2
                        gt_data_a = gt_data[:, :, :half_size, :]
                        gt_data_b = gt_data[:, :, half_size:, :]
                        gt_label = torch.stack(gt_label).to(self.device)
                        gt_onehot_label = gt_label  # label_to_onehot(gt_label)
                    elif self.dataset == 'nuswide':
                        gt_data_a, gt_data_b, gt_label = [], [], []
                        for i in range(0, batch_size):
                            sample_idx = torch.randint(len(x_image), size=(1,)).item()
                            gt_data_a.append(torch.tensor(x_text[sample_idx], dtype=torch.float32))
                            gt_data_b.append(torch.tensor(x_image[sample_idx], dtype=torch.float32))
                            gt_label.append(torch.tensor(Y[sample_idx], dtype=torch.float32))
                        gt_data_a = torch.stack(gt_data_a).to(self.device)
                        gt_data_b = torch.stack(gt_data_b).to(self.device)
                        gt_label = torch.stack(gt_label).to(self.device)
                        gt_onehot_label = gt_label  # label_to_onehot(gt_label)
                    if self.apply_grad_perturb:
                        gt_onehot_label = label_perturb(gt_onehot_label, self.perturb_epsilon)
                    if self.apply_RRwithPrior:
                        gt_onehot_label = RRwithPrior(gt_onehot_label, self.RRwithPrior_epsilon, gt_equal_probability)
                    if self.apply_encoder:
                        _, gt_onehot_label = self.encoder(gt_onehot_label)
                    # if self.apply_encoder:
                    #     if not self.apply_random_encoder:
                    #         _, gt_onehot_label = self.encoder(gt_onehot_label) # get the result given by AutoEncoder.forward
                    #     else:
                    #         _, gt_onehot_label = self.get_random_softmax_onehot_label(gt_onehot_label)
                    # if self.apply_adversarial_encoder:
                    #     _, gt_onehot_label = self.encoder(gt_data_a)
                    # set model
                    if self.model == 'MLP2':
                        net_a = MLP2(np.prod(list(gt_data_a.size())[1:]), num_classes).to(self.device)
                        net_b = MLP2(np.prod(list(gt_data_b.size())[1:]), num_classes).to(self.device)
                    elif self.model == 'resnet18':
                        net_a = resnet18(num_classes).to(self.device)
                        net_b = resnet18(num_classes).to(self.device)
                    
                    # ......if args.apply_certify != 0 and epoch >= args.certify_start_epoch:
                    #     .....

                    criterion = cross_entropy_for_onehot
                    pred_a = net_a(gt_data_a) # for passive party: H_p, Z
                    pred_b = net_b(gt_data_b) # for active party
                    ######################## defense start ############################
                    ######################## defense1: trainable layer ############################
                    if self.apply_trainable_layer:
                        active_aggregate_model = ActivePartyWithTrainableLayer(hidden_dim=num_classes * 2, output_dim=num_classes)
                        dummy_active_aggregate_model = ActivePartyWithTrainableLayer(hidden_dim=num_classes * 2, output_dim=num_classes)
                    else:
                        active_aggregate_model = ActivePartyWithoutTrainableLayer()
                        dummy_active_aggregate_model = ActivePartyWithoutTrainableLayer()
                    active_aggregate_model = active_aggregate_model.to(self.device)
                    dummy_active_aggregate_model = dummy_active_aggregate_model.to(self.device)
                    pred = active_aggregate_model(pred_a, pred_b)
                    loss = criterion(pred, gt_onehot_label)
                    ######################## defense3: mutual information defense ############################
                    if self.apply_mid:
                        epsilon = torch.empty((pred_a.size()[0],pred_a.size()[1]*BOTTLENECK_SCALE))
                        
                        # # discrete form of reparameterization
                        # torch.nn.init.uniform(epsilon) # epsilon is initialized
                        # epsilon = - torch.log(epsilon + torch.tensor(1e-07))
                        # epsilon = - torch.log(epsilon + torch.tensor(1e-07)) # prevent if epsilon=0.0
                        # pred_Z = F.softmax(pred_a) + epsilon.to(self.device)
                        # pred_Z = F.softmax(pred_Z / torch.tensor(self.mid_tau).to(self.device), -1)
                        
                        # continuous form of reparameterization
                        torch.nn.init.normal(epsilon, mean=0, std=1) # epsilon is initialized
                        epsilon = epsilon.to(self.device)
                        # # pred_a.size() = (batch_size, class_num)
                        pred_a_double = self.mid_enlarge_model(pred_a)
                        mu, std = pred_a_double[:,:num_classes*BOTTLENECK_SCALE], pred_a_double[:,num_classes*BOTTLENECK_SCALE:]
                        std = F.softplus(std-5) # ? F.softplus(std-5)
                        # print("mu, std: ", mu.size(), std.size())
                        # print(pred_a.size(), pred_a_double.size(), mu.size(), std.size(), epsilon.size())
                        pred_Z = mu+std*epsilon
                        # assert(pred_Z.size()==pred_a.size())
                        # print(pred_Z.size())
                        pred_Z = pred_Z.to(self.device)
                        
                        pred_Z = self.mid_model(pred_Z)
                        pred = active_aggregate_model(pred_Z, F.softmax(pred_b))
                        # # loss for discrete form of reparameterization
                        # loss = criterion(pred, gt_onehot_label) + self.mid_loss_lambda * entropy_for_probability_vector(pred_a)
                        # loss for continuous form of reparameterization
                        loss = criterion(pred, gt_onehot_label) + self.mid_loss_lambda * torch.mean(torch.sum((-0.5)*(1+2*torch.log(std)-mu**2 - std**2),1))
                    ######################## defense4: distance correlation ############################
                    if self.apply_distance_correlation:
                        # loss = criterion(pred, gt_onehot_label) + self.distance_correlation_lambda * torch.mean(torch.cdist(pred_a, gt_onehot_label, p=2))
                        loss = criterion(pred, gt_onehot_label) + self.distance_correlation_lambda * torch.log(tf_distance_cov_cor(pred_a, gt_onehot_label))
                    ######################## defense with loss change end ############################
                    pred_a_gradients = torch.autograd.grad(loss, pred_a, retain_graph=True)
                    # print("pred_a_gradients:", pred_a_gradients)
                    pred_a_gradients_clone = pred_a_gradients[0].detach().clone()
                    ######################## defense2: dp ############################
                    if self.apply_laplace and self.dp_strength != 0 or self.apply_gaussian and self.dp_strength != 0:
                        location = 0.0
                        threshold = 0.2  # 1e9
                        if self.apply_laplace:
                            with torch.no_grad():
                                scale = self.dp_strength
                                # clip 2-norm per sample
                                norm_factor_a = torch.div(torch.max(torch.norm(pred_a_gradients_clone, dim=1)),threshold + 1e-6).clamp(min=1.0)
                                # add laplace noise
                                dist_a = torch.distributions.laplace.Laplace(location, scale)
                                pred_a_gradients_clone = torch.div(pred_a_gradients_clone, norm_factor_a) + \
                                           dist_a.sample(pred_a_gradients_clone.shape).to(self.device)
                        elif self.apply_gaussian:
                            with torch.no_grad():
                                scale = self.dp_strength
                                norm_factor_a = torch.div(torch.max(torch.norm(pred_a_gradients_clone, dim=1)),
                                                           threshold + 1e-6).clamp(min=1.0)
                                pred_a_gradients_clone = torch.div(pred_a_gradients_clone, norm_factor_a) + \
                                                       torch.normal(location, scale, pred_a_gradients_clone.shape).to(self.device)
                    ######################## defense3: gradient sparsification ############################
                    elif self.apply_grad_spar and self.grad_spars != 0:
                        with torch.no_grad():
                            percent = self.grad_spars / 100.0
                            up_thr = torch.quantile(torch.abs(pred_a_gradients_clone), percent)
                            active_up_gradients_res = torch.where(
                                torch.abs(pred_a_gradients_clone).double() < up_thr.item(),
                                pred_a_gradients_clone.double(), float(0.)).to(self.device)
                            pred_a_gradients_clone = pred_a_gradients_clone - active_up_gradients_res
                    # ######################## defense4: marvell ############################
                    # elif self.apply_marvell and self.marvell_s != 0 and num_classes == 2:
                    #     # for marvell, change label to [0,1]
                    #     marvell_y = []
                    #     for i in range(len(gt_label)):
                    #         marvell_y.append(int(gt_label[i][1]))
                    #     marvell_y = np.array(marvell_y)
                    #     shared_var.batch_y = np.asarray(marvell_y)
                    #     logdir = 'marvell_logs/dlg_task/{}_logs/{}'.format(self.dataset, time.strftime("%Y%m%d-%H%M%S"))
                    #     writer = tf.summary.create_file_writer(logdir)
                    #     shared_var.writer = writer
                    #     with torch.no_grad():
                    #         pred_a_gradients_clone = KL_gradient_perturb(pred_a_gradients_clone, classes, self.marvell_s)
                    #         pred_a_gradients_clone = pred_a_gradients_clone.to(self.device)
                    ######################## defense5: ppdl, GradientCompression, laplace_noise, DiscreteSGD ############################
                    # elif self.apply_ppdl:
                    #     dp_gc_ppdl(epsilon=1.8, sensitivity=1, layer_grad_list=[pred_a_gradients_clone], theta_u=self.ppdl_theta_u, gamma=0.001, tau=0.0001)
                    # elif self.apply_gc:
                    #     tensor_pruner = TensorPruner(zip_percent=self.gc_preserved_percent)
                    #     tensor_pruner.update_thresh_hold(pred_a_gradients_clone)
                    #     pred_a_gradients_clone = tensor_pruner.prune_tensor(pred_a_gradients_clone)
                    # elif self.apply_lap_noise:
                    #     dp = DPLaplacianNoiseApplyer(beta=self.noise_scale)
                    #     pred_a_gradients_clone = dp.laplace_mech(pred_a_gradients_clone)
                    elif self.apply_discrete_gradients:
                        pred_a_gradients_clone = multistep_gradient(pred_a_gradients_clone, bins_num=self.discrete_gradients_bins, bound_abs=self.discrete_gradients_bound)
                    ######################## defense end #################################################### defense3: mid ############################
                    original_dy_dx = torch.autograd.grad(pred_a, net_a.parameters(), grad_outputs=pred_a_gradients_clone)

                    dummy_pred_b = torch.randn(pred_b.size()).to(self.device).requires_grad_(True)
                    dummy_label = torch.randn(gt_onehot_label.size()).to(self.device).requires_grad_(True)

                    if self.apply_trainable_layer:
                        optimizer = torch.optim.Adam([dummy_pred_b, dummy_label] + list(dummy_active_aggregate_model.parameters()), lr=self.lr)
                    else:
                        optimizer = torch.optim.Adam([dummy_pred_b, dummy_label], lr=self.lr)

                    for iters in range(1, self.epochs + 1):
                        def closure():
                            optimizer.zero_grad()
                            dummy_pred = dummy_active_aggregate_model(net_a(gt_data_a), dummy_pred_b)

                            dummy_onehot_label = F.softmax(dummy_label, dim=-1)
                            dummy_loss = criterion(dummy_pred, dummy_onehot_label)
                            dummy_dy_dx_a = torch.autograd.grad(dummy_loss, net_a.parameters(), create_graph=True)
                            grad_diff = 0
                            for (gx, gy) in zip(dummy_dy_dx_a, original_dy_dx):
                                grad_diff += ((gx - gy) ** 2).sum()
                            grad_diff.backward()
                            return grad_diff

                        rec_rate = self.calc_label_recovery_rate(dummy_label, gt_label)
                        # if iters == 1:
                        #     append_exp_res(f'exp_result/{self.dataset}/exp_on_{self.dataset}_rec_rate_change.txt',
                        #                    f'{batch_size} 0 {rec_rate} {closure()}')
                        optimizer.step(closure)
                        # if self.calc_label_recovery_rate(dummy_label, gt_label) >= 99.99:
                        #     break

                        # append_exp_res(f'exp_result/{self.dataset}/exp_on_{self.dataset}_rec_rate_change.txt',
                        #                f'{batch_size} {iters} {rec_rate} {closure()}')
                        # if rec_rate >= 0.999:
                        #     break
                        # print(iters, "%.4f" % closure().item())
                        # if iters % 10 == 0:
                        #     print(iters, torch.sum(torch.argmax(dummy_label, dim=-1) == torch.argmax(gt_label, dim=-1)).item() / batch_size)
                        #     print(f'iters:{iters}, loss:{closure().item()}')
                        #     append_exp_res(f'exp_result/exp_on_{self.dataset}_loss.txt', f'{iters} {closure().item()}')
                        if self.early_stop == True:
                            if closure().item() < self.early_stop_param:
                                break

                    rec_rate = self.calc_label_recovery_rate(dummy_label, gt_label)
                    recovery_rate_history.append(rec_rate)
                    end_time = time.time()
                    # output the rec_info of this exp
                    if self.apply_laplace or self.apply_gaussian:
                        print(f'batch_size=%d,class_num=%d,exp_id=%d,dp_strength=%lf,recovery_rate=%lf,time_used=%lf'
                              % (batch_size, num_classes, i_run, self.dp_strength,rec_rate, end_time - start_time))
                    elif self.apply_grad_spar:
                        print(f'batch_size=%d,class_num=%d,exp_id=%d,grad_spars=%lf,recovery_rate=%lf,time_used=%lf'
                              % (batch_size, num_classes, i_run, self.grad_spars,rec_rate, end_time - start_time))
                    if self.apply_mid:
                        print(f'MID: batch_size=%d,class_num=%d,exp_id=%d,recovery_rate=%lf,time_used=%lf'
                              % (batch_size, num_classes, i_run, rec_rate, end_time - start_time))
                    elif self.apply_mi:
                        print(f'batch_size=%d,class_num=%d,exp_id=%d,mi_loss_lambda=%lf,recovery_rate=%lf,time_used=%lf'
                              % (batch_size, num_classes, i_run, self.mi_loss_lambda,rec_rate, end_time - start_time))
                    elif self.apply_RRwithPrior:
                        print(f'batch_size=%d,class_num=%d,exp_id=%d,RRwithPrior_epsilon=%lf,recovery_rate=%lf,time_used=%lf'
                              % (batch_size, num_classes, i_run, self.RRwithPrior_epsilon,rec_rate, end_time - start_time))
                    elif self.apply_distance_correlation:
                        print(f'batch_size=%d,class_num=%d,exp_id=%d,distance_correlationlambda=%lf,recovery_rate=%lf,time_used=%lf'
                              % (batch_size, num_classes, i_run, self.distance_correlation_lambda,rec_rate, end_time - start_time))
                    elif self.apply_grad_perturb:
                        print(f'batch_size=%d,class_num=%d,exp_id=%d,perturb_epsilon=%lf,recovery_rate=%lf,time_used=%lf'
                              % (batch_size, num_classes, i_run, self.perturb_epsilon,rec_rate, end_time - start_time))
                    elif self.apply_discrete_gradients:
                        print(f'batch_size=%d,class_num=%d,exp_id=%d,ae_lambda=%s,recovery_rate=%lf,time_used=%lf'
                              % (batch_size, num_classes, i_run, self.ae_lambda,rec_rate, end_time - start_time))
                    elif self.apply_encoder:
                        print(f'batch_size=%d,class_num=%d,exp_id=%d,ae_lambda=%s,recovery_rate=%lf,time_used=%lf'
                              % (batch_size, num_classes, i_run, self.ae_lambda,rec_rate, end_time - start_time))
                    elif self.apply_marvell:
                        print(f'batch_size=%d,class_num=%d,exp_id=%d,marvel_s=%lf,recovery_rate=%lf,time_used=%lf'
                              % (batch_size, num_classes, i_run, self.marvell_s,rec_rate, end_time - start_time))
                    else:
                        print(f'batch_size=%d,class_num=%d,exp_id=%d,recovery_rate=%lf,time_used=%lf'
                              % (batch_size, num_classes, i_run, rec_rate, end_time - start_time))
                avg_rec_rate = np.mean(recovery_rate_history)
                if self.apply_laplace or self.apply_gaussian:
                    exp_result = str(self.dp_strength) + ' ' + str(avg_rec_rate) + ' ' + str(recovery_rate_history) + ' ' + str(np.max(recovery_rate_history))
                elif self.apply_grad_spar:
                    exp_result = str(self.grad_spars) + ' ' + str(avg_rec_rate) + ' ' + str(recovery_rate_history) + ' ' + str(np.max(recovery_rate_history))
                # elif self.apply_encoder or self.apply_adversarial_encoder:
                #     exp_result = str(self.ae_lambda) + ' ' + str(avg_rec_rate) + ' ' + str(recovery_rate_history) + ' ' + str(np.max(recovery_rate_history))
                # elif self.apply_ppdl:
                #     exp_result = str(self.ppdl_theta_u) + ' ' + str(avg_rec_rate) + ' ' + str(recovery_rate_history) + ' ' + str(np.max(recovery_rate_history))
                # elif self.apply_gc:
                #     exp_result = str(self.gc_preserved_percent) + ' ' + str(avg_rec_rate) + ' ' + str(recovery_rate_history) + ' ' + str(np.max(recovery_rate_history))
                # elif self.apply_lap_noise:
                #     exp_result = str(self.noise_scale) + ' ' + str(avg_rec_rate) + ' ' + str(recovery_rate_history) + ' ' + str(np.max(recovery_rate_history))
                elif self.apply_discrete_gradients:
                    exp_result = str(self.discrete_gradients_bins) + ' ' + str(avg_rec_rate) + ' ' + str(recovery_rate_history) + ' ' + str(np.max(recovery_rate_history))
                elif self.apply_mid:
                    exp_result = str(self.mid_loss_lambda) + ' ' + str(avg_rec_rate) + ' ' + str(recovery_rate_history) + ' ' + str(np.max(recovery_rate_history)) + 'MID'
                elif self.apply_mi:
                    exp_result = str(self.mi_loss_lambda) + ' ' + str(avg_rec_rate) + ' ' + str(recovery_rate_history) + ' ' + str(np.max(recovery_rate_history))
                elif self.apply_RRwithPrior:
                    exp_result = str(self.RRwithPrior_epsilon) + ' ' + str(avg_rec_rate) + ' ' + str(recovery_rate_history) + ' ' + str(np.max(recovery_rate_history))
                elif self.apply_distance_correlation:
                    exp_result = str(self.distance_correlation_lambda) + ' ' + str(avg_rec_rate) + ' ' + str(recovery_rate_history) + ' ' + str(np.max(recovery_rate_history))
                elif self.apply_grad_perturb:
                    exp_result = str(self.perturb_epsilon) + ' ' + str(avg_rec_rate) + ' ' + str(recovery_rate_history) + ' ' + str(np.max(recovery_rate_history))
                elif self.apply_encoder:
                    exp_result = str(self.ae_lambda) + ' ' + str(avg_rec_rate) + ' ' + str(recovery_rate_history) + ' ' + str(np.max(recovery_rate_history))
                elif self.apply_marvell:
                    exp_result = str(self.marvell_s) + ' ' + str(avg_rec_rate) + ' ' + str(recovery_rate_history) + ' ' + str(np.max(recovery_rate_history))
                else:
                    exp_result = f"bs|num_class|recovery_rate,%d|%d| %lf %s %lf" % (batch_size, num_classes, avg_rec_rate, str(recovery_rate_history), np.max(recovery_rate_history))

                append_exp_res(self.exp_res_path, exp_result)
                print(exp_result)

if __name__ == '__main__':
    pass