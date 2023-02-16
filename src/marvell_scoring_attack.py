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
from scipy.stats import norm


tf.compat.v1.enable_eager_execution() 

class ScoringAttack(object):
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

    def train(self):


        print(f"Running on %s{torch.cuda.current_device()}" % self.device)
        if self.dataset == 'nuswide':
            all_nuswide_labels = []
            for line in os.listdir('../../../share_dataset/NUS_WIDE/Groundtruth/AllLabels'):
                all_nuswide_labels.append(line.split('_')[1][:-4])
        for batch_size in self.batch_size_list:
            for num_classes in self.num_class_list:
                if self.apply_mid:
                    self.mid_model = MID_layer(num_classes, num_classes).to(self.device)
                    self.mid_enlarge_model = MID_enlarge_layer(num_classes,num_classes*2).to(self.device)
                classes = [None] * num_classes
                gt_equal_probability = torch.from_numpy(np.array([1/num_classes]*num_classes)).to(self.device)
                if self.dataset == 'cifar100':
                    # if apply the defense, we only use cifar20
                    # if self.apply_laplace or self.apply_gaussian or self.apply_grad_spar:
                    #     classes = [i for i in range(num_classes)]
                    # else:
                    #     classes = random.sample(list(range(100)), num_classes)
                    classes = random.sample(list(range(100)), num_classes)
                    # classes = random.sample(list(range(20)), num_classes)
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
                norm_leak_auc_history = []
                cosine_leak_auc_history = []
                norm_leak_acc_history = []
                cosine_leak_acc_history = []
                for i_run in range(1, self.num_exp + 1):
                    start_time = time.time()
                    # randomly sample
                    if self.dataset == 'mnist' or self.dataset == 'cifar100' or self.dataset == 'cifar10':
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
                        # print(gt_data[0].size())
                        # half_size = list(gt_data[-1].size())[-1] // 2
                        # gt_data_a = [gt_data[i][:, :half_size, :] for i in range(len(gt_data))]
                        # gt_data_b = [gt_data[i][:, half_size:, :] for i in range(len(gt_data))]
                        gt_onehot_label = gt_label
                        gt_onehot_label = torch.stack(gt_onehot_label).to(self.device)
                        # gt_label = torch.stack(gt_label).to(self.device)
                        # gt_onehot_label = gt_label  # label_to_onehot(gt_label)
                    elif self.dataset == 'nuswide':
                        gt_data_a, gt_data_b, gt_label = [], [], []
                        for i in range(0, batch_size):
                            sample_idx = torch.randint(len(x_image), size=(1,)).item()
                            gt_data_a.append(torch.tensor(x_text[sample_idx], dtype=torch.float32))
                            gt_data_b.append(torch.tensor(x_image[sample_idx], dtype=torch.float32))
                            gt_label.append(torch.tensor(Y[sample_idx], dtype=torch.float32))
                        gt_data_a = torch.stack(gt_data_a).to(self.device)
                        gt_data_b = torch.stack(gt_data_b).to(self.device)
                        gt_onehot_label = gt_label  # label_to_onehot(gt_label)
                        gt_onehot_label = torch.stack(gt_onehot_label).to(self.device)
                        # gt_label = torch.stack(gt_label).to(self.device)
                        # gt_onehot_label = gt_label  # label_to_onehot(gt_label)
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
                        # net_a = MLP2(np.prod(list(gt_data_a[0].size())), num_classes).to(self.device)
                        # net_b = MLP2(np.prod(list(gt_data_b[0].size())), num_classes).to(self.device)
                    elif self.model == 'resnet18':
                        net_a = resnet18(num_classes).to(self.device)
                        net_b = resnet18(num_classes).to(self.device)
                    
                    # print("gt_label:", gt_label, len(gt_label), gt_label[0])
                    # assert 1==0
                    # ......if args.apply_certify != 0 and epoch >= args.certify_start_epoch:
                    #     .....

                    criterion = cross_entropy_for_onehot
                    # criterion = cross_entropy_for_onehot_samplewise
                    pred_a = net_a(gt_data_a) # for passive party: H_p, Z
                    pred_b = net_b(gt_data_b) # for active party
                    # pred_a_list = [net_a(gt_data_a[i]) for i in range(gt_data_a.size()[0])]
                    # pred_b_list = [net_b(gt_data_b[i]) for i in range(gt_data_b.size()[0])]
                    ######################## defense start ############################
                    ######################## defense1: trainable layer ############################
                    if self.apply_trainable_layer:
                        active_aggregate_model = ActivePartyWithTrainableLayer(hidden_dim=num_classes * 2, output_dim=num_classes)
                    else:
                        active_aggregate_model = ActivePartyWithoutTrainableLayer()
                    active_aggregate_model = active_aggregate_model.to(self.device)
                    pred = active_aggregate_model(pred_a, pred_b)
                    loss = criterion(pred, gt_onehot_label)
                    # print(type(loss), loss.size(),loss[-1])
                    # pred_list = [active_aggregate_model(pred_a_list[i], pred_b_list[i]) for i in range(len(pred_a_list))]
                    # loss_list = [criterion(pred_list[i], gt_onehot_label[i]) for i in range(len(pred_list))]
                    # # print(type(loss), loss.size(),loss[-1])

                    ################ scoring attack ################
                    ################ find a positive gradient ################
                    pos_idx = np.random.randint(len(gt_label))
                    while torch.argmax(gt_label[pos_idx]) != torch.tensor(1):
                        pos_idx += 1
                        if pos_idx >= len(gt_label):
                            pos_idx -= len(gt_label)
                    # print(pos_idx)
                    # pos_pred_a = net_a(gt_data_a[pos_idx:pos_idx+1]) # for passive party: H_p, Z
                    # pos_pred_b = net_b(gt_data_b[pos_idx:pos_idx+1]) # for active party
                    # pos_pred = active_aggregate_model(pos_pred_a, pos_pred_b)
                    # pos_loss = criterion(pos_pred, gt_onehot_label[pos_idx:pos_idx+1])
                    ################ found positive gradient ################

                    ######################## defense3: mutual information defense ############################
                    if self.apply_mid:
                        epsilon = torch.empty(pred_a.size())
                        
                        # # discrete form of reparameterization
                        # torch.nn.init.uniform(epsilon) # epsilon is initialized
                        # epsilon = - torch.log(epsilon + torch.tensor(1e-07))
                        # epsilon = - torch.log(epsilon + torch.tensor(1e-07)) # prevent if epsilon=0.0
                        # pred_Z = F.softmax(pred_a,dim=-1) + epsilon.to(self.device)
                        # pred_Z = F.softmax(pred_Z / torch.tensor(self.mid_tau).to(self.device), ,dim=-1)
                        
                        # continuous form of reparameterization
                        torch.nn.init.normal_(epsilon, mean=0, std=1) # epsilon is initialized
                        epsilon = epsilon.to(self.device)
                        # # pred_a.size() = (batch_size, class_num)
                        pred_a_double = self.mid_enlarge_model(pred_a)
                        mu, std = pred_a_double[:,:num_classes], pred_a_double[:,num_classes:]
                        std = F.softplus(std-5, beta=1) # ? F.softplus(std-5)
                        # print("mu, std: ", mu.size(), std.size())
                        pred_Z = mu+std*epsilon
                        assert(pred_Z.size()==pred_a.size())
                        pred_Z = pred_Z.to(self.device)
                        
                        pred_Z = self.mid_model(pred_Z)
                        pred = active_aggregate_model(pred_Z, pred_b)
                        # # loss for discrete form of reparameterization
                        # loss = criterion(pred, gt_onehot_label) + self.mid_loss_lambda * entropy_for_probability_vector(pred_a)
                        # loss for continuous form of reparameterization
                        loss = criterion(pred, gt_onehot_label) + self.mid_loss_lambda * torch.mean(torch.sum((-0.5)*(1+2*torch.log(std)-mu**2 - std**2),1))
                        # pos_loss = criterion(pred[pos_idx:pos_idx+1], gt_onehot_label[pos_idx:pos_idx+1]) + self.mid_loss_lambda * torch.mean(torch.sum((-0.5)*(1+2*torch.log(std)-mu**2 - std**2),1)) / pred_Z.size()[0]

                        # ########################### v2 #############################################
                        # t_samples = self.mid_model(pred_a)
                        # positdive = torch.zeros_like(t_samples)
                        # prediction_1 = t_samples.unsqueeze(1)  # [nsample,1,dim]
                        # t_samples_1 = t_samples.unsqueeze(0)  # [1,nsample,dim]
                        # negative = - ((t_samples_1 - prediction_1) ** 2).mean(dim=1) / 2.   # [nsample, dim]
                        # pred = active_aggregate_model(t_samples, F.softmax(pred_b,dim=-1))
                        # loss = criterion(pred, gt_onehot_label) + self.mid_loss_lambda * (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()
                        # ########################### v3 #############################################
                        # epsilon = torch.empty((pred_a.size()[0],pred_a.size()[1]))
                        # torch.nn.init.normal_(epsilon, mean=0, std=1) # epsilon is initialized
                        # epsilon = epsilon.to(self.device)
                        # mu = torch.mean(pred_a)
                        # std = torch.std(pred_a, unbiased=False)
                        # # mu, std = norm.fit(pred_a.cpu().detach().numpy())
                        # _samples = mu + std * epsilon
                        # _samples = _samples.to(self.device)
                        # t_samples = self.mid_model(_samples)
                        # pred = active_aggregate_model(t_samples, pred_b)
                        # loss = criterion(pred, gt_onehot_label) + self.mid_loss_lambda * (-0.5)*(1+2*torch.log(std)-mu**2 - std**2)
                        # # loss = criterion(pred, gt_onehot_label) + self.mid_loss_lambda * torch.mean(torch.sum((-0.5)*(1+2*torch.log(std)-mu**2 - std**2),1))

                    ######################## defense4: distance correlation ############################
                    if self.apply_distance_correlation:
                        loss = criterion(pred, gt_onehot_label) + self.distance_correlation_lambda * torch.log(tf_distance_cov_cor(pred_a, gt_onehot_label))
                        # pos_loss = criterion(pred[pos_idx:pos_idx+1], gt_onehot_label[pos_idx:pos_idx+1]) + self.distance_correlation_lambda * torch.mean(torch.cdist(pred_a, gt_onehot_label, p=2)) / pred_Z.size()[0]
                    ######################## defense with loss change end ############################
                    pred_a_gradients = torch.autograd.grad(loss, pred_a, retain_graph=True)
                    # print("pred_a_gradients:", pred_a_gradients)
                    pred_a_gradients_clone = pred_a_gradients[0].detach().clone()
                    # pos_pred_a_gradients = torch.autograd.grad(pos_loss, pos_pred_a, retain_graph=True)
                    # # print("pos_pred_a_gradients:", pos_pred_a_gradients)
                    # pos_pred_a_gradients_clone = pos_pred_a_gradients[0].detach().clone()
                    # print("size of two clone gradients:", pred_a_gradients_clone.size(),pos_pred_a_gradients_clone.size())
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
                                # # positive clip 2-norm per sample
                                # pos_norm_factor_a = torch.div(torch.max(torch.norm(pos_pred_a_gradients_clone, dim=1)),threshold + 1e-6).clamp(min=1.0)
                                # # add laplace noise
                                # pos_dist_a = torch.distributions.laplace.Laplace(location, scale)
                                # pos_pred_a_gradients_clone = torch.div(pos_pred_a_gradients_clone, pos_norm_factor_a) + \
                                #            pos_dist_a.sample(pos_pred_a_gradients_clone.shape).to(self.device)
                        elif self.apply_gaussian:
                            with torch.no_grad():
                                scale = self.dp_strength
                                norm_factor_a = torch.div(torch.max(torch.norm(pred_a_gradients_clone, dim=1)),
                                                        threshold + 1e-6).clamp(min=1.0)
                                temp = torch.empty(pred_a_gradients_clone.shape)
                                torch.nn.init.normal_(temp, mean=location, std=scale)
                                # pred_a_gradients_clone = torch.div(pred_a_gradients_clone, norm_factor_a) + \
                                #                     torch.normal_(location, scale, pred_a_gradients_clone.shape).to(self.device)
                                pred_a_gradients_clone = torch.div(pred_a_gradients_clone, norm_factor_a) + temp.to(self.device)
                                # pos_norm_factor_a = torch.div(torch.max(torch.norm(pos_pred_a_gradients_clone, dim=1)),
                                #                            threshold + 1e-6).clamp(min=1.0)
                                # pos_pred_a_gradients_clone = torch.div(pos_pred_a_gradients_clone, pos_norm_factor_a) + \
                                #                        torch.normal_(location, scale, pos_pred_a_gradients_clone.shape).to(self.device)
                    ######################## defense3: gradient sparsification ############################
                    elif self.apply_grad_spar and self.grad_spars != 0:
                        with torch.no_grad():
                            percent = self.grad_spars / 100.0
                            up_thr = torch.quantile(torch.abs(pred_a_gradients_clone), percent)
                            active_up_gradients_res = torch.where(
                                torch.abs(pred_a_gradients_clone).double() < up_thr.item(),
                                pred_a_gradients_clone.double(), float(0.)).to(self.device)
                            pred_a_gradients_clone = pred_a_gradients_clone - active_up_gradients_res
                            # pos_up_thr = torch.quantile(torch.abs(pos_pred_a_gradients_clone), percent)
                            # pos_active_up_gradients_res = torch.where(
                            #     torch.abs(pos_pred_a_gradients_clone).double() < pos_up_thr.item(),
                            #     pos_pred_a_gradients_clone.double(), float(0.)).to(self.device)
                            # pos_pred_a_gradients_clone = pos_pred_a_gradients_clone - pos_active_up_gradients_res
                    ######################## defense4: marvell ############################
                    elif self.apply_marvell and self.marvell_s != 0 and num_classes == 2:
                        # for marvell, change label to [0,1]
                        marvell_y = []
                        for i in range(len(gt_label)):
                            marvell_y.append(int(gt_label[i][1]))
                        marvell_y = np.array(marvell_y)
                        shared_var.batch_y = np.asarray(marvell_y)
                        logdir = 'marvell_logs/dlg_task/{}_logs/{}'.format(self.dataset, time.strftime("%Y%m%d-%H%M%S"))
                        writer = tf.summary.create_file_writer(logdir)
                        shared_var.writer = writer
                        with torch.no_grad():
                            pred_a_gradients_clone = KL_gradient_perturb(pred_a_gradients_clone, classes, self.marvell_s)
                            pred_a_gradients_clone = pred_a_gradients_clone.to(self.device)
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
                        # pos_pred_a_gradients_clone = multistep_gradient(pos_pred_a_gradients_clone, bins_num=self.discrete_gradients_bins, bound_abs=self.discrete_gradients_bound)
                    ######################## defense end #################################################### defense3: mid ############################
                    
                    original_dy_dx = torch.autograd.grad(pred_a, net_a.parameters(), grad_outputs=pred_a_gradients_clone)
                    # pos_original_dy_dx = torch.autograd.grad(pred_a[pos_idx:pos_idx+1], net_a.parameters(), grad_outputs=pos_pred_a_gradients_clone)
                    
                    tf_pred_a_gradients_clone = tf.convert_to_tensor(pred_a_gradients_clone.cpu().numpy())
                    tf_gt_label = tf.convert_to_tensor([tf.convert_to_tensor(torch.argmax(gt_label[i]).cpu().numpy()) for i in range(len(gt_label))])
                    # print(tf_pred_a_gradients_clone[0], tf_gt_label[0], type(tf_pred_a_gradients_clone))
                    # assert 0==1
                    # gradient_norm = tf.norm(tf_pred_a_gradients_clone, axis=-1, keepdims=False)
                    # numpy_gradients_norm = gradient_norm.numpy()
                    # numpy_label = tf_gt_label.numpy()
                    # negative_gradients_norm = []
                    # positive_gradients_norm = []
                    # for i in range(len(numpy_label)):
                    #     if float(numpy_label[i]) == 0.0:
                    #         negative_gradients_norm.append(gradient_norm[i].numpy())
                    #     else:
                    #         positive_gradients_norm.append(gradient_norm[i].numpy())
                    # print(negative_gradients_norm[0],positive_gradients_norm[0])
                    # import matplotlib.pyplot as plt
                    # plt.cla()
                    # plt.hist(negative_gradients_norm, color='b', bins = 20, alpha=0.5, label='negative')
                    # plt.hist(positive_gradients_norm, color='r', bins = 20, alpha=0.5, label='positive')
                    # plt.savefig(f'images_bins/{i_run}_bins.png')

                    norm_leak_auc, norm_leak_acc = update_all_norm_leak_auc(
                        norm_leak_auc_dict={'only':''},
                        grad_list=[tf_pred_a_gradients_clone],
                        y=tf_gt_label)
                    cosine_leak_auc, cosine_leak_acc = update_all_cosine_leak_auc(
                        cosine_leak_auc_dict={'only':''},
                        grad_list=[tf_pred_a_gradients_clone],
                        pos_grad_list=[tf_pred_a_gradients_clone[pos_idx:pos_idx+1]],
                        y=tf_gt_label)
                    print("norm_leak_auc:", norm_leak_auc)
                    print("cosine_leak_auc:", cosine_leak_auc)
                    print("norm_leak_acc:", norm_leak_acc)
                    print("cosine_leak_acc:", cosine_leak_acc)
                    norm_leak_auc_history.append(norm_leak_auc)
                    cosine_leak_auc_history.append(cosine_leak_auc)
                    norm_leak_acc_history.append(norm_leak_acc)
                    cosine_leak_acc_history.append(cosine_leak_acc)


                    # rec_rate = self.calc_label_recovery_rate(dummy_label, gt_label)
                    # recovery_rate_history.append(rec_rate)
                    end_time = time.time()
                    # output the rec_info of this exp
                    if self.apply_laplace or self.apply_gaussian:
                        print(f'batch_size=%d,class_num=%d,exp_id=%d,dp_strength=%lf,norm_leak_auc=%lf,cosine_leak_auc=%lf,time_used=%lf'
                              % (batch_size, num_classes, i_run, self.dp_strength,norm_leak_auc, cosine_leak_auc,  end_time - start_time))
                        print(f'batch_size=%d,class_num=%d,exp_id=%d,dp_strength=%lf,norm_leak_acc=%lf,cosine_leak_acc=%lf,time_used=%lf'
                              % (batch_size, num_classes, i_run, self.dp_strength,norm_leak_acc, cosine_leak_acc,  end_time - start_time))
                    elif self.apply_grad_spar:
                        print(f'batch_size=%d,class_num=%d,exp_id=%d,grad_spars=%lf,norm_leak_auc=%lf,cosine_leak_auc=%lf,time_used=%lf'
                              % (batch_size, num_classes, i_run, self.grad_spars,norm_leak_auc, cosine_leak_auc,  end_time - start_time))
                        print(f'batch_size=%d,class_num=%d,exp_id=%d,grad_spars=%lf,norm_leak_acc=%lf,cosine_leak_acc=%lf,time_used=%lf'
                              % (batch_size, num_classes, i_run, self.grad_spars,norm_leak_acc, cosine_leak_acc,  end_time - start_time))
                    if self.apply_mid:
                        print(f'MID: batch_size=%d,class_num=%d,exp_id=%d,norm_leak_auc=%lf,cosine_leak_auc=%lf,time_used=%lf'
                              % (batch_size, num_classes, i_run, norm_leak_auc, cosine_leak_auc,  end_time - start_time))
                        print(f'MID: batch_size=%d,class_num=%d,exp_id=%d,norm_leak_acc=%lf,cosine_leak_acc=%lf,time_used=%lf'
                              % (batch_size, num_classes, i_run, norm_leak_acc, cosine_leak_acc,  end_time - start_time))
                    elif self.apply_mi:
                        print(f'batch_size=%d,class_num=%d,exp_id=%d,mi_loss_lambda=%lf,norm_leak_auc=%lf,cosine_leak_auc=%lf,time_used=%lf'
                              % (batch_size, num_classes, i_run, self.mi_loss_lambda,norm_leak_auc, cosine_leak_auc,  end_time - start_time))
                        print(f'batch_size=%d,class_num=%d,exp_id=%d,mi_loss_lambda=%lf,norm_leak_acc=%lf,cosine_leak_acc=%lf,time_used=%lf'
                              % (batch_size, num_classes, i_run, self.mi_loss_lambda,norm_leak_acc, cosine_leak_acc,  end_time - start_time))
                    elif self.apply_distance_correlation:
                        print(f'batch_size=%d,class_num=%d,exp_id=%d,distance_correlationlambda=%lf,norm_leak_auc=%lf,cosine_leak_auc=%lf,time_used=%lf'
                              % (batch_size, num_classes, i_run, self.distance_correlation_lambda,norm_leak_auc, cosine_leak_auc,  end_time - start_time))
                        print(f'batch_size=%d,class_num=%d,exp_id=%d,distance_correlationlambda=%lf,norm_leak_acc=%lf,cosine_leak_acc=%lf,time_used=%lf'
                              % (batch_size, num_classes, i_run, self.distance_correlation_lambda,norm_leak_acc, cosine_leak_acc,  end_time - start_time))
                    elif self.apply_RRwithPrior:
                        print(f'batch_size=%d,class_num=%d,exp_id=%d,RRwithPrior_epsilon=%lf,norm_leak_auc=%lf,cosine_leak_auc=%lf,time_used=%lf'
                              % (batch_size, num_classes, i_run, self.RRwithPrior_epsilon ,norm_leak_auc, cosine_leak_auc,  end_time - start_time))
                        print(f'batch_size=%d,class_num=%d,exp_id=%d,RRwithPrior_epsilon=%lf,norm_leak_acc=%lf,cosine_leak_acc=%lf,time_used=%lf'
                              % (batch_size, num_classes, i_run, self.RRwithPrior_epsilon,norm_leak_acc, cosine_leak_acc,  end_time - start_time))
                    elif self.apply_grad_perturb:
                        print(f'batch_size=%d,class_num=%d,exp_id=%d,perturb_epsilon=%lf,norm_leak_auc=%lf,cosine_leak_auc=%lf,time_used=%lf'
                              % (batch_size, num_classes, i_run, self.perturb_epsilon,norm_leak_auc, cosine_leak_auc,  end_time - start_time))
                        print(f'batch_size=%d,class_num=%d,exp_id=%d,perturb_epsilon=%lf,norm_leak_acc=%lf,cosine_leak_acc=%lf,time_used=%lf'
                              % (batch_size, num_classes, i_run, self.perturb_epsilon,norm_leak_acc, cosine_leak_acc,  end_time - start_time))
                    elif self.apply_discrete_gradients:
                        print(f'batch_size=%d,class_num=%d,exp_id=%d,ae_lambda=%s,norm_leak_auc=%lf,cosine_leak_auc=%lf,time_used=%lf'
                              % (batch_size, num_classes, i_run, self.ae_lambda,norm_leak_auc, cosine_leak_auc,  end_time - start_time))
                        print(f'batch_size=%d,class_num=%d,exp_id=%d,ae_lambda=%s,norm_leak_acc=%lf,cosine_leak_acc=%lf,time_used=%lf'
                              % (batch_size, num_classes, i_run, self.ae_lambda,norm_leak_acc, cosine_leak_acc,  end_time - start_time))
                    elif self.apply_encoder:
                        print(f'batch_size=%d,class_num=%d,exp_id=%d,ae_lambda=%s,norm_leak_auc=%lf,cosine_leak_auc=%lf,time_used=%lf'
                              % (batch_size, num_classes, i_run, self.ae_lambda,norm_leak_auc, cosine_leak_auc,  end_time - start_time))
                        print(f'batch_size=%d,class_num=%d,exp_id=%d,ae_lambda=%s,norm_leak_acc=%lf,cosine_leak_acc=%lf,time_used=%lf'
                              % (batch_size, num_classes, i_run, self.ae_lambda,norm_leak_acc, cosine_leak_acc,  end_time - start_time))
                    elif self.apply_marvell:
                        print(f'batch_size=%d,class_num=%d,exp_id=%d,marvel_s=%lf,norm_leak_auc=%lf,cosine_leak_auc=%lf,time_used=%lf'
                              % (batch_size, num_classes, i_run, self.marvell_s,norm_leak_auc, cosine_leak_auc,  end_time - start_time))
                        print(f'batch_size=%d,class_num=%d,exp_id=%d,marvel_s=%lf,norm_leak_acc=%lf,cosine_leak_acc=%lf,time_used=%lf'
                              % (batch_size, num_classes, i_run, self.marvell_s,norm_leak_acc, cosine_leak_acc,  end_time - start_time))
                    else:
                        print(f'batch_size=%d,class_num=%d,exp_id=%d,norm_leak_auc=%lf,cosine_leak_auc=%lf,time_used=%lf'
                              % (batch_size, num_classes, i_run, norm_leak_auc, cosine_leak_auc, end_time - start_time))
                        print(f'batch_size=%d,class_num=%d,exp_id=%d,norm_leak_acc=%lf,cosine_leak_acc=%lf,time_used=%lf'
                              % (batch_size, num_classes, i_run, norm_leak_acc, cosine_leak_acc, end_time - start_time))
                # avg_rec_rate = np.mean(recovery_rate_history)
                if self.apply_laplace or self.apply_gaussian:
                    exp_result = str(self.dp_strength) + ' ' + str(np.mean(norm_leak_auc_history)) + ' AUC ' + str(norm_leak_auc_history) + ' ' + str(np.max(norm_leak_auc_history))
                    exp_result1 = str(self.dp_strength) + ' ' + str(np.mean(cosine_leak_auc_history)) + ' AUC ' + str(cosine_leak_auc_history) + ' ' + str(np.max(cosine_leak_auc_history))
                elif self.apply_grad_spar:
                    exp_result = str(self.grad_spars) + ' ' + str(np.mean(norm_leak_auc_history)) + ' AUC ' + str(norm_leak_auc_history) + ' ' + str(np.max(norm_leak_auc_history))
                    exp_result1 = str(self.grad_spars) + ' ' + str(np.mean(cosine_leak_auc_history)) + ' AUC ' + str(cosine_leak_auc_history) + ' ' + str(np.max(cosine_leak_auc_history))
                # elif self.apply_encoder or self.apply_adversarial_encoder:
                #     exp_result = str(self.ae_lambda) + ' ' + str(np.mean(norm_leak_auc_history)) + ' AUC ' + str(norm_leak_auc_history) + ' ' + str(np.max(norm_leak_auc_history))
                #     exp_result1 = str(self.ae_lambda) + ' ' + str(np.mean(cosine_leak_auc_history)) + ' AUC ' + str(cosine_leak_auc_history) + ' ' + str(np.max(cosine_leak_auc_history))
                # elif self.apply_ppdl:
                #     exp_result = str(self.ppdl_theta_u) + ' ' + str(np.mean(norm_leak_auc_history)) + ' AUC ' + str(norm_leak_auc_history) + ' ' + str(np.max(norm_leak_auc_history))
                #     exp_result1 = str(self.ppdl_theta_u) + ' ' + str(np.mean(cosine_leak_auc_history)) + ' AUC ' + str(cosine_leak_auc_history) + ' ' + str(np.max(cosine_leak_auc_history))
                # elif self.apply_gc:
                #     exp_result = str(self.gc_preserved_percent) + ' ' + str(np.mean(norm_leak_auc_history)) + ' AUC ' + str(norm_leak_auc_history) + ' ' + str(np.max(norm_leak_auc_history))
                #     exp_result1 = str(self.gc_preserved_percent) + ' ' + str(np.mean(cosine_leak_auc_history)) + ' AUC ' + str(cosine_leak_auc_history) + ' ' + str(np.max(cosine_leak_auc_history))
                # elif self.apply_lap_noise:
                #     exp_result = str(self.noise_scale) + ' ' + str(np.mean(norm_leak_auc_history)) + ' AUC ' + str(norm_leak_auc_history) + ' ' + str(np.max(norm_leak_auc_history))
                #     exp_result1 = str(self.noise_scale) + ' ' + str(np.mean(cosine_leak_auc_history)) + ' AUC ' + str(cosine_leak_auc_history) + ' ' + str(np.max(cosine_leak_auc_history))
                elif self.apply_encoder:
                    exp_result = str(self.ae_lambda) + ' ' + str(np.mean(norm_leak_auc_history)) + ' AUC ' + str(norm_leak_auc_history) + ' ' + str(np.max(norm_leak_auc_history))
                    exp_result1 = str(self.ae_lambda) + ' ' + str(np.mean(cosine_leak_auc_history)) + ' AUC ' + str(cosine_leak_auc_history) + ' ' + str(np.max(cosine_leak_auc_history))
                elif self.apply_discrete_gradients:
                    exp_result = str(self.discrete_gradients_bins) + ' ' + str(np.mean(norm_leak_auc_history)) + ' AUC ' + str(norm_leak_auc_history) + ' ' + str(np.max(norm_leak_auc_history))
                    exp_result1 = str(self.discrete_gradients_bins) + ' ' + str(np.mean(cosine_leak_auc_history)) + ' AUC ' + str(cosine_leak_auc_history) + ' ' + str(np.max(cosine_leak_auc_history))
                elif self.apply_mid:
                    exp_result = str(self.mid_loss_lambda) + ' ' + str(np.mean(norm_leak_auc_history)) + ' AUC ' + str(norm_leak_auc_history) + ' ' + str(np.max(norm_leak_auc_history))
                    exp_result1 = str(self.mid_loss_lambda) + ' ' + str(np.mean(cosine_leak_auc_history)) + ' AUC ' + str(cosine_leak_auc_history) + ' ' + str(np.max(cosine_leak_auc_history))
                elif self.apply_mi:
                    exp_result = str(self.mi_loss_lambda) + ' ' + str(np.mean(norm_leak_auc_history)) + ' AUC ' + str(norm_leak_auc_history) + ' ' + str(np.max(norm_leak_auc_history))
                    exp_result1 = str(self.mi_loss_lambda) + ' ' + str(np.mean(cosine_leak_auc_history)) + ' AUC ' + str(cosine_leak_auc_history) + ' ' + str(np.max(cosine_leak_auc_history))
                elif self.apply_RRwithPrior:
                    exp_result = str(self.RRwithPrior_epsilon) + ' ' + str(np.mean(norm_leak_auc_history)) + ' AUC ' + str(norm_leak_auc_history) + ' ' + str(np.max(norm_leak_auc_history))
                    exp_result1 = str(self.RRwithPrior_epsilon) + ' ' + str(np.mean(cosine_leak_auc_history)) + ' AUC ' + str(cosine_leak_auc_history) + ' ' + str(np.max(cosine_leak_auc_history))
                elif self.apply_distance_correlation:
                    exp_result = str(self.distance_correlation_lambda) + ' ' + str(np.mean(norm_leak_auc_history)) + ' AUC ' + str(norm_leak_auc_history) + ' ' + str(np.max(norm_leak_auc_history))
                    exp_result1 = str(self.distance_correlation_lambda) + ' ' + str(np.mean(cosine_leak_auc_history)) + ' AUC ' + str(cosine_leak_auc_history) + ' ' + str(np.max(cosine_leak_auc_history))
                elif self.apply_grad_perturb:
                    exp_result = str(self.perturb_epsilon) + ' ' + str(np.mean(norm_leak_auc_history)) + ' AUC ' + str(norm_leak_auc_history) + ' ' + str(np.max(norm_leak_auc_history))
                    exp_result1 = str(self.perturb_epsilon) + ' ' + str(np.mean(cosine_leak_auc_history)) + ' AUC ' + str(cosine_leak_auc_history) + ' ' + str(np.max(cosine_leak_auc_history))
                elif self.apply_marvell:
                    exp_result = str(self.marvell_s) + ' ' + str(np.mean(norm_leak_auc_history)) + ' AUC ' + str(norm_leak_auc_history) + ' ' + str(np.max(norm_leak_auc_history))
                    exp_result1 = str(self.marvell_s) + ' ' + str(np.mean(cosine_leak_auc_history)) + ' AUC ' + str(cosine_leak_auc_history) + ' ' + str(np.max(cosine_leak_auc_history))
                else:
                    exp_result = f"bs|num_class|recovery_rate,%d|%d| %lf %s %s %lf" % ((batch_size), (num_classes), (np.mean(norm_leak_auc_history)), 'AUC', str(norm_leak_auc_history), (np.max(norm_leak_auc_history)))
                    exp_result1 = f"bs|num_class|recovery_rate,%d|%d| %lf %s %s %lf" % ((batch_size), (num_classes), (np.mean(cosine_leak_auc_history)), 'AUC', str(cosine_leak_auc_history), (np.max(cosine_leak_auc_history)))

                append_exp_res(self.exp_res_path[0], exp_result)
                append_exp_res(self.exp_res_path[1], exp_result1)
                print(self.exp_res_path[0], self.exp_res_path[1])
                print(exp_result)
                print(exp_result1)

                if self.apply_laplace or self.apply_gaussian:
                    exp_result = str(self.dp_strength) + ' ' + str(np.mean(norm_leak_acc_history)) + ' ACC ' + str(norm_leak_acc_history) + ' ' + str(np.max(norm_leak_acc_history))
                    exp_result1 = str(self.dp_strength) + ' ' + str(np.mean(cosine_leak_acc_history)) + ' ACC ' + str(cosine_leak_acc_history) + ' ' + str(np.max(cosine_leak_acc_history))
                elif self.apply_grad_spar:
                    exp_result = str(self.grad_spars) + ' ' + str(np.mean(norm_leak_acc_history)) + ' ACC ' + str(norm_leak_acc_history) + ' ' + str(np.max(norm_leak_acc_history))
                    exp_result1 = str(self.grad_spars) + ' ' + str(np.mean(cosine_leak_acc_history)) + ' ACC ' + str(cosine_leak_acc_history) + ' ' + str(np.max(cosine_leak_acc_history))
                # elif self.apply_encoder or self.apply_adversarial_encoder:
                #     exp_result = str(self.ae_lambda) + ' ' + str(np.mean(norm_leak_acc_history)) + ' ACC ' + str(norm_leak_acc_history) + ' ' + str(np.max(norm_leak_acc_history))
                #     exp_result1 = str(self.ae_lambda) + ' ' + str(np.mean(cosine_leak_acc_history)) + ' ACC ' + str(cosine_leak_acc_history) + ' ' + str(np.max(cosine_leak_acc_history))
                # elif self.apply_ppdl:
                #     exp_result = str(self.ppdl_theta_u) + ' ' + str(np.mean(norm_leak_acc_history)) + ' ACC ' + str(norm_leak_acc_history) + ' ' + str(np.max(norm_leak_acc_history))
                #     exp_result1 = str(self.ppdl_theta_u) + ' ' + str(np.mean(cosine_leak_acc_history)) + ' ACC ' + str(cosine_leak_acc_history) + ' ' + str(np.max(cosine_leak_acc_history))
                # elif self.apply_gc:
                #     exp_result = str(self.gc_preserved_percent) + ' ' + str(np.mean(norm_leak_acc_history)) + ' ACC ' + str(norm_leak_acc_history) + ' ' + str(np.max(norm_leak_acc_history))
                #     exp_result1 = str(self.gc_preserved_percent) + ' ' + str(np.mean(cosine_leak_acc_history)) + ' ACC ' + str(cosine_leak_acc_history) + ' ' + str(np.max(cosine_leak_acc_history))
                # elif self.apply_lap_noise:
                #     exp_result = str(self.noise_scale) + ' ' + str(np.mean(norm_leak_acc_history)) + ' ACC ' + str(norm_leak_acc_history) + ' ' + str(np.max(norm_leak_acc_history))
                #     exp_result1 = str(self.noise_scale) + ' ' + str(np.mean(cosine_leak_acc_history)) + ' ACC ' + str(cosine_leak_acc_history) + ' ' + str(np.max(cosine_leak_acc_history))
                elif self.apply_encoder:
                    exp_result = str(self.ae_lambda) + ' ' + str(np.mean(norm_leak_acc_history)) + ' ACC ' + str(norm_leak_acc_history) + ' ' + str(np.max(norm_leak_acc_history))
                    exp_result1 = str(self.ae_lambda) + ' ' + str(np.mean(cosine_leak_acc_history)) + ' ACC ' + str(cosine_leak_acc_history) + ' ' + str(np.max(cosine_leak_acc_history))
                elif self.apply_discrete_gradients:
                    exp_result = str(self.discrete_gradients_bins) + ' ' + str(np.mean(norm_leak_acc_history)) + ' ACC ' + str(norm_leak_acc_history) + ' ' + str(np.max(norm_leak_acc_history))
                    exp_result1 = str(self.discrete_gradients_bins) + ' ' + str(np.mean(cosine_leak_acc_history)) + ' ACC ' + str(cosine_leak_acc_history) + ' ' + str(np.max(cosine_leak_acc_history))
                elif self.apply_mid:
                    exp_result = str(self.mid_loss_lambda) + ' ' + str(np.mean(norm_leak_acc_history)) + ' ACC ' + str(norm_leak_acc_history) + ' ' + str(np.max(norm_leak_acc_history))
                    exp_result1 = str(self.mid_loss_lambda) + ' ' + str(np.mean(cosine_leak_acc_history)) + ' ACC ' + str(cosine_leak_acc_history) + ' ' + str(np.max(cosine_leak_acc_history))
                elif self.apply_mi:
                    exp_result = str(self.mi_loss_lambda) + ' ' + str(np.mean(norm_leak_acc_history)) + ' ACC ' + str(norm_leak_acc_history) + ' ' + str(np.max(norm_leak_acc_history))
                    exp_result1 = str(self.mi_loss_lambda) + ' ' + str(np.mean(cosine_leak_acc_history)) + ' ACC ' + str(cosine_leak_acc_history) + ' ' + str(np.max(cosine_leak_acc_history))
                elif self.apply_RRwithPrior:
                    exp_result = str(self.RRwithPrior_epsilon) + ' ' + str(np.mean(norm_leak_acc_history)) + ' ACC ' + str(norm_leak_acc_history) + ' ' + str(np.max(norm_leak_acc_history))
                    exp_result1 = str(self.RRwithPrior_epsilon) + ' ' + str(np.mean(cosine_leak_acc_history)) + ' ACC ' + str(cosine_leak_acc_history) + ' ' + str(np.max(cosine_leak_acc_history))
                elif self.apply_distance_correlation:
                    exp_result = str(self.distance_correlation_lambda) + ' ' + str(np.mean(norm_leak_acc_history)) + ' ACC ' + str(norm_leak_acc_history) + ' ' + str(np.max(norm_leak_acc_history))
                    exp_result1 = str(self.distance_correlation_lambda) + ' ' + str(np.mean(cosine_leak_acc_history)) + ' ACC ' + str(cosine_leak_acc_history) + ' ' + str(np.max(cosine_leak_acc_history))
                elif self.apply_grad_perturb:
                    exp_result = str(self.perturb_epsilon) + ' ' + str(np.mean(norm_leak_acc_history)) + ' ACC ' + str(norm_leak_acc_history) + ' ' + str(np.max(norm_leak_acc_history))
                    exp_result1 = str(self.perturb_epsilon) + ' ' + str(np.mean(cosine_leak_acc_history)) + ' ACC ' + str(cosine_leak_acc_history) + ' ' + str(np.max(cosine_leak_acc_history))
                elif self.apply_marvell:
                    exp_result = str(self.marvell_s) + ' ' + str(np.mean(norm_leak_acc_history)) + ' ACC ' + str(norm_leak_acc_history) + ' ' + str(np.max(norm_leak_acc_history))
                    exp_result1 = str(self.marvell_s) + ' ' + str(np.mean(cosine_leak_acc_history)) + ' ACC ' + str(cosine_leak_acc_history) + ' ' + str(np.max(cosine_leak_acc_history))
                else:
                    exp_result = f"bs|num_class|recovery_rate,%d|%d| %lf %s %s %lf" % ((batch_size), (num_classes), (np.mean(norm_leak_acc_history)), 'ACC', str(norm_leak_acc_history), (np.max(norm_leak_acc_history)))
                    exp_result1 = f"bs|num_class|recovery_rate,%d|%d| %lf %s %s %lf" % ((batch_size), (num_classes), (np.mean(cosine_leak_acc_history)), 'ACC', str(cosine_leak_acc_history), (np.max(cosine_leak_acc_history)))

                append_exp_res(self.exp_res_path[0], exp_result)
                append_exp_res(self.exp_res_path[1], exp_result1)
                print(exp_result)
                print(exp_result1)




# def tf_train(model, train_set, test_set, loss_function, num_epochs, writer,
#           trainer=None, regularization_weight=0.1, period=None, num_hints=None):
#     """[summary]

#     Args:
#         model ([type]): [description]
#         train_set ([type]): [description]
#         test_set ([type]): [description]
#         loss_function ([type]): [description]
#         num_epochs ([type]): [description]
#         writer ([type]): [description]
#         trainer ([type], optional): [description]. Defaults to None.
#         regularization_weight (float, optional): [description]. Defaults to 0.1.
#         period ([type], optional): [description]. Defaults to None.
#         num_hints (list, optional): specifies how many hints to use for the hint attack.
#                             Defaults to None which means no evaluation using hint

#     Returns:
#         [type]: [description]
#     """
#     best_test_auc = 0
#     best_epoch = 0

#     train_loss = tf.keras.metrics.Mean()
#     train_accu = tf.keras.metrics.BinaryAccuracy()
#     train_auc = tf.keras.metrics.AUC()

#     # p_norm = tf.keras.metrics.Mean()
#     # n_norm = tf.keras.metrics.Mean()
#     norm_leak_auc_dict = model.leak_auc_dict(attack_method='norm_leak')
#     ip_leak_auc_dict = model.leak_auc_dict(attack_method='ip_leak')
#     cosine_leak_auc_dict = model.leak_auc_dict(attack_method='cosine_leak')
#     if num_hints is not None:
#         hint_attack_norm_leak_auc_dicts = {}
#         hint_attack_ip_leak_auc_dicts = {}
#         for n_hint in num_hints:
#             hint_attack_norm_leak_auc_dicts[n_hint] = model.leak_auc_dict(attack_method=f"{n_hint}hint_norm")
#             hint_attack_ip_leak_auc_dicts[n_hint] = model.leak_auc_dict(attack_method=f"{n_hint}hint_inner_product")

#     # global_batch_idx = 0

#     # @tf.function
#     def train_step(model, X, y):
#         """[the feedforward and backprop for one training update
#             tf.function constructs the computation graph and avoids memory leaks?
#             defined inside avoids passing trainer as an argument]

#         Args:
#             model (tf.keras.Model): the model to feedforward and backprop
#             X ([type]): input
#             y ([type]): target output

#         Returns:
#             loss, logits, activation_grad_list
#         """    
#         with tf.GradientTape(persistent=False) as tape:
#             # start = time.time()
#             activations_by_layer = model(X, no_noise=False)
#             # model.save_weights('model.ckpt')
#             # print(model.num_params())
#             # print(time.time() - start)

#             logits = activations_by_layer[-1]
#             loss = tf.math.reduce_mean(loss_function(y_hat=logits, y=y)) + regularization_weight * model.regularization_losses()
        
#         params = model.trainable_variables
#         grad_list = tape.gradient(loss, params + activations_by_layer)

#         # apply grad to only the parameters but not the activations
#         parameter_grad_list = grad_list[:len(params)]
#         activation_grad_list = grad_list[len(params):]
#         # print('logit gradient')
#         # print(activation_grad_list[-1])
#         # print('pos_prob')
#         # print(tf.math.sigmoid(logits))
#         trainer.apply_gradients(zip(parameter_grad_list, params))

#         return loss, logits, activation_grad_list

#     for epoch in range(num_epochs):
#         print("epoch {}:".format(epoch))

#         # gradients_over_training_set = defaultdict(list)
#         # labels_over_training_set = []

#         e_s = datetime.datetime.now()

#         for (batch_idx, (X, y)) in enumerate(train_set, 1):

#             # print('number of positive examples', tf.math.reduce_sum(y))
#             if tf.math.reduce_sum(y).numpy() == 0:
#                 # if the batch has no positive examples, continue
#                 continue


#             # store the batch label in shared_var for the custom gradient noise_layer_function to access
#             shared_var.batch_y = y
#             # global_batch_idx += 1
#             b_s = datetime.datetime.now()

#             ###########################################################
#             ######## preparation for direction attack begins ##########
#             ###########################################################
#             # get a positive example now that there is at least one positive example in the batch
#             # do forward and backward on this single variable, store the grad_activation_list in the shared_var
#             pos_idx = np.random.choice(np.where(y.numpy() == 1)[0], size=1)[0]
#             with tf.GradientTape(persistent=False) as tape:
#                 if isinstance(X, OrderedDict):
#                     pos_X = {
#                         key: value[pos_idx:pos_idx+1] for key, value in X.items()
#                     }
#                 elif isinstance(X, tf.Tensor):
#                     pos_X = X[pos_idx: pos_idx+1]
#                 else:
#                     assert False, 'unsupported X type'

#                 pos_activations_by_layer = model(pos_X, no_noise=True)
#                 pos_logits = pos_activations_by_layer[-1]
#                 pos_loss = tf.math.reduce_mean(loss_function(y_hat=pos_logits, y=y[pos_idx:pos_idx+1])) + regularization_weight * model.regularization_losses()
#             pos_activation_grad_list = tape.gradient(pos_loss, pos_activations_by_layer)

#             ###########################################################
#             ######## preparation for direction attack ends ############
#             ###########################################################


#             # start = time.time()
#             loss, logits, activation_grad_list = train_step(model, X, y)

#             # save these for generating the plots
#             # if shared_var.counter < 2000:
#             #     for layer_name, grad in zip(model.layer_names, activation_grad_list):
#             #         np.save(file=os.path.join(shared_var.logdir, layer_name + '_' + 'itr' + str(shared_var.counter) + '.npy'),
#             #                     arr=grad.numpy())
#             #     np.save(file=os.path.join(shared_var.logdir, 'y' + '_' + 'itr' + str(shared_var.counter) + '.npy'),
#             #             arr=y.numpy())

#             # print(time.time() - start)

#             # update training statistics
#             start = time.time()
#             train_loss.update_state(loss.numpy())
#             train_accu.update_state(y_true=y, 
#                                     y_pred=tf.math.sigmoid(logits))
#             train_auc.update_state(tf.reshape(y, [-1, 1]), tf.reshape(tf.math.sigmoid(logits), [-1, 1]))

#             layer_idx = model.config.index('noise_layer')

#             # p_norm.update_state(tf.norm(activation_grad_list[layer_idx][y==1], axis=1))
#             # n_norm.update_state(tf.norm(activation_grad_list[layer_idx][y==0], axis=1))
#             update_all_norm_leak_auc(
#                 norm_leak_auc_dict=norm_leak_auc_dict,
#                 grad_list=activation_grad_list,
#                 y=y)
#             update_all_ip_leak_auc(
#                 ip_leak_auc_dict=ip_leak_auc_dict,
#                 grad_list=activation_grad_list,
#                 pos_grad_list=pos_activation_grad_list,
#                 y=y)
#             update_all_cosine_leak_auc(
#                 cosine_leak_auc_dict=cosine_leak_auc_dict,
#                 grad_list=activation_grad_list,
#                 pos_grad_list=pos_activation_grad_list,
#                 y=y)

#             if num_hints is not None:
#                 for n_hint in num_hints:
#                     update_all_hint_norm_attack_leak_auc(
#                         hint_attack_norm_leak_auc_dicts[n_hint],
#                         activation_grad_list,
#                         y,
#                         num_hints=int(n_hint))
#                     update_all_hint_inner_product_attack_leak_auc(
#                         hint_attack_ip_leak_auc_dicts[n_hint],
#                         activation_grad_list,
#                         y,
#                         num_hints=int(n_hint))


#             # records the gradient for each layer over this batch
#             # for layer_name, layer_grad in zip(model.layer_names, activation_grad_list):
#             #     gradients_over_training_set[layer_name].append(layer_grad)
#             # record the labels
#             # labels_over_training_set.append(y)

#             with writer.as_default():
#                 tf.summary.scalar(name='p_norm_mean',
#                                   data=tf.reduce_mean(tf.norm(activation_grad_list[layer_idx][y==1], axis=1, keepdims=False)),
#                                   step=shared_var.counter)
#                                 #   step=global_batch_idx)
#                 tf.summary.scalar(name='n_norm_mean',
#                                   data=tf.reduce_mean(tf.norm(activation_grad_list[layer_idx][y==0], axis=1, keepdims=False)),
#                                   step=shared_var.counter)
#                                 #   step=global_batch_idx)
#             # print('logging', time.time() - start)
            

#             # clear out memory manually
#             # del params # is this correct?
#             # del loss
#             # del activations_by_layer
#             # del grad_list

#             '''
#             if epoch == num_epochs - 1:
#                 # checking the predicted probabilities after the last epoch's update
#                 batch_predicted_probability_positive_class = tf.math.sigmoid(model.predict(X))

#                 print("pos example pos prob: min: {:.4f}, mean: {:.4f}, max: {:.4f}".format(tf.reduce_min(batch_predicted_probability_positive_class[y==1]), 
#                                                                                             tf.reduce_mean(batch_predicted_probability_positive_class[y==1]),
#                                                                                             tf.reduce_max(batch_predicted_probability_positive_class[y==1])))
#                 print("neg example pos prob: min: {:.4f}, mean: {:.4f}, max: {:.4f}".format(tf.reduce_min(batch_predicted_probability_positive_class[y==0]),
#                                                                                            tf.reduce_mean(batch_predicted_probability_positive_class[y==0]),
#                                                                                            tf.reduce_max(batch_predicted_probability_positive_class[y==0])))
#                 print()
#             '''

#             b_e = datetime.datetime.now()
        
#             if tf.data.experimental.cardinality(train_set).numpy() == -2:
#                 # for make_csv_dataset or datasets that have used filter in general
#                 # where the total number of examples is unknown
#                 predicate = (batch_idx == 1) or (batch_idx % period == 0)
#             else:
#                 # for standard dataset
#                 predicate = batch_idx == len(train_set)

#             # print(tf.data.experimental.cardinality(train_set).numpy())
#             # print('predicate', predicate)
#             if predicate:
#                 # log statistics in terminal
            
#                 e_e = datetime.datetime.now()
#                 print("train loss: {:.4f}\ntrain accu: {:.4f}\ntrain auc: {:.4f}\ntime used: {}s\n".format(train_loss.result(), train_accu.result(), train_auc.result(), e_e - e_s))
#                 # print("pos norm: {:.4f}\nneg norm {:.4f}".format(p_norm.result(), n_norm.result()))
#                 print_all_leak_auc(leak_auc_dict=norm_leak_auc_dict)
#                 print_all_leak_auc(leak_auc_dict=ip_leak_auc_dict)
#                 print_all_leak_auc(leak_auc_dict=cosine_leak_auc_dict)
#                 if num_hints is not None:
#                     for n_hint in num_hints:
#                         print_all_leak_auc(leak_auc_dict=hint_attack_norm_leak_auc_dicts[n_hint])
#                         print_all_leak_auc(leak_auc_dict=hint_attack_ip_leak_auc_dicts[n_hint])

#                 # records the gradient for each layer over this batch
#                 # for layer_name, layer_grad in zip(model.layer_names, activation_grad_list):
#                 #     gradients_over_training_set[layer_name].append(layer_grad)
#                 # record the labels
#                 # labels_over_training_set.append(y)

#                 # concatenate batchs of gradients into one matrix for every layer
#                 # for layer_name in model.layer_names:
#                 #     gradients_over_training_set[layer_name] = tf.concat(gradients_over_training_set[layer_name], axis=0)
#                 # one array for the labels of the entire training set
#                 # labels_over_training_set = tf.concat(labels_over_training_set, axis=0)

#                 gradient_norm_by_layer = {}
#                 gradient_inner_product_by_layer = {}
#                 gradient_cosine_by_layer = {}

#                 for layer_name, layer_grad in zip(model.layer_names, activation_grad_list):
#                     gradient_norm_by_layer[layer_name] = compute_gradient_norm(layer_grad, y)
#                     gradient_inner_product_by_layer[layer_name] = compute_sampled_inner_product(layer_grad, y, sample_ratio=0.02)
#                     gradient_cosine_by_layer[layer_name] = compute_sampled_cosine(layer_grad, y, sample_ratio=0.02)

#                 test_loss, test_accu, test_auc = test(test_set=test_set,
#                                                     model=model, 
#                                                     loss_function=loss_function,
#                                                     regularization_weight=regularization_weight)

#                 # log statisitcs on tensorboard
#                 with writer.as_default():
#                     tf.summary.scalar('train_loss', train_loss.result(), step=shared_var.counter)
#                     tf.summary.scalar('train_auc',  train_auc.result(), step=shared_var.counter)
#                     tf.summary.scalar('train_accu', train_accu.result(), step=shared_var.counter)
#                     tf_summary_all_leak_auc(norm_leak_auc_dict, step=shared_var.counter)
#                     tf_summary_all_leak_auc(ip_leak_auc_dict, step=shared_var.counter)
#                     tf_summary_all_leak_auc(cosine_leak_auc_dict, step=shared_var.counter)
#                     if num_hints is not None:
#                         for n_hint in num_hints:
#                             tf_summary_all_leak_auc(hint_attack_norm_leak_auc_dicts[n_hint], step=shared_var.counter)
#                             tf_summary_all_leak_auc(hint_attack_ip_leak_auc_dicts[n_hint], step=shared_var.counter)

#                     tf.summary.scalar('test_loss', test_loss, step=shared_var.counter)
#                     tf.summary.scalar('test_accu', test_accu, step=shared_var.counter)
#                     tf.summary.scalar('test_auc', test_auc, step=shared_var.counter)

#                     for layer_name, info in gradient_norm_by_layer.items():
#                         for name, item in info.items():
#                             tf.summary.histogram(layer_name+'_norm/'+name, item, step=shared_var.counter)

#                         '''
#                         if shared_var.counter < 5000:
#                             for name, item in info.items():
#                                 if name in ['pos_grad_norm', 'neg_grad_norm']:
#                                     np.save(file=os.path.join(shared_var.logdir, layer_name + '_' + name + '_' + 'batch' + str(batch_idx)),
#                                             arr=item.numpy())
#                         '''


#                     for layer_name, info in gradient_inner_product_by_layer.items():
#                         for name, item in info.items():
#                             tf.summary.histogram(layer_name+'_inner_product/'+name, item, step=shared_var.counter)
#                     for layer_name, info in gradient_cosine_by_layer.items():
#                         if 'logits' not in layer_name:
#                             for name, item in info.items():
#                                 tf.summary.histogram(layer_name+'_cosine/'+name, item, step=shared_var.counter)

#                 if test_auc > best_test_auc:
#                     best_test_auc = max(test_auc, best_test_auc)
#                     best_epoch = epoch, batch_idx, shared_var.counter
#                     print("current best test auc: {:.4f}".format(best_test_auc))
#                     print("current best model: {:d}".format(shared_var.counter))
                
#                 ################################################
#                 ############ reset all the statistics ##########
#                 ################################################
#                 train_loss.reset_states()
#                 train_accu.reset_states()
#                 train_auc.reset_states()

#                 # p_norm.reset_states()
#                 # n_norm.reset_states()
#                 reset_all_leak_auc(norm_leak_auc_dict)
#                 reset_all_leak_auc(ip_leak_auc_dict)
#                 reset_all_leak_auc(cosine_leak_auc_dict)
#                 if num_hints is not None:
#                     for n_hint in num_hints:
#                         reset_all_leak_auc(hint_attack_norm_leak_auc_dicts[n_hint])
#                         reset_all_leak_auc(hint_attack_ip_leak_auc_dicts[n_hint])

#                 e_s = datetime.datetime.now()

#             shared_var.counter += 1 # count how many batches (iterations) done so far. 
#         print()

#     print("best test auc: {:.4f}".format(best_test_auc))
#     print("best epoch: ", best_epoch)


# def tf_test(test_set, model, loss_function, regularization_weight):
#     test_loss = tf.keras.metrics.Mean()
#     test_accu = tf.keras.metrics.BinaryAccuracy()
#     test_auc = tf.keras.metrics.AUC()

#     start = time.time()
#     for (idx, (X, y)) in enumerate(test_set):
#         logits = model.predict(X)
#         loss = tf.math.reduce_mean(loss_function(y_hat=logits, y=y)) + regularization_weight * model.regularization_losses()
#         test_loss.update_state(loss.numpy())
#         y = tf.cast(y, dtype=tf.float32)
#         test_accu.update_state(y_true=tf.reshape(y, [-1, 1]), 
#                                y_pred=tf.math.sigmoid(logits))
#         test_auc.update_state(tf.reshape(y, [-1,1]), tf.math.sigmoid(logits))
#     end = time.time()
#     print('test takes {}s'.format(end - start))

#     print("test loss: {:4f}\ntest accu: {:4f}\ntest auc: {:4f}".format(test_loss.result(), test_accu.result(), test_auc.result()))

#     return test_loss.result(), test_accu.result(), test_auc.result()