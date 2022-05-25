import logging
import pprint
import time

import tensorflow as tf

import torch
import torch.nn.functional as F
import numpy as np

from models.vision import *
from utils import *

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
        # self.apply_laplace = args.apply_laplace
        # self.apply_gaussian = args.apply_gaussian
        # self.dp_strength = args.dp_strength
        # self.apply_grad_spar = args.apply_grad_spar
        # self.grad_spars = args.grad_spars
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
        # self.apply_discrete_gradients = args.apply_discrete_gradients
        # self.discrete_gradients_bins = args.discrete_gradients_bins
        # self.discrete_gradients_bound = args.discrete_gradients_bound
        self.apply_mi = args.apply_mi
        self.mi_loss_lambda = args.mi_loss_lambda

        self.show_param()

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
            for line in os.listdir('./data/NUS_WIDE/Groundtruth/AllLabels'):
                all_nuswide_labels.append(line.split('_')[1][:-4])
        for batch_size in self.batch_size_list:
            for num_classes in self.num_class_list:
                classes = [None] * num_classes
                gt_equal_probability = torch.from_numpy(np.array([1/num_classes]*num_classes)).to(self.device)
                print("gt_equal_probability:", gt_equal_probability)
                if self.dataset == 'cifar100':
                    # if apply the defense, we only use cifar20
                    # if self.apply_laplace or self.apply_gaussian or self.apply_grad_spar:
                    #     classes = [i for i in range(num_classes)]
                    # else:
                    #     classes = random.sample(list(range(100)), num_classes)
                    classes = random.sample(list(range(100)), num_classes)
                    all_data, all_label = get_class_i(self.dst, classes)
                elif self.dataset == 'mnist':
                    classes = random.sample(list(range(10)), num_classes)
                    all_data, all_label = get_class_i(self.dst, classes)
                elif self.dataset == 'nuswide':
                    classes = random.sample(all_nuswide_labels, num_classes)
                    x_image, x_text, Y = get_labeled_data('./data/NUS_WIDE', classes, None, 'Train')
                elif self.dataset == 'cifar10':
                    classes = random.sample(list(range(10)), num_classes)
                    all_data, all_label = get_class_i(self.dst, classes)

                recovery_rate_history = []
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
                    if self.apply_encoder:
                        _, gt_onehot_label = self.get_random_softmax_onehot_label(gt_onehot_label)
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
                    pred_a = net_a(gt_data_a)
                    pred_b = net_b(gt_data_b)
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
                    ######################## defense2: mi ############################
                    if self.apply_mi:
                        # loss = criterion(pred, gt_onehot_label) - self.mi_loss_lambda * criterion(pred_a, gt_onehot_label) - criterion(pred,pred) + self.mi_loss_lambda * criterion(pred_a, pred_a)
                        # loss = criterion(pred, gt_onehot_label) + self.mi_loss_lambda * criterion(pred_a, gt_equal_probability)
                        loss = criterion(pred, gt_onehot_label) + self.mi_loss_lambda * criterion(pred_a, gt_equal_probability) - self.mi_loss_lambda * criterion(pred_a, pred_a)
                        # mu, std = norm.fit(pred.cpu().numpy())
                        # loss = criterion(pred, gt_onehot_label) + 
                    ######################## defense end ############################
                    pred_a_gradients = torch.autograd.grad(loss, pred_a, retain_graph=True)
                    pred_a_gradients_clone = pred_a_gradients[0].detach().clone()
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
                    # if self.apply_laplace or self.apply_gaussian:
                    #     print(f'batch_size=%d,class_num=%d,exp_id=%d,dp_strength=%lf,recovery_rate=%lf,time_used=%lf'
                    #           % (batch_size, num_classes, i_run, self.dp_strength,rec_rate, end_time - start_time))
                    # elif self.apply_grad_spar:
                    #     print(f'batch_size=%d,class_num=%d,exp_id=%d,grad_spars=%lf,recovery_rate=%lf,time_used=%lf'
                    #           % (batch_size, num_classes, i_run, self.grad_spars,rec_rate, end_time - start_time))
                    
                    if self.apply_mi:
                        print(f'batch_size=%d,class_num=%d,exp_id=%d,mi_loss_lambda=%lf,recovery_rate=%lf,time_used=%lf'
                              % (batch_size, num_classes, i_run, self.mi_loss_lambda,rec_rate, end_time - start_time))
                    elif self.apply_encoder:
                        print(f'batch_size=%d,class_num=%d,exp_id=%d,ae_lambda=%lf,recovery_rate=%lf,time_used=%lf'
                              % (batch_size, num_classes, i_run, self.ae_lambda,rec_rate, end_time - start_time))
                    elif self.apply_marvell:
                        print(f'batch_size=%d,class_num=%d,exp_id=%d,marvel_s=%lf,recovery_rate=%lf,time_used=%lf'
                              % (batch_size, num_classes, i_run, self.marvell_s,rec_rate, end_time - start_time))
                    else:
                        print(f'batch_size=%d,class_num=%d,exp_id=%d,recovery_rate=%lf,time_used=%lf'
                              % (batch_size, num_classes, i_run, rec_rate, end_time - start_time))
                avg_rec_rate = np.mean(recovery_rate_history)
                # if self.apply_laplace or self.apply_gaussian:
                #     exp_result = str(self.dp_strength) + ' ' + str(avg_rec_rate) + ' ' + str(recovery_rate_history) + ' ' + str(np.max(recovery_rate_history))
                # elif self.apply_grad_spar:
                #     exp_result = str(self.grad_spars) + ' ' + str(avg_rec_rate) + ' ' + str(recovery_rate_history) + ' ' + str(np.max(recovery_rate_history))
                # elif self.apply_encoder or self.apply_adversarial_encoder:
                #     exp_result = str(self.ae_lambda) + ' ' + str(avg_rec_rate) + ' ' + str(recovery_rate_history) + ' ' + str(np.max(recovery_rate_history))
                # elif self.apply_marvell:
                #     exp_result = str(self.marvell_s) + ' ' + str(avg_rec_rate) + ' ' + str(recovery_rate_history) + ' ' + str(np.max(recovery_rate_history))
                # elif self.apply_ppdl:
                #     exp_result = str(self.ppdl_theta_u) + ' ' + str(avg_rec_rate) + ' ' + str(recovery_rate_history) + ' ' + str(np.max(recovery_rate_history))
                # elif self.apply_gc:
                #     exp_result = str(self.gc_preserved_percent) + ' ' + str(avg_rec_rate) + ' ' + str(recovery_rate_history) + ' ' + str(np.max(recovery_rate_history))
                # elif self.apply_lap_noise:
                #     exp_result = str(self.noise_scale) + ' ' + str(avg_rec_rate) + ' ' + str(recovery_rate_history) + ' ' + str(np.max(recovery_rate_history))
                # elif self.apply_discrete_gradients:
                #     exp_result = str(self.discrete_gradients_bins) + ' ' + str(avg_rec_rate) + ' ' + str(recovery_rate_history) + ' ' + str(np.max(recovery_rate_history))
                if self.apply_mi:
                    exp_result = str(self.mi_loss_lambda) + ' ' + str(avg_rec_rate) + ' ' + str(recovery_rate_history) + ' ' + str(np.max(recovery_rate_history))
                else:
                    exp_result = f"bs|num_class|recovery_rate,%d|%d|%lf|%s|%lf" % (batch_size, num_classes, avg_rec_rate, str(recovery_rate_history), np.max(recovery_rate_history))

                append_exp_res(self.exp_res_path, exp_result)
                print(exp_result)

if __name__ == '__main__':
    pass