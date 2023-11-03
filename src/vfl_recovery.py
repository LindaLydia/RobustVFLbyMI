import logging
import pprint
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

import tensorflow as tf
import math
import copy

from marvell_model import (
    update_all_norm_leak_auc,
    update_all_cosine_leak_auc,
    KL_gradient_perturb
)
import marvell_shared_values as shared_var
BOTTLENECK_SCALE = 1

from models.vision import *
from utils import *

tf.compat.v1.enable_eager_execution() 


class FeatureRecovery(object):

    def __init__(self, args):
        self.device = args.device
        self.dataset_name = args.dataset_name
        self.train_dataset = args.train_dataset
        self.val_dataset = args.val_dataset
        self.half_dim = args.half_dim
        self.num_classes = args.num_classes

        self.apply_trainable_layer = args.apply_trainable_layer
        self.apply_laplace = args.apply_laplace
        self.apply_gaussian = args.apply_gaussian
        self.dp_strength = args.dp_strength
        self.apply_grad_spar = args.apply_grad_spar
        self.grad_spars = args.grad_spars
        self.apply_encoder = args.apply_encoder
        # self.apply_random_encoder = args.apply_random_encoder
        # self.apply_adversarial_encoder = args.apply_adversarial_encoder
        self.encoder = args.encoder
        self.gradients_res_a = None
        self.gradients_res_b = None
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
        self.epochs = args.epochs
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.models_dict = args.models_dict
        
        self.apply_mi = args.apply_mi
        self.mi_loss_lambda = args.mi_loss_lambda
        self.apply_mid = args.apply_mid
        self.mid_tau = args.mid_tau
        self.mid_loss_lambda = args.mid_loss_lambda
        self.mid_model = args.mid_model
        self.mid_enlarge_model = args.mid_enlarge_model
        self.apply_distance_correlation = args.apply_distance_correlation
        self.distance_correlation_lambda = args.distance_correlation_lambda
        self.apply_grad_perturb = args.apply_grad_perturb
        self.perturb_epsilon = args.perturb_epsilon
        self.apply_RRwithPrior = args.apply_RRwithPrior
        self.RRwithPrior_epsilon = args.RRwithPrior_epsilon      

        self.apply_dravl = args.apply_dravl
        self.dravl_w = args.dravl_w       

        self.path = args.path
        self.unknownVarLambda = args.unknownVarLambda  
        self.losses = []

    def fetch_parties_data(self, data):
        if self.dataset_name == 'nuswide':
            data_a = data[0]
            data_b = data[1]
        elif self.dataset_name == 'credit':
            data_a = data[:,:self.half_dim[0]]
            data_b = data[:,self.half_dim[0]:]
        else:
            data_a = data[:, :, :self.half_dim, :]
            data_b = data[:, :, self.half_dim:, :]
        return data_a.to(self.device), data_b.to(self.device)

    def build_models(self, num_classes):
        if self.dataset_name == 'cifar100' or self.dataset_name == 'cifar10':
            net_a = self.models_dict[self.dataset_name](num_classes).to(self.device)
            net_b = self.models_dict[self.dataset_name](num_classes).to(self.device)
        elif self.dataset_name == 'mnist':
            net_a = self.models_dict[self.dataset_name](self.half_dim * self.half_dim * 2, num_classes).to(self.device)
            net_b = self.models_dict[self.dataset_name](self.half_dim * self.half_dim * 2, num_classes).to(self.device)
        elif self.dataset_name == 'nuswide':
            net_a = self.models_dict[self.dataset_name](self.half_dim[0], num_classes).to(self.device)
            net_b = self.models_dict[self.dataset_name](self.half_dim[1], num_classes).to(self.device)
        elif self.dataset_name == 'credit':
            net_a = self.models_dict[self.dataset_name](self.half_dim[0], num_classes).to(self.device)
            net_b = self.models_dict[self.dataset_name](self.half_dim[1], num_classes).to(self.device)
            # self.net_total = self.models_dict[self.dataset_name](self.half_dim[0]+self.half_dim[1], num_classes).to(self.device)
        return net_a, net_b

    def label_to_one_hot(self, target, num_classes=10):
        target = torch.unsqueeze(target, 1).to(self.device)
        onehot_target = torch.zeros(target.size(0), num_classes, device=self.device)
        onehot_target.scatter_(1, target, 1)
        return onehot_target

    def get_loader(self, dst, batch_size):
        # return torch.utils.data.DataLoader(dst, batch_size=batch_size, sampler=train_sampler, num_workers=4)
        return DataLoader(dst, batch_size=batch_size)

    def get_random_softmax_onehot_label(self, gt_onehot_label):
        _random = torch.randn(gt_onehot_label.size()).to(self.device)
        for i in range(len(gt_onehot_label)):
            # print("random[i] and onehot[i]:", _random[i], "|", gt_onehot_label[i])
            max_index, = torch.where(_random[i] == _random[i].max())
            # print("max_index:", max_index)
            max_label, = torch.where(gt_onehot_label[i] == gt_onehot_label[i].max())
            while len(max_index) > 1:
                temp = torch.randn(gt_onehot_label[i].size()).to(self.device)
                # temp = torch.randn(gt_onehot_label[i].size())
                # print("temp:", temp)
                max_index, = torch.where(temp == temp.max())
                # print("max_index:", max_index)
                _random[i] = temp.clone()
            assert(len(max_label)==1)
            # print("max_label:", max_label)
            max_index = max_index.item()
            max_label = max_label.item()
            # print(max_index, max_label)
            if max_index != max_label:
                temp = _random[i][int(max_index)].clone()
                _random[i][int(max_index)] = _random[i][int(max_label)].clone()
                _random[i][int(max_label)] = temp.clone()
            _random[i] = F.softmax(_random[i], dim=-1)
            # print("after softmax: _random[i]", _random[i])
        return self.encoder(_random)
    
    def MSE_PSNR(self, batch_real_image, batch_dummy_image):
        '''
        compute MSE and PSNR
        :param batch_real_image:
        :param batch_dummy_image:
        :return:
        '''
        # print(batch_real_image.size(),batch_dummy_image.size())
        batch_real_image = batch_real_image.reshape(batch_dummy_image.size())
        mse = torch.mean((batch_real_image-batch_dummy_image)**2)
        psnr = 20 * torch.log10(1/torch.sqrt(mse))
        return mse.cpu().numpy(), psnr.cpu().numpy()
    
    def train_batch(self, batch_data_a, batch_data_b, batch_label, net_a, net_b, encoder, model_optimizer, criterion):
        
        self.optimizerG.zero_grad()
        
        gt_equal_probability = torch.from_numpy(np.array([1/self.num_classes]*self.num_classes)).to(self.device)
        if self.apply_encoder:
            if self.encoder:
                _, gt_one_hot_label = self.encoder(batch_label)
            else:
                assert(encoder != None)
        elif self.apply_grad_perturb:
            gt_one_hot_label = batch_label
            perturb_one_hot_label = label_perturb(batch_label, self.perturb_epsilon)
        elif self.apply_RRwithPrior:
            gt_one_hot_label = batch_label
            RRwP_one_hot_label = RRwithPrior(batch_label, self.RRwithPrior_epsilon, gt_equal_probability)
        else:
            gt_one_hot_label = batch_label

        # generate "fake inputs"
        noise_data_b = torch.randn(batch_data_b.size()).to(self.device) # attack from passive side, data_b is at active side need to be generated from noise at passive side
        # print(noise_data_b.size(), batch_data_a.size())
        # print(torch.cat((batch_data_a,noise_data_b),dim=2).size())
        if self.dataset_name == 'credit':
            generated_data_b = self.netG(torch.cat((batch_data_a,noise_data_b),dim=1))
        else:           
            generated_data_b = self.netG(torch.cat((batch_data_a,noise_data_b),dim=2))
            generated_data_b = generated_data_b.reshape(batch_data_b.size())
        # compute logits of generated data
        pred_a = net_a(batch_data_a)
        pred_b = net_b(generated_data_b)
        
        # aggregate logits of clients
        ######################## defense start ############################
        ######################## defense: trainable_layer(top_model) ############################
        pred = self.active_aggregate_model(pred_a, pred_b)
        # pred = self.net_total(torch.cat((batch_data_a,generated_data_b),dim=1))
        # loss = criterion(pred, gt_one_hot_label)
        # if self.apply_grad_perturb:
        #     loss_perturb_label = criterion(pred,perturb_one_hot_label)
        # if self.apply_RRwithPrior:
        #     loss_rr_label = criterion(pred,RRwP_one_hot_label)
        ######################## defense3: mid ############################
        if self.apply_mid:
            # print("mid in training (dummy calculation) for different prediction")
            # pred_Z = Somefunction(pred_a)
            # print("pred_a.size(): ",pred_a.size())
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
            # mu, std = norm.fit(pred_a.cpu().detach().numpy())
            mu, std = pred_a_double[:,:self.num_classes*BOTTLENECK_SCALE], pred_a_double[:,self.num_classes*BOTTLENECK_SCALE:]
            std = F.softplus(std-0.5) # ? F.softplus(std-5)
            # std = F.softplus(std-5) # ? F.softplus(std-5)
            # print("mu, std: ", mu.size(), std.size())
            pred_Z = mu+std*epsilon
            # assert(pred_Z.size()==pred_a.size())
            pred_Z = pred_Z.to(self.device)

            pred_Z = self.mid_model(pred_Z)
            # pred = self.active_aggregate_model(pred_Z, F.softmax(pred_b))
            pred = self.active_aggregate_model(pred_Z, pred_b)
            # # loss for discrete form of reparameterization
            # loss = criterion(pred, gt_one_hot_label) + self.mid_loss_lambda * entropy_for_probability_vector(pred_a)
            # loss for continuous form of reparameterization
            # loss = criterion(pred, gt_one_hot_label) + self.mid_loss_lambda * torch.mean(torch.sum((-0.5)*(1+2*torch.log(std)-mu**2 - std**2),1))
            # print("loss: ", loss)
        # ######################## defense: mi ############################
        # elif self.apply_mi:
        #     # loss = criterion(pred_b, gt_one_hot_label)
        #     # loss = criterion(pred, gt_one_hot_label) - self.mi_loss_lambda * criterion(pred_a, gt_one_hot_label) + self.mi_loss_lambda * criterion(pred_a, pred_a) # - criterion(pred,pred)
        #     # loss = criterion(pred, gt_one_hot_label) + self.mi_loss_lambda * criterion(pred_a, gt_equal_probability)
        #     loss = criterion(pred, gt_one_hot_label) + self.mi_loss_lambda * criterion(pred_a, gt_equal_probability) - self.mi_loss_lambda * criterion(pred_a, pred_a)
        #     # mu, std = norm.fit(pred.cpu().numpy())
        #     # loss = criterion(pred, gt_onehot_label) + 
        # ######################## defense4: distance correlation ############################
        # elif self.apply_distance_correlation:
        #     # loss = criterion(pred, gt_one_hot_label) + self.distance_correlation_lambda * torch.mean(torch.cdist(pred_a, gt_one_hot_label, p=2))
        #     loss = criterion(pred, gt_one_hot_label) + self.distance_correlation_lambda * torch.log(tf_distance_cov_cor(pred_a, gt_one_hot_label))


        # ground_truth of prediction
        # ====== normal vertical federated learning ======
        # compute logits of generated data
        ground_truth_pred_a = net_a(batch_data_a)
        ground_truth_pred_b = net_b(batch_data_b)
        # aggregate logits of clients
        ######################## defense start ############################
        ######################## defense: trainable_layer(top_model) ############################
        ground_truth_pred = self.active_aggregate_model(ground_truth_pred_a, ground_truth_pred_b)
        # ground_truth_pred = self.net_total(torch.cat((batch_data_a,batch_data_b),dim=1))
        # ground_truth_loss = criterion(ground_truth_pred, gt_one_hot_label)
        # if self.apply_grad_perturb:
        #     ground_truth_loss_perturb_label = criterion(ground_truth_pred,perturb_one_hot_label)
        # if self.apply_RRwithPrior:
        #     ground_truth_loss_rr_label = criterion(ground_truth_pred,RRwP_one_hot_label)
        ######################## defense3: mid ############################
        if self.apply_mid:
            # print("mid in training (ground_truth calculation) for different prediction")
            # pred_Z = Somefunction(pred_a)
            # print("pred_a.size(): ",pred_a.size())
            epsilon = torch.empty((ground_truth_pred_a.size()[0],ground_truth_pred_a.size()[1]*BOTTLENECK_SCALE))

            # continuous form of reparameterization
            torch.nn.init.normal(epsilon, mean=0, std=1) # epsilon is initialized
            epsilon = epsilon.to(self.device)
            # # pred_a.size() = (batch_size, class_num)
            ground_truth_pred_a_double = self.mid_enlarge_model(ground_truth_pred_a)
            # mu, std = norm.fit(pred_a.cpu().detach().numpy())
            mu, std = ground_truth_pred_a_double[:,:self.num_classes*BOTTLENECK_SCALE], ground_truth_pred_a_double[:,self.num_classes*BOTTLENECK_SCALE:]
            std = F.softplus(std-0.5) # ? F.softplus(std-5)
            # std = F.softplus(std-5) # ? F.softplus(std-5)
            # print("mu, std: ", mu.size(), std.size())
            ground_truth_pred_Z = mu+std*epsilon
            # assert(pred_Z.size()==pred_a.size())
            ground_truth_pred_Z = ground_truth_pred_Z.to(self.device)

            ground_truth_pred_Z = self.mid_model(ground_truth_pred_Z)
            # pred = self.active_aggregate_model(pred_Z, F.softmax(pred_b))
            ground_truth_pred = self.active_aggregate_model(ground_truth_pred_Z, ground_truth_pred_b)
            # # loss for discrete form of reparameterization
            # loss = criterion(pred, gt_one_hot_label) + self.mid_loss_lambda * entropy_for_probability_vector(pred_a)
            # loss for continuous form of reparameterization
            # ground_truth_loss = criterion(ground_truth_pred, gt_one_hot_label) + self.mid_loss_lambda * torch.mean(torch.sum((-0.5)*(1+2*torch.log(std)-mu**2 - std**2),1))
            # print("loss: ", loss)
        # ######################## defense: mi ############################
        # elif self.apply_mi:
        #     # loss = criterion(pred_b, gt_one_hot_label)
        #     # loss = criterion(pred, gt_one_hot_label) - self.mi_loss_lambda * criterion(pred_a, gt_one_hot_label) + self.mi_loss_lambda * criterion(pred_a, pred_a) # - criterion(pred,pred)
        #     # loss = criterion(pred, gt_one_hot_label) + self.mi_loss_lambda * criterion(pred_a, gt_equal_probability)
        #     ground_truth_loss = criterion(ground_truth_pred, gt_one_hot_label) + self.mi_loss_lambda * criterion(ground_truth_pred_a, gt_equal_probability) - self.mi_loss_lambda * criterion(ground_truth_pred_a, ground_truth_pred_a)
        #     # mu, std = norm.fit(pred.cpu().numpy())
        #     # loss = criterion(pred, gt_onehot_label) + 
        # ######################## defense4: distance correlation ############################
        # elif self.apply_distance_correlation:
        #     # loss = criterion(pred, gt_one_hot_label) + self.distance_correlation_lambda * torch.mean(torch.cdist(pred_a, gt_one_hot_label, p=2))
        #     ground_truth_loss = criterion(ground_truth_pred, gt_one_hot_label) + self.distance_correlation_lambda * torch.log(tf_distance_cov_cor(ground_truth_pred_a, gt_one_hot_label))
        
        unknown_var_loss = 0.0
        for i in range(generated_data_b.size(0)):
            unknown_var_loss = unknown_var_loss + (generated_data_b[i].var())     # var() unknown

        # print(unknown_var_loss, ((pred.detach() - ground_truth_pred.detach())**2).sum())
        # print((pred.detach() - ground_truth_pred.detach())[-5:])
        loss = (((F.softmax(pred,dim=-1) - F.softmax(ground_truth_pred,dim=-1))**2).sum() + self.unknownVarLambda * unknown_var_loss * 0.25)
        # _w = copy.deepcopy(list(self.netG.parameters()))
        # gradient = torch.autograd.grad(loss, self.netG.parameters(), retain_graph=True)
        # print(f"gradient for netG: {gradient}")
        loss.backward()
        self.losses.append(loss.detach())
        self.optimizerG.step() 

        # print(list(self.netG.parameters())[0]-_w[0])

        # predict_prob = F.softmax(pred, dim=-1)
        # suc_cnt = torch.sum(torch.argmax(predict_prob, dim=-1) == torch.argmax(batch_label, dim=-1)).item()
        # train_acc = suc_cnt / predict_prob.shape[0]
        # return loss.item(), train_acc
        return loss.item(), None

    def train(self):

        print("check dataset length", len(self.train_dataset),len(self.val_dataset))
        train_loader = self.get_loader(self.train_dataset, batch_size=self.batch_size)
        val_loader = self.get_loader(self.val_dataset, batch_size=self.batch_size)

        # for num_classes in self.num_class_list:
        # n_minibatches = len(train_loader)
        print_every = 1
        if self.dataset_name == 'cifar100' or self.dataset_name == 'cifar10':
            print_every = 1
        elif self.dataset_name == 'mnist':
            print_every = 1
        elif self.dataset_name == 'nuswide':
            print_every = 1
        elif self.dataset_name == 'credit':
            print_every = 1
        # net_a refers to passive model, net_b refers to active model
        net_a, net_b = self.build_models(self.num_classes)
        if self.apply_trainable_layer:
            print("use_trainable_layer")
            self.active_aggregate_model = ActivePartyWithTrainableLayer(hidden_dim=self.num_classes * 2, output_dim=self.num_classes)
        else:
            self.active_aggregate_model = ActivePartyWithoutTrainableLayer()
        self.active_aggregate_model = self.active_aggregate_model.to(self.device)
        if self.apply_mid:
            self.mid_model = self.mid_model.to(self.device)
            self.mid_enlarge_model = self.mid_enlarge_model.to(self.device)
        # load trained models
        if self.apply_trainable_layer:
            if self.apply_mid:
                models_params = torch.load(f"./saved_models/{self.dataset_name}/top/mid_{self.mid_loss_lambda}.pkl",map_location=self.device)
                net_a.load_state_dict(models_params[0])
                net_b.load_state_dict(models_params[1])
                self.active_aggregate_model.load_state_dict(models_params[2])
                self.mid_model.load_state_dict(models_params[3])
                self.mid_enlarge_model.load_state_dict(models_params[4])
                net_a.eval()
                net_b.eval()
                self.active_aggregate_model.eval()
                self.mid_model.eval()
                self.mid_enlarge_model.eval()
            else:
                if self.apply_gaussian:
                    models_params = torch.load(f"./saved_models/{self.dataset_name}/top/gaussian_{self.dp_strength}.pkl",map_location=self.device)
                if self.apply_dravl:
                    print(f"./saved_models/{self.dataset_name}/top/dravl_{self.dravl_w}.pkl",map_location=self.device)
                    models_params = torch.load(f"./saved_models/{self.dataset_name}/top/dravl_{self.dravl_w}.pkl",map_location=self.device)
                else:
                    models_params = torch.load(f"./saved_models/{self.dataset_name}/top/normal_{self.mid_loss_lambda}.pkl",map_location=self.device)
                net_a.load_state_dict(models_params[0])
                net_b.load_state_dict(models_params[1])
                self.active_aggregate_model.load_state_dict(models_params[2])
                net_a.eval()
                net_b.eval()
                self.active_aggregate_model.eval()
        else:
            if self.apply_mid:
                models_params = torch.load(f"./saved_models/{self.dataset_name}/no_top/mid_{self.mid_loss_lambda}.pkl",map_location=self.device)
                net_a.load_state_dict(models_params[0])
                net_b.load_state_dict(models_params[1])
                self.mid_model.load_state_dict(models_params[2])
                self.mid_enlarge_model.load_state_dict(models_params[3])
                net_a.eval()
                net_b.eval()
                self.mid_model.eval()
                self.mid_enlarge_model.eval()
            else:
                if self.apply_gaussian:
                    models_params = torch.load(f"./saved_models/{self.dataset_name}/no_top/gaussian_{self.dp_strength}.pkl",map_location=self.device)
                if self.apply_dravl:
                    print(f"./saved_models/{self.dataset_name}/no_top/dravl_{self.dravl_w}.pkl")
                    models_params = torch.load(f"./saved_models/{self.dataset_name}/no_top/dravl_{self.dravl_w}.pkl",map_location=self.device)
                else:
                    models_params = torch.load(f"./saved_models/{self.dataset_name}/no_top/normal_{self.mid_loss_lambda}.pkl",map_location=self.device)
                net_a.load_state_dict(models_params[0])
                net_b.load_state_dict(models_params[1])
                net_a.eval()
                net_b.eval()
        # new_model.forward(input)
        # model_params = torch.load(f"./saved_models/{self.dataset_name}/no_top/total_{self.mid_loss_lambda}.pkl",map_location=self.device)
        # self.net_total.load_state_dict(model_params)
        # self.net_total.eval()

        if self.dataset_name == 'credit':
            self.netG = Generator(self.half_dim[0]+self.half_dim[1], self.half_dim[1])
        elif self.dataset_name == 'cifar10' or self.dataset_name == 'cifar100':
            self.netG = Generator(self.half_dim*self.half_dim*2*2*3, self.half_dim*self.half_dim*2*3)
        else:
            self.netG = Generator(self.half_dim*self.half_dim*2*2, self.half_dim*self.half_dim*2)
        self.netG = self.netG.to(self.device)
        self.optimizerG = torch.optim.Adam(self.netG.parameters(), lr=self.lr)
        # if self.apply_trainable_layer:
        #     if self.apply_mid:
        #         model_optimizer = torch.optim.Adam(list(net_a.parameters()) + list(net_b.parameters()) + list(self.active_aggregate_model.parameters()) + list(self.mid_model.parameters()) + list(self.mid_enlarge_model.parameters()), lr=self.lr)
        #     else:
        #         model_optimizer = torch.optim.Adam(list(net_a.parameters()) + list(net_b.parameters()) + list(self.active_aggregate_model.parameters()), lr=self.lr)
        # else:
        #     if self.apply_mid:
        #         model_optimizer = torch.optim.Adam(list(net_a.parameters()) + list(net_b.parameters()) + list(self.mid_model.parameters()) + list(self.mid_enlarge_model.parameters()), lr=self.lr)
        #     else:
        #         model_optimizer = torch.optim.Adam(list(net_a.parameters()) + list(net_b.parameters()), lr=self.lr)
        criterion = nn.MSELoss()

        start_time = time.time()
        test_acc = 0.0
        test_acc_topk = 0.0
        for i_epoch in range(self.epochs):
            self.netG.train()
            tqdm_train = tqdm(train_loader, desc='Training Generator with testset (epoch #{})'.format(i_epoch + 1))
            # tqdm_train = tqdm(val_loader, desc='Training Generator with testset (epoch #{})'.format(i_epoch + 1))
            postfix = {'train_loss': 0.0, 'train_acc': 0.0, 'test_acc': 0.0}
            loss = 0.0
            for i, (gt_data, gt_label) in enumerate(tqdm_train):
                gt_data_a, gt_data_b = self.fetch_parties_data(gt_data)
                gt_one_hot_label = self.label_to_one_hot(gt_label, self.num_classes)
                gt_one_hot_label = gt_one_hot_label.to(self.device)
                # print('before batch, gt_one_hot_label:', gt_one_hot_label)
                # ====== train batch ======
                loss, train_acc = self.train_batch(gt_data_a, gt_data_b, gt_one_hot_label,
                                              net_a, net_b, self.encoder, self.optimizerG, criterion)
            
            # validation
            if (i_epoch + 1) % print_every == 0:
                # print("validate and test")
                self.netG.eval()
                suc_cnt = 0
                sample_cnt = 0

                MSE = []
                PSNR = []

                with torch.no_grad():
                    # enc_result_matrix = np.zeros((self.num_classes, self.num_classes), dtype=int)
                    # result_matrix = np.zeros((self.num_classes, self.num_classes), dtype=int)
                    # gt_val_data= torch.tensor(self.train_dataset.data, dtype=torch.float32)
                    # gt_val_label = torch.tensor(self.train_dataset.labels, dtype=torch.long)
                    gt_val_data= torch.tensor(self.val_dataset.data, dtype=torch.float32)
                    gt_val_label = torch.tensor(self.val_dataset.labels, dtype=torch.long)

                # for gt_val_data, gt_val_label in val_loader:
                    gt_val_one_hot_label = self.label_to_one_hot(gt_val_label, self.num_classes)
                    test_data_a, test_data_b = self.fetch_parties_data(gt_val_data)
                    noise_data_b = torch.randn(test_data_b.size()).to(self.device)

                    if self.dataset_name == 'credit':
                        generated_data_b = self.netG(torch.cat((test_data_a,noise_data_b),dim=1))
                    else:
                        generated_data_b = self.netG(torch.cat((test_data_a,noise_data_b),dim=2))

                    mse, psnr = self.MSE_PSNR(test_data_b, generated_data_b)
                    
                    MSE.append(mse)
                    PSNR.append(psnr)
                    # test_logit_a = net_a(test_data_a)
                    # test_logit_b = net_b(test_data_b)
                    # test_logit = self.active_aggregate_model(test_logit_a, test_logit_b)

                    # if self.apply_mid:
                    #     epsilon = torch.empty((test_logit_a.size()[0],test_logit_a.size()[1]*BOTTLENECK_SCALE))
                    #     # # discrete form of reparameterization
                    #     # torch.nn.init.uniform(epsilon) # epsilon is initialized
                    #     # epsilon = - torch.log(epsilon + torch.tensor(1e-07))
                    #     # epsilon = - torch.log(epsilon + torch.tensor(1e-07)) # prevent if epsilon=0.0
                    #     # test_logit_Z = F.softmax(test_logit_a) + epsilon.to(self.device)
                    #     # test_logit_Z = F.softmax(test_logit_Z / torch.tensor(self.mid_tau).to(self.device), -1)

                    #     # continuous form of reparameterization
                    #     torch.nn.init.normal(epsilon, mean=0, std=1) # epsilon is initialized
                    #     epsilon = epsilon.to(self.device)
                    #     # mu, std = norm.fit(test_logit_a.cpu().detach().numpy())
                    #     test_logit_a_double = self.mid_enlarge_model(test_logit_a)
                    #     mu, std = test_logit_a_double[:,:self.num_classes*BOTTLENECK_SCALE], test_logit_a_double[:,self.num_classes*BOTTLENECK_SCALE:]
                    #     std = F.softplus(std-0.5) # ? F.softplus(std-5)
                    #     # std = F.softplus(std-5) # ? F.softplus(std-5)
                    #     # print("mu, std: ", mu, std)
                    #     test_logit_Z = mu+std*epsilon
                    #     # assert(test_logit_Z.size()==test_logit_a.size())
                    #     test_logit_Z = test_logit_Z.to(self.device)

                    #     test_logit_Z = self.mid_model(test_logit_Z)
                    #     test_logit = self.active_aggregate_model(test_logit_Z, test_logit_b)
                    #     # test_logit = self.active_aggregate_model(test_logit_Z, F.softmax(test_logit_b,dim=-1))
                    #     # test_logit = self.active_aggregate_model(F.softmax(test_logit_Z,dim=-1), F.softmax(test_logit_b,dim=-1))

                    # enc_predict_prob = F.softmax(test_logit, dim=-1)
                    # if self.apply_encoder:
                    #     dec_predict_prob = self.encoder.decoder(enc_predict_prob)
                    #     predict_label = torch.argmax(dec_predict_prob, dim=-1)
                    # else:
                    #     predict_label = torch.argmax(enc_predict_prob, dim=-1)

                    # # enc_predict_label = torch.argmax(enc_predict_prob, dim=-1)
                    # actual_label = torch.argmax(gt_val_one_hot_label, dim=-1)
                    # sample_cnt += predict_label.shape[0]
                    # suc_cnt += torch.sum(predict_label == actual_label).item()
                    average_mse = np.mean(MSE)
                    average_psnr = np.mean(PSNR)

                    postfix['train_loss'] = loss
                    postfix['test_mse'] = '{:.2f}%'.format(average_mse)
                    postfix['test_psnr'] = '{:.2f}%'.format(average_psnr)
                    tqdm_train.set_postfix(postfix)
                    print('Epoch {}% \t train_loss:{:.2f} average_mse:{:.2f} average_psnr:{:.2f}'.format(
                        i_epoch, loss, average_mse, average_psnr))
        end_time = time.time()
        print(f"time used = {end_time-start_time}")

        _dir = self.path + 'recover_image/'
        if self.apply_laplace or self.apply_gaussian:
            _dir += str(self.dp_strength) + '/'
        elif self.apply_grad_spar:
            _dir += str(self.grad_spars) + '/'
        elif self.apply_discrete_gradients:
            _dir += str(self.discrete_gradients_bins) + '/'
        elif self.apply_mid:
            _dir += str(self.mid_loss_lambda) + '/'
        elif self.apply_mi:
            _dir += str(self.mi_loss_lambda) + '/'
        elif self.apply_RRwithPrior:
            _dir += str(self.RRwithPrior_epsilon) + '/'
        elif self.apply_distance_correlation:
            _dir += str(self.distance_correlation_lambda) + '/'
        elif self.apply_grad_perturb:
            _dir += str(self.perturb_epsilon) + '/'
        elif self.apply_encoder:
            _dir += str(self.ae_lambda) + '/'
        elif self.apply_marvell:
            _dir += str(self.marvell_s) + '/'
        if not os.path.exists(_dir):
            os.makedirs(_dir)
        

        data_a_to_be_save = test_data_a.cpu().numpy()
        data_b_to_be_save = generated_data_b.cpu().numpy()
        np.save(_dir + 'a_real_dummy.npy', data_a_to_be_save)
        np.save(_dir + 'b_generate_dummy.npy', data_b_to_be_save)
        np.save(_dir + 'label.npy', torch.argmax(gt_val_one_hot_label).cpu().numpy())

        # assert 1 == 0
        return average_psnr, average_mse


if __name__ == '__main__':
    pass