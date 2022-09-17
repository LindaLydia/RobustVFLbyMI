import os
from symbol import star_expr
import sys
# from itsdangerous import NoneAlgorithm
import numpy as np
import time

import random
import logging
import argparse
import torch.nn as nn
from torch.types import Device
import torch.utils
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
import utils
import copy

import torch
import torch.nn.functional as F

from dataset.mnist_dataset_vfl import MNISTDatasetVFL, need_poison_down_check_mnist_vfl
from dataset.nuswide_dataset_vfl import NUSWIDEDatasetVFL, need_poison_down_check_nuswide_vfl

from dataset.cifar10_dataset_vfl import Cifar10DatasetVFL, need_poison_down_check_cifar10_vfl
from dataset.cifar100_dataset_vfl import Cifar100DatasetVFL, Cifar100DatasetVFL20Classes, \
    need_poison_down_check_cifar100_vfl

from models.model_templates import ClassificationModelGuest, ClassificationModelHostHead, \
    ClassificationModelHostTrainableHead, ClassificationModelHostHeadWithSoftmax,\
    ClassificationModelHostHead_quarantine, \
    ClassificationModelHostTrainableHead_quarantine, ClassificationModelHostHeadWithSoftmax_quarantine,\
    SimpleCNN, LeNet5, MLP2
from models.resnet_torch import resnet18, resnet50
from models.vision import MID_enlarge_layer, MID_layer, RAE


def main():
    parser = argparse.ArgumentParser("backdoor")
    parser.add_argument('--dataset', type=str, default='mnist',
                        help='location of the data corpus')
    parser.add_argument('--name', type=str, default='test', help='experiment name')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='init learning rate')
    parser.add_argument('--momentum', type=float, default=0.0, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=3e-5, help='weight decay')
    parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
    parser.add_argument('--gamma', type=float, default=0.97, help='learning rate decay')
    parser.add_argument('--decay_period', type=int, default=1, help='epochs between two learning rate decays')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--workers', type=int, default=0, help='num of workers')
    parser.add_argument('--epochs', type=int, default=20, help='num of training epochs')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--k', type=int, default=3, help='num of client')
    parser.add_argument('--model', default='mlp2', help='resnet')
    parser.add_argument('--input_size', type=int, default=28, help='resnet')
    parser.add_argument('--use_project_head', type=int, default=0)
    parser.add_argument('--backdoor', type=int, default=0)
    parser.add_argument('--amplify_rate', type=float, default=10)
    parser.add_argument('--amplify_rate_output', type=float, default=1)
    parser.add_argument('--explicit_softmax', type=int, default=0)
    parser.add_argument('--random_output', type=int, default=0)
    parser.add_argument('--dp_type', type=str, default='none', help='[laplace, gaussian]')
    parser.add_argument('--dp_strength', type=float, default=0, help='[0.1, 0.075, 0.05, 0.025,...]')
    parser.add_argument('--gradient_sparsification', type=float, default=0)
    parser.add_argument('--writer', type=int, default=0)
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--defense_up', type=int, default=1)
    parser.add_argument("--certify", type=int, default=0, help="CertifyFLBaseline")
    parser.add_argument("--sigma", type=float, default=0, help='sigma for certify')
    parser.add_argument("--M", type=int, default=1000, help="voting party count in CertifyFL")
    parser.add_argument("--adversarial_start_epoch", type=int, default=0, help="adversarial_start")
    parser.add_argument("--certify_start_epoch", type=int, default=1, help="number of epoch when the cerfity ClipAndPerturb start")
    parser.add_argument('--autoencoder', type=int, default=0)
    parser.add_argument('--lba', type=float, default=0)
    parser.add_argument('--mid', type=int, default=0, help='whether to use Mutual Information Defense')
    parser.add_argument('--mid_lambda', type=float, default=0)
    parser.add_argument('--apply_discrete_gradients', default=False, type=bool, help='whether to use Discrete Gradients')
    parser.add_argument('--discrete_gradients_bins', default=0, type=int, help='number of bins for discrete gradients')
    parser.add_argument('--discrete_gradients_bound', default=3e-4, type=float, help='value of bound for discrete gradients')

    parser.add_argument('--backdoor_scale', type=float, default=1.0, help="the color of backdoor triger")

    parser.add_argument('--rvfr', type=int, default=1, help='whether to use robust feature subspace recovery')
    parser.add_argument('--quarantine_epochs', type=int, default=4, help='num of quarantine training epochs')
    parser.add_argument('--rae_pretrain_epochs', type=int, default=4, help='num of rae pre-training epochs')
    parser.add_argument('--rae_tune_epochs', type=int, default=1, help='num of rae tuning epochs')
    parser.add_argument('--rvfr_alpha', type=float, default=1.0)
    parser.add_argument('--rvfr_beta', type=float, default=0.1)
    parser.add_argument('--rvfr_lambda', type=float, default=0.1)


    args = parser.parse_args()

    args.name = 'experiment_rvfr_{}_{}/{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}'.format(
        args.rae_tune_epochs,
        args.name, args.epochs, args.dataset, args.model, args.batch_size, args.name,args.backdoor, args.amplify_rate,
        args.amplify_rate_output, args.dp_type, args.dp_strength, args.gradient_sparsification, args.certify, args.sigma, args.autoencoder, args.lba, args.mid, args.mid_lambda, args.apply_discrete_gradients, args.discrete_gradients_bins, args.rvfr, args.rvfr_alpha, args.seed,
        args.use_project_head, args.random_output, args.learning_rate, time.strftime("%Y%m%d-%H%M%S"))
    utils.create_exp_dir(args.name)

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.name, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    full_path = 'loss_log/{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}.txt'.format(
        args.epochs, args.dataset, args.model, args.batch_size, args.backdoor, args.backdoor_scale, args.amplify_rate,
        args.amplify_rate_output, args.dp_type, args.dp_strength, args.gradient_sparsification, args.certify, args.sigma, args.autoencoder, args.lba, args.mid, args.mid_lambda, args.seed,
        args.use_project_head, args.random_output, args.learning_rate, time.strftime("%Y%m%d-%H%M%S")) 
    file = open(full_path, 'w+')
    file.write("backdoor scale "+str(args.backdoor_scale))
    # file.write(msg) 
    # file.close()

    # tensorboard
    if args.writer == 1:
        writer = SummaryWriter(log_dir=os.path.join(args.name, 'tb'))
        writer.add_text('experiment', args.name, 0)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    args.device = device

    logging.info('***** USED DEVICE: {}'.format(device))

    # set seed for target label and poisoned and target sample selection
    manual_seed = 42
    random.seed(manual_seed)

    # set seed for model initialization
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        cudnn.benchmark = True
        cudnn.enabled = True
        torch.cuda.manual_seed_all(args.seed)
        logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)


    ##### set dataset
    input_dims = None

    if args.dataset == 'mnist':
        NUM_CLASSES = 10
        input_dims = [14 * 28, 14 * 28]
        args.input_size = 28
        DATA_DIR = './dataset/MNIST'
        # target_label = random.randint(0, NUM_CLASSES-1) # a class-id randomly created
        target_label = 5
        logging.info('target label: {}'.format(target_label))

        train_dataset = MNISTDatasetVFL(DATA_DIR, 'train', args.input_size, args.input_size, 600, 10, target_label, args.backdoor_scale)
        valid_dataset = MNISTDatasetVFL(DATA_DIR, 'test', args.input_size, args.input_size, 100, 10, target_label, args.backdoor_scale)

        # set poison_check function
        # need_poison_down_check = need_poison_down_check_mnist_vfl

    elif args.dataset == 'nuswide':
        NUM_CLASSES = 5
        input_dims = [634, 1000]
        DATA_DIR = './dataset/NUS_WIDE'

        target_label = 1 #random.sample([0,1,3,4], 1)[0]

        logging.info('target label: {}'.format(target_label))

        train_dataset = NUSWIDEDatasetVFL(DATA_DIR, 'train', 600, 10, target_label, args.backdoor_scale)
        valid_dataset = NUSWIDEDatasetVFL(DATA_DIR, 'test', 400, 10, target_label, args.backdoor_scale)

        # set poison_check function
        # need_poison_down_check = need_poison_down_check_nuswide_vfl

    elif args.dataset == 'cifar10':
        NUM_CLASSES = 10
        input_dims = [16 * 32, 16 * 32]
        args.input_size = 32
        # input_dims = [1536, 1536]
        # args.input_size = 66

        DATA_DIR = './dataset/cifar-10-batches-py'

        target_label = random.randint(0, NUM_CLASSES-1)
        logging.info('target label: {}'.format(target_label))

        train_dataset = Cifar10DatasetVFL(DATA_DIR, 'train', args.input_size, args.input_size, 500, 10, target_label, args.backdoor_scale)
        valid_dataset = Cifar10DatasetVFL(DATA_DIR, 'test', args.input_size, args.input_size, 100, 10, target_label, args.backdoor_scale)

        # set poison_check function
        # need_poison_down_check = need_poison_down_check_cifar10_vfl
    elif args.dataset == 'cifar100':
        NUM_CLASSES = 100
        input_dims = [16 * 32, 16 * 32]
        args.input_size = 32

        DATA_DIR = './dataset/cifar-100-python'

        target_label = random.randint(0, NUM_CLASSES-1)
        logging.info('target label: {}'.format(target_label))

        train_dataset = Cifar100DatasetVFL(DATA_DIR, 'train', args.input_size, args.input_size, 500, 10, target_label, args.backdoor_scale)
        valid_dataset = Cifar100DatasetVFL(DATA_DIR, 'test', args.input_size, args.input_size, 100, 10, target_label, args.backdoor_scale)

        # set poison_check function
        # need_poison_down_check = need_poison_down_check_cifar100_vfl
    elif args.dataset == 'cifar20':
        NUM_CLASSES = 20
        input_dims = [16 * 32, 16 * 32]
        args.input_size = 32

        DATA_DIR = './dataset/cifar-100-python'

        target_label = random.randint(0, NUM_CLASSES-1)
        logging.info('target label: {}'.format(target_label))

        train_dataset = Cifar100DatasetVFL20Classes(DATA_DIR, 'train', args.input_size, args.input_size, 200, 10, target_label, args.backdoor_scale)
        valid_dataset = Cifar100DatasetVFL20Classes(DATA_DIR, 'test', args.input_size, args.input_size, 20, 10, target_label, args.backdoor_scale)

        # set poison_check function
        # need_poison_down_check = need_poison_down_check_cifar100_vfl

    n_train = len(train_dataset)
    n_valid = len(valid_dataset)

    train_indices = list(range(n_train))
    valid_indices = list(range(n_valid))
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_indices)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               sampler=train_sampler,
                                               num_workers=args.workers,
                                               shuffle=False,
                                               pin_memory=True,
                                               drop_last=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=args.batch_size,
                                               sampler=valid_sampler,
                                               num_workers=args.workers,
                                               pin_memory=True)

    # check poisoned samples
    # print('train poison samples:', sum(need_poison_down_check(train_dataset.x[1],args.backdoor_scale)))
    # print('test poison samples:', sum(need_poison_down_check(valid_dataset.x[1],args.backdoor_scale)))
    print(train_dataset.poison_list[:10])
    poison_list = train_dataset.poison_list

    ##### set model
    local_models = []
    if args.model == 'mlp2':
        for i in range(args.k-1):
            backbone = MLP2(input_dims[i], NUM_CLASSES)
            local_models.append(backbone)
    elif args.model == 'resnet18':
        for i in range(args.k-1):
            backbone = resnet18(NUM_CLASSES)
            local_models.append(backbone)
    elif args.model == 'resnet50':
        for i in range(args.k-1):
            backbone = resnet50(NUM_CLASSES)
            local_models.append(backbone)
    elif args.model == 'simplecnn':
        for i in range(args.k-1):
            backbone = SimpleCNN(NUM_CLASSES)
            local_models.append(backbone)
    elif args.model == 'lenet':
        for i in range(args.k-1):
            backbone = LeNet5(NUM_CLASSES)
            local_models.append(backbone)

    criterion = nn.CrossEntropyLoss()

    model_list = []
    for i in range(args.k):
        if i == 0:
            if args.use_project_head == 1:
                active_model = ClassificationModelHostTrainableHead(NUM_CLASSES*2, NUM_CLASSES).to(device)
                logging.info('Trainable active party')
            else:
                if args.explicit_softmax == 1:
                    active_model = ClassificationModelHostHeadWithSoftmax().to(device)
                    criterion = nn.NLLLoss()
                    logging.info('Non-trainable active party with softmax layer')
                else:
                    active_model = ClassificationModelHostHead().to(device)
                logging.info('Non-trainable active party')
        else:
            model_list.append(ClassificationModelGuest(local_models[i-1]))
    # quarantine top model
    quarantine_active_models = []
    if args.rvfr:
        for i in range(args.k-1):
            if args.use_project_head == 1:
                quarantine_active_models.append(ClassificationModelHostTrainableHead_quarantine(NUM_CLASSES, NUM_CLASSES).to(device))
            else:
                if args.explicit_softmax == 1:
                    quarantine_active_models.append(ClassificationModelHostHeadWithSoftmax_quarantine().to(device))
                else:
                    quarantine_active_models.append(ClassificationModelHostHead_quarantine().to(device))

    local_models = None
    model_list = [model.to(device) for model in model_list]

    criterion = criterion.to(device)

    # weights optimizer
    optimizer_active_model = None
    if args.use_project_head == 1:
        optimizer_active_model = torch.optim.SGD(active_model.parameters(), args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
        optimizer_list = [
            torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
            for model in model_list]
    else:
        optimizer_list = [
            torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum,
                            weight_decay=args.weight_decay)
            for model in model_list]
    # quarantien top model
    if args.rvfr:
        optimizer_quarantine_active_models = [None]*len(quarantine_active_models)
        if args.use_project_head == 1:
            for i in range(args.k-1):
                optimizer_quarantine_active_models[i] = torch.optim.SGD(quarantine_active_models[i].parameters(), args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

    scheduler_list = []
    if args.learning_rate == 0.025:
        if optimizer_active_model is not None:
            scheduler_list.append(torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_active_model, float(args.epochs)))
        scheduler_list = scheduler_list + [
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
            for optimizer in optimizer_list]
    else:
        if optimizer_active_model is not None:
            scheduler_list.append(torch.optim.lr_scheduler.StepLR(optimizer_active_model, args.decay_period, gamma=args.gamma))
        scheduler_list = [torch.optim.lr_scheduler.StepLR(optimizer, args.decay_period, gamma=args.gamma) for optimizer
                          in optimizer_list]
    # for quarantine training
    quarantine_scheduler_list = []
    if args.rvfr:
        if args.learning_rate == 0.025:
            if optimizer_active_model is not None:
                for i in range(args.k-1):
                    quarantine_scheduler_list.append(torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_quarantine_active_models[i], float(args.quarantine_epochs)))
            quarantine_scheduler_list = quarantine_scheduler_list + [
                torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.quarantine_epochs))
                for optimizer in optimizer_list]
        else:
            if optimizer_active_model is not None:
                for i in range(args.k-1):
                    quarantine_scheduler_list.append(torch.optim.lr_scheduler.StepLR(optimizer_quarantine_active_models[i], args.decay_period, gamma=args.gamma))
            quarantine_scheduler_list = [torch.optim.lr_scheduler.StepLR(optimizer, args.decay_period, gamma=args.gamma) for optimizer
                            in optimizer_list]

    best_acc_top1 = 0.

    # get train backdoor data
    train_backdoor_images, train_backdoor_true_labels = train_dataset.get_poison_data()

    # get train target data
    # train_target_images, train_target_labels = train_dataset.get_target_data()
    # print('train target_ data', train_target_images[0].shape, train_target_labels)
    # print('train poison samples:', sum(need_poison_down_check(train_backdoor_images[1],args.backdoor_scale))) #TODO::QUESTION::should be zero???

    # get test backdoor data
    test_backdoor_images, test_backdoor_true_labels = valid_dataset.get_poison_data()

    # set test backdoor label
    test_backdoor_labels = copy.deepcopy(test_backdoor_true_labels)
    # test_backdoor_labels[:] = valid_dataset.target_label

    target_label = train_dataset.target_label
    print('the label of the sample need copy = ', train_dataset.target_label, valid_dataset.target_label)


    amplify_rate = torch.tensor(args.amplify_rate).float().to(device)

    args.mid_model = None
    args.mid_enlarge_model = None
    args.mid_optimizer = None
    args.mid_enlarge_optimizer = None
    if args.mid:
        args.mid_model = MID_layer(NUM_CLASSES, NUM_CLASSES).to(args.device)
        args.mid_enlarge_model = MID_enlarge_layer(NUM_CLASSES,NUM_CLASSES*2).to(args.device)
        args.mid_optimizer = torch.optim.SGD(args.mid_model.parameters(), args.learning_rate, 
                                    momentum=args.momentum, weight_decay=args.weight_decay)
        args.mid_enlarge_optimizer = torch.optim.SGD(args.mid_enlarge_model.parameters(), args.learning_rate, 
                                    momentum=args.momentum, weight_decay=args.weight_decay)

    args.RAE = None
    args.rae_optimizer = None
    args.logits_optimizer = None
    if args.rvfr:
        args.RAE = RAE(NUM_CLASSES*(args.k-1)).to(args.device)
        args.rae_optimizer = torch.optim.SGD(args.RAE.parameters(), args.learning_rate, 
                                    momentum=args.momentum, weight_decay=args.weight_decay)

    active_up_gradients_res = None
    active_down_gradients_res = None
    
    if args.rvfr:
        # quarantine training loop
        for epoch in range(args.quarantine_epochs):
            for model in quarantine_active_models:
                model.train()
            for model in model_list:
                model.train()

            for step, (trn_X, trn_y) in enumerate(train_loader):
                # # select one backdoor data
                # id = random.randint(0, train_backdoor_images[0].shape[0]-1)
                # backdoor_image_up = train_backdoor_images[0][id]
                # backdoor_image_down = train_backdoor_images[1][id]
                # backdoor_label = train_backdoor_true_labels[id]
                # # select one target data
                # id = random.randint(0, train_target_images[0].shape[0]-1)
                # target_image_up = train_target_images[0][id]
                # target_image_down = train_target_images[1][id]

                # merge normal train data with selected backdoor and target data
                trn_X_up = np.concatenate([trn_X[0].numpy()])
                trn_X_down = np.concatenate([trn_X[1].numpy()])
                trn_y = np.concatenate([trn_y.numpy()])

                trn_X_up = torch.from_numpy(trn_X_up).float().to(device)
                trn_X_down = torch.from_numpy(trn_X_down).float().to(device)
                target = torch.from_numpy(trn_y).view(-1).long().to(device)

                N = target.size(0)

                # passive party 0 generate output
                z_up = model_list[0](trn_X_up)
                z_up_clone = z_up.detach().clone()
                z_up_clone = torch.autograd.Variable(z_up_clone, requires_grad=True).to(args.device)

                # passive party 1 generate output
                z_down = model_list[1](trn_X_down)
                z_down_clone = z_down.detach().clone()
                z_down_clone = torch.autograd.Variable(z_down_clone, requires_grad=True).to(args.device)

                # active party backward
                logits = [quarantine_active_models[0](z_up_clone), quarantine_active_models[1](z_down_clone)]
                loss = [criterion(_logits, target) for _logits in logits]

                z_gradients_up = torch.autograd.grad(loss[0], z_up_clone, retain_graph=True)
                z_gradients_down = torch.autograd.grad(loss[1], z_down_clone, retain_graph=True)
                z_gradients_up_clone = z_gradients_up[0].detach().clone()
                z_gradients_down_clone = z_gradients_down[0].detach().clone()

                # update quarantine active model
                for i in range(args.k-1):
                    if optimizer_quarantine_active_models[i] is not None:
                        optimizer_quarantine_active_models[i].zero_grad()
                        loss[i].backward(retain_graph=True)
                        optimizer_quarantine_active_models[i].step()
                
                # update passive model 0
                optimizer_list[0].zero_grad()
                weights_gradients_up = torch.autograd.grad(z_up, model_list[0].parameters(),grad_outputs=z_gradients_up_clone)
                for w, g in zip(model_list[0].parameters(), weights_gradients_up):
                    if w.requires_grad:
                        w.grad = g.detach()
                optimizer_list[0].step()
                # update passive model 1
                optimizer_list[1].zero_grad()
                weights_gradients_down = torch.autograd.grad(z_down, model_list[1].parameters(),grad_outputs=z_gradients_down_clone)
                for w, g in zip(model_list[1].parameters(), weights_gradients_down):
                    if w.requires_grad:
                        w.grad = g.detach()
                optimizer_list[1].step()
            # update scheduler
            for scheduler in quarantine_scheduler_list:
                scheduler.step()
        print("[logging info] quarantine training finished")
        # pre-train RAE loop
        for e in range(args.rae_pretrain_epochs):
            args.RAE.train()
            for step, (trn_X, trn_y) in enumerate(train_loader):
                # # select one backdoor data
                # id = random.randint(0, train_backdoor_images[0].shape[0]-1)
                # backdoor_image_up = train_backdoor_images[0][id]
                # backdoor_image_down = train_backdoor_images[1][id]
                # backdoor_label = train_backdoor_true_labels[id]
                # # select one target data
                # id = random.randint(0, train_target_images[0].shape[0]-1)
                # target_image_up = train_target_images[0][id]
                # target_image_down = train_target_images[1][id]

                # merge normal train data with selected backdoor and target data
                trn_X_up = np.concatenate([trn_X[0].numpy()])
                trn_X_down = np.concatenate([trn_X[1].numpy()])
                trn_y = np.concatenate([trn_y.numpy()])

                trn_X_up = torch.from_numpy(trn_X_up).float().to(device)
                trn_X_down = torch.from_numpy(trn_X_down).float().to(device)
                target = torch.from_numpy(trn_y).view(-1).long().to(device)

                N = target.size(0)

                # passive party 0 generate output
                z_up = model_list[0](trn_X_up)
                z_up_clone = z_up.detach().clone()
                # passive party 1 generate output
                z_down = model_list[1](trn_X_down)
                z_down_clone = z_down.detach().clone()

                z_up_clone = F.softmax(z_up_clone,dim=-1)
                z_down_clone = F.softmax(z_down_clone,dim=-1)
                z_cat = torch.cat([z_up_clone,z_down_clone],dim=1)
                z_cat = torch.autograd.Variable(z_cat, requires_grad=True).to(args.device)

                decode, encode = args.RAE(z_cat)
                rae_loss = nn.MSELoss().to(args.device)(decode,z_cat)
                rae_gradients = torch.autograd.grad(rae_loss, decode, retain_graph=True)
                rae_gradients_clone = rae_gradients[0].detach().clone()
                
                args.rae_optimizer.zero_grad()
                weight_rae_gradients = torch.autograd.grad(decode, args.RAE.parameters(), grad_outputs=rae_gradients_clone)
                for w, g in zip(args.RAE.parameters(), weight_rae_gradients):
                    if w.requires_grad:
                        w.grad = g.detach()
                args.rae_optimizer.step()
        print("[logging info] RAE pretrain finished")
        # fine-tune RAE and calculate Low and E to make H_input=Low+E
        # step-1: prepare H_input and origianl Low
        H_input = []
        Low = []
        for step, (trn_X, trn_y) in enumerate(train_loader):
            # # select one backdoor data
            # id = random.randint(0, train_backdoor_images[0].shape[0]-1)
            # backdoor_image_up = train_backdoor_images[0][id]
            # backdoor_image_down = train_backdoor_images[1][id]
            # backdoor_label = train_backdoor_true_labels[id]
            # # select one target data
            # id = random.randint(0, train_target_images[0].shape[0]-1)
            # target_image_up = train_target_images[0][id]
            # target_image_down = train_target_images[1][id]

            # merge normal train data with selected backdoor and target data
            trn_X_up = np.concatenate([trn_X[0].numpy()])
            trn_X_down = np.concatenate([trn_X[1].numpy()])
            trn_y = np.concatenate([trn_y.numpy()])

            trn_X_up = torch.from_numpy(trn_X_up).float().to(device)
            trn_X_down = torch.from_numpy(trn_X_down).float().to(device)
            target = torch.from_numpy(trn_y).view(-1).long().to(device)

            N = target.size(0)

            # passive party 0 generate output
            z_up = model_list[0](trn_X_up)
            z_up_clone = z_up.detach().clone()
            # passive party 1 generate output
            z_down = model_list[1](trn_X_down)
            z_down_clone = z_down.detach().clone()
            # concatenate
            z_up_clone = F.softmax(z_up_clone,dim=-1)
            z_down_clone = F.softmax(z_down_clone,dim=-1)
            z_cat = torch.cat([z_up_clone,z_down_clone],dim=1)
            z_cat = torch.autograd.Variable(z_cat, requires_grad=True).to(args.device)
            H_input.append(z_cat.detach().clone())
            Low.append(z_cat.detach().clone())
        count = len(H_input)
        # H_input = torch.stack(H_input).float()
        # Low = torch.stack(Low).float()
        H_input = torch.nn.utils.rnn.pad_sequence(H_input,batch_first=True).float()
        Low = torch.nn.utils.rnn.pad_sequence(Low,batch_first=True).float()
        H_input = torch.autograd.Variable(H_input, requires_grad=True).to(args.device)
        Low = torch.autograd.Variable(Low, requires_grad=False).to(args.device)
        # args.logits_optimizer = torch.optim.SGD([Low], args.learning_rate, 
        #                             momentum=args.momentum, weight_decay=args.weight_decay)
        # step-2: fine-tune and train RAE and Low alternatively
        for e in range(args.rae_tune_epochs):
            args.RAE.train()
            # train RAE
            for step in range(count):
                decode, encode = args.RAE(Low[step])
                # print("decod",decode,"Low[step]",Low[step])
                rae_loss = args.rvfr_lambda * torch.tensor(utils.l21_colnorm(encode.cpu().detach().numpy())).to(args.device) + args.rvfr_beta * nn.MSELoss().to(args.device)(decode,Low[step])
                # rae_gradients = torch.autograd.grad(rae_loss, Low[step], retain_graph=True)
                # rae_gradients_clone = rae_gradients[0].detach().clone()
                
                args.rae_optimizer.zero_grad()
                # rae_gradients = torch.autograd.grad(decode, args.RAE.parameters(),grad_outputs=rae_gradients_clone)
                rae_gradients = torch.autograd.grad(rae_loss, args.RAE.parameters(),retain_graph=True)
                for w, g in zip(args.RAE.parameters(), rae_gradients):
                    if w.requires_grad:
                        w.grad = g.detach()
                args.rae_optimizer.step()
            # train robust logits L
            for step in range(count):
                low = torch.autograd.Variable(Low[step], requires_grad=True)
                args.logits_optimizer = torch.optim.SGD([low],1)
                decode, encode = args.RAE(low)
                l_loss = args.rvfr_lambda * torch.tensor(utils.l21_colnorm(encode.cpu().detach().numpy())).to(args.device) + args.rvfr_beta * nn.MSELoss().to(args.device)(decode,low) + args.rvfr_lambda * torch.tensor(utils.l21_rownorm((H_input[step] - low).cpu().detach().numpy())).to(args.device)
                # l_loss *= 1000
                args.logits_optimizer.zero_grad()
                # print("low.grad1",low.grad)
                torch.autograd.backward(l_loss, inputs=low, retain_graph=True)
                # print("low.grad2",low.grad)
                # print("l_gradients:",l_gradients)
                # l_loss.backward()
                # for w, g in zip([low], l_gradients):
                #     if w.requires_grad:
                #         print("w.grad", w.grad)
                #         print("g",g)
                #         # print("w.grad.size()", w.grad.size(),"g.size()",g.size())
                #         w.grad = g.detach()
                # original_L_step = low.detach().clone()
                args.logits_optimizer.step()
                # updated_L_step = low.detach().clone()
                # print("L[step] is updated? ", updated_L_step-original_L_step)
                # Low[step] = low.detach().clone()
                # Low[step] = low - l_gradients[0]
                # assert 0==1
            # see the whole loss of H_input and Low
            H1 = torch.split(H_input,1,dim=0)
            H2 = torch.cat(H1,dim=1)
            H3 = H2.squeeze(0)
            L1 = torch.split(Low,1,dim=0)
            L2 = torch.cat(L1,dim=1)
            L3 = L2.squeeze(0)
            assert L3.size() == H3.size()
            decode, encode = args.RAE(Low)
            l_loss = args.rvfr_lambda * torch.tensor(utils.l21_colnorm(encode.cpu().detach().numpy())).to(args.device) + args.rvfr_beta * nn.MSELoss().to(args.device)(decode,Low) + args.rvfr_lambda * torch.tensor(utils.l21_rownorm((H_input - Low).cpu().detach().numpy())).to(args.device)
            print("whole loss for rae fine-tune", l_loss.item())
    print("[logging info]: rae fine-tune done")

    # normal training loop
    for epoch in range(args.epochs):

        output_replace_count = 0
        gradient_replace_count = 0
        ########### TRAIN ###########
        top1 = utils.AverageMeter()
        losses = utils.AverageMeter()

        cur_step = epoch * len(train_loader)
        cur_lr = optimizer_list[0].param_groups[0]['lr']
        if args.writer == 1:
            writer.add_scalar('train/lr', cur_lr, cur_step)

        # if args.certify != 0 and epoch >= args.certify_start_epoch:
        #     print("ClipAndPerturb the model in the training process")
        #     model_list[0] = utils.ClipAndPerturb(model_list[0],device,epoch*0.1+2,args.sigma)
        #     model_list[1] = utils.ClipAndPerturb(model_list[1],device,epoch*0.1+2,args.sigma)

        for model in model_list:
            active_model.train()
            model.train()
            model.eval()

        for step, (trn_X, trn_y) in enumerate(train_loader):

            # select one backdoor data
            id = random.randint(0, train_backdoor_images[0].shape[0]-1)
            backdoor_image_up = train_backdoor_images[0][id]
            backdoor_image_down = train_backdoor_images[1][id]
            backdoor_label = train_backdoor_true_labels[id]
            # # select one target data
            # id = random.randint(0, train_target_images[0].shape[0]-1)
            # target_image_up = train_target_images[0][id]
            # target_image_down = train_target_images[1][id]

            # merge normal train data with selected backdoor and target data
            trn_X_up = np.concatenate([trn_X[0].numpy()])
            trn_X_down = np.concatenate([trn_X[1].numpy()])
            trn_y = np.concatenate([trn_y.numpy()])

            trn_X_up = torch.from_numpy(trn_X_up).float().to(device)
            trn_X_down = torch.from_numpy(trn_X_down).float().to(device)
            target = torch.from_numpy(trn_y).view(-1).long().to(device)

            N = target.size(0)
            if args.rvfr:
                if N < args.batch_size:
                    trn_X_up = F.pad(trn_X_up,(0,args.batch_size-N),mode='constant')
                    trn_X_down = F.pad(trn_X_down,(0,args.batch_size-N),mode='constant')
                    target = F.pad(target,(0,args.batch_size-N),mode='constant')

            # passive party 0 generate output
            z_up = model_list[0](trn_X_up)
            z_up_clone = z_up.detach().clone()
            z_up_clone = torch.autograd.Variable(z_up_clone, requires_grad=True).to(args.device)

            # passive party 1 generate output
            z_down = model_list[1](trn_X_down)
            if args.certify != 0 and epoch >= args.certify_start_epoch:
                z_down = utils.ClipAndPerturb(z_down,device,epoch*0.1+2,args.sigma)
            z_down_clone = z_down.detach().clone()

            z_down_clone = torch.autograd.Variable(z_down_clone, requires_grad=True).to(args.device)

            # active party backward
            if args.rvfr:
                z_up = F.softmax(z_up_clone,dim=-1)
                z_down = F.softmax(z_down_clone,dim=-1)
                recover_z, new_L = utils.RAE_purify(args, torch.cat([z_up,z_down],dim=1), args.RAE, Low[step], is_train=True)
                Low[step] = new_L.detach().clone()
                # z_up_clone = recover_z[0].detach().clone()
                # z_up_clone = torch.autograd.Variable(z_up_clone, requires_grad=True).to(args.device)
                z_down_clone = recover_z[1].detach().clone()
                z_down_clone = torch.autograd.Variable(z_down_clone, requires_grad=True).to(args.device)
                logits = active_model(z_up_clone, z_down_clone)
            else:
                logits = active_model(z_up_clone, z_down_clone)
            # logits = active_model(z_up_clone, z_down_clone)
            loss = criterion(logits, target)
            if args.mid == 1 and args.mid_lambda > 0.0:
                epsilon = torch.empty(z_down_clone.size())
                torch.nn.init.normal_(epsilon, mean=0, std=1) # epsilon is initialized
                epsilon = epsilon.to(args.device)
                pred_a_double = args.mid_enlarge_model(z_down_clone)
                mu, std = pred_a_double[:,:NUM_CLASSES], pred_a_double[:,NUM_CLASSES:]
                std = F.softplus(std-5, beta=1) # ? F.softplus(std-5)
                pred_Z = mu+std*epsilon
                assert(pred_Z.size()==z_down_clone.size())
                pred_Z = pred_Z.to(args.device)
                pred_Z = args.mid_model(pred_Z)
                logits = active_model(z_up_clone, pred_Z)
                loss = criterion(logits, target) + args.mid_lambda * torch.mean(torch.sum((-0.5)*(1+2*torch.log(std)-mu**2 - std**2),1))

                # # print("before mid")
                # t_samples = args.mid_model(z_down_clone)
                # positive = torch.zeros_like(t_samples)
                # prediction_1 = t_samples.unsqueeze(1)  # [nsample,1,dim]
                # t_samples_1 = t_samples.unsqueeze(0)  # [1,nsample,dim]
                # negative = - ((t_samples_1 - prediction_1) ** 2).mean(dim=1) / 2.   # [nsample, dim]
                # logits = active_model(z_up_clone, t_samples)
                # loss = criterion(logits, target )+ args.mid_lambda * (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()

                # # new version of mid
                # ########################### v3 #############################################
                # epsilon = torch.empty((z_down_clone.size()[0],z_down_clone.size()[1]))
                # torch.nn.init.normal_(epsilon, mean=0, std=1) # epsilon is initialized
                # epsilon = epsilon.to(args.device)
                # mu = torch.mean(z_down_clone)
                # std = torch.std(z_down_clone, unbiased=False)
                # std = F.softplus(std-5, beta=1)
                # # mu, std = norm.fit(z_down_clone.cpu().detach().numpy())
                # _samples = mu + std * epsilon
                # _samples = _samples.to(args.device)
                # t_samples = args.mid_model(_samples)
                # logits = active_model(z_up_clone, t_samples)
                # loss = criterion(logits, target) + args.mid_lambda * (-0.5)*(1+2*torch.log(std)-mu**2 - std**2)


            # loss_benign = criterion(logits[:-1],target[:-1])
            # loss_melicious = criterion(logits[-1:],target[-1:])
            # file.write("Epoch: {} Step: {} loss_benign: {} loss_melicious: {}\n".format(epoch, step, loss_benign.item(), loss_melicious.item()))
            # print(loss_benign.item(),loss_melicious.item())

            if args.mid and args.mid_lambda > 0.0:
                pred_a_double_gradients = torch.autograd.grad(loss, pred_a_double, retain_graph=True)
                pred_a_double_gradients_clone = pred_a_double_gradients[0].detach().clone()
                pred_Z_gradients = torch.autograd.grad(loss, pred_Z, retain_graph=True)
                pred_Z_gradients_clone = pred_Z_gradients[0].detach().clone()
                # t_samples_gradients = torch.autograd.grad(loss, t_samples, retain_graph=True)
                # t_samples_gradients_clone = t_samples_gradients[0].detach().clone()

            z_gradients_up = torch.autograd.grad(loss, z_up_clone, retain_graph=True)
            z_gradients_down = torch.autograd.grad(loss, z_down_clone, retain_graph=True)

            z_gradients_up_clone = z_gradients_up[0].detach().clone()
            z_gradients_down_clone = z_gradients_down[0].detach().clone()

            # update active model
            if optimizer_active_model is not None:
                optimizer_active_model.zero_grad()
                loss.backward(retain_graph=True)
                optimizer_active_model.step()

            ########### defense start here ##########
            location = 0.0
            threshold = 0.2
            if args.dp_type == 'laplace':
                with torch.no_grad():
                    scale = args.dp_strength
                    # clip 2-norm per sample
                    norm_factor_up = torch.div(torch.max(torch.norm(z_gradients_up_clone, dim=1)),
                                               threshold + 1e-6).clamp(min=1.0)
                    norm_factor_down = torch.div(torch.max(torch.norm(z_gradients_down_clone, dim=1)),
                                               threshold + 1e-6).clamp(min=1.0)

                    # add laplace noise
                    dist_up = torch.distributions.laplace.Laplace(location, scale)
                    dist_down = torch.distributions.laplace.Laplace(location, scale)

                    if args.defense_up == 1:
                        z_gradients_up_clone = torch.div(z_gradients_up_clone, norm_factor_up) + \
                                               dist_up.sample(z_gradients_up_clone.shape).to(device)
                    z_gradients_down_clone = torch.div(z_gradients_down_clone, norm_factor_down) + \
                                             dist_down.sample(z_gradients_down_clone.shape).to(device)
            if args.dp_type == 'gaussian':
                with torch.no_grad():
                    scale = args.dp_strength
                    norm_factor_up = torch.div(torch.max(torch.norm(z_gradients_up_clone, dim=1)),
                                               threshold + 1e-6).clamp(min=1.0)
                    norm_factor_down = torch.div(torch.max(torch.norm(z_gradients_down_clone, dim=1)),
                                               threshold + 1e-6).clamp(min=1.0)
                    if args.defense_up == 1:
                        z_gradients_up_clone = torch.div(z_gradients_up_clone, norm_factor_up) + \
                                               torch.normal(location, scale, z_gradients_up_clone.shape).to(device)
                    z_gradients_down_clone = torch.div(z_gradients_down_clone, norm_factor_down) + \
                                             torch.normal(location, scale, z_gradients_down_clone.shape).to(device)
            if args.gradient_sparsification != 0:
                with torch.no_grad():
                    percent = args.gradient_sparsification / 100.0
                    if active_up_gradients_res is not None and \
                            z_gradients_up_clone.shape[0] == active_up_gradients_res.shape[0] and args.defense_up == 1:
                        z_gradients_up_clone = z_gradients_up_clone + active_up_gradients_res
                    if active_down_gradients_res is not None and \
                            z_gradients_down_clone.shape[0] == active_down_gradients_res.shape[0]:
                        z_gradients_down_clone = z_gradients_down_clone + active_down_gradients_res
                    up_thr = torch.quantile(torch.abs(z_gradients_up_clone), percent)
                    down_thr = torch.quantile(torch.abs(z_gradients_down_clone), percent)

                    active_up_gradients_res = torch.where(torch.abs(z_gradients_up_clone).double() < up_thr.item(),
                                                          z_gradients_up_clone.double(), float(0.)).to(device)
                    active_down_gradients_res = torch.where(
                        torch.abs(z_gradients_down_clone).double() < down_thr.item(), z_gradients_down_clone.double(),
                        float(0.)).to(device)
                    if args.defense_up == 1:
                        z_gradients_up_clone = z_gradients_up_clone - active_up_gradients_res
                    z_gradients_down_clone = z_gradients_down_clone - active_down_gradients_res
            
            if args.apply_discrete_gradients:
                z_gradients_up_clone = utils.multistep_gradient(z_gradients_up_clone, bins_num=args.discrete_gradients_bins, bound_abs=args.discrete_gradients_bound)
                z_gradients_down_clone = utils.multistep_gradient(z_gradients_down_clone, bins_num=args.discrete_gradients_bins, bound_abs=args.discrete_gradients_bound)
            ########### defense end here ##########

            # # update mid_model and mid_enlarge_model if it exists
            # if args.mid and args.mid_lambda > 0.0 and args.mid_optimizer != None:
            #     args.mid_enlarge_optimizer.zero_grad()
            #     weights_gradients_mid = torch.autograd.grad(pred_a_double, args.mid_enlarge_model.parameters(),
            #                                         grad_outputs=pred_a_double_gradients_clone)

            #     for w, g in zip(args.mid_enlarge_model.parameters(), weights_gradients_mid):
            #         if w.requires_grad:
            #             w.grad = g.detach()
            #     args.mid_enlarge_optimizer.step()
            #     args.mid_optimizer.zero_grad()
            #     weights_gradients_mid = torch.autograd.grad(pred_Z, args.mid_model.parameters(),
            #                                         grad_outputs=pred_Z_gradients_clone)

            #     for w, g in zip(args.mid_model.parameters(), weights_gradients_mid):
            #         if w.requires_grad:
            #             w.grad = g.detach()
            #     args.mid_optimizer.step()
            #     # args.mid_optimizer.zero_grad()
            #     # weights_gradients_mid = torch.autograd.grad(t_samples, args.mid_model.parameters(),
            #     #                                     grad_outputs=t_samples_gradients_clone)

            #     # for w, g in zip(args.mid_model.parameters(), weights_gradients_mid):
            #     #     if w.requires_grad:
            #     #         w.grad = g.detach()
            #     # args.mid_optimizer.step()

            # # update passive model 0
            # optimizer_list[0].zero_grad()
            # weights_gradients_up = torch.autograd.grad(z_up, model_list[0].parameters(),
            #                                         grad_outputs=z_gradients_up_clone)

            # for w, g in zip(model_list[0].parameters(), weights_gradients_up):
            #     if w.requires_grad:
            #         w.grad = g.detach()
            # optimizer_list[0].step()

            # # update passive model 1
            # optimizer_list[1].zero_grad()
            # if args.backdoor == 1:
            #     weights_gradients_down = torch.autograd.grad(z_down[:-1], model_list[1].parameters(),
            #                                                  grad_outputs=z_gradients_down_clone[:-1])
            # else:
            #     weights_gradients_down = torch.autograd.grad(z_down, model_list[1].parameters(),
            #                                                  grad_outputs=z_gradients_down_clone)

            # for w, g in zip(model_list[1].parameters(), weights_gradients_down):
            #     if w.requires_grad:
            #         w.grad = g.detach()
            # optimizer_list[1].step()

            # train metrics
            prec1 = utils.accuracy(logits, target, topk=(1,))
            losses.update(loss.item(), N)
            top1.update(prec1[0].item(), N)
            if args.writer == 1:
                writer.add_scalar('train/loss', losses.avg, cur_step)
                writer.add_scalar('train/top1', top1.avg, cur_step)
            cur_step += 1

        cur_step = (epoch + 1) * len(train_loader)

        ########### VALIDATION ###########
        top1_valid = utils.AverageMeter()
        losses_valid = utils.AverageMeter()

        # validate_model_list = []
        for model in model_list:
            active_model.eval()
            model.eval()

        if args.certify != 0 and epoch >= args.certify_start_epoch:
            # print("validation with voting")
            # for m in range(args.M):
            #     validate_model_list.append([utils.ClipAndPerturb(model_list[0],device,args.epochs*0.1+2,args.sigma),\
            #                                 utils.ClipAndPerturb(model_list[1],device,args.epochs*0.1+2,args.sigma)])
            # for m in range(args.M):
            #     active_model.eval()
            #     validate_model_list[m][0].eval()
            #     validate_model_list[m][1].eval()

            with torch.no_grad():
                # test accuracy
                for step, (val_X, val_y) in enumerate(valid_loader):
                    val_X = [x.float().to(args.device) for x in val_X]
                    target = val_y.view(-1).long().to(args.device)
                    N = target.size(0)
                
                    loss_list = []
                    logits_list = []
                    vote_list = []
                    z_up = model_list[0](val_X[0])
                    for m in range(args.M):
                        z_down = model_list[1](val_X[1])
                        z_down = utils.ClipAndPerturb(z_down,device,args.epochs*0.1+2,args.sigma)

                        logits = active_model(z_up, z_down)
                        loss = criterion(logits, target)
                        logits_list.append(logits) #logits.shape=[64,10]
                        loss_list.append(loss)

                        vote = utils.vote(logits,topk=(1,))
                        vote_list.append(vote.cpu().numpy())
                    #vote_list.shape=(m,N) in cpu as numpy.array
                    # print("N=",N,"vote_list.shape=",(np.asarray(vote_list)).shape) #N=64, vote_list.shape=(m,N)
                    vote_list = np.transpose(np.asarray(vote_list))
                    vote_result = np.apply_along_axis(lambda x: np.bincount(x, minlength=logits.shape[1]), axis=1, arr=vote_list)
                    # print(vote_result.shape) #(N,logits.shape[1]=#class)
                    vote_result = torch.from_numpy(vote_result)
                    vote_result.to(device)
                    
                    prec1 = utils.accuracy2(vote_result, target, args.M, device, topk=(1,2,))
                    
                    losses_valid.update(loss.item(), N)
                    top1_valid.update(prec1[0].item(), N)

                # backdoor related metrics
                backdoor_X_up = torch.from_numpy(test_backdoor_images[0]).float().to(args.device)
                backdoor_X_down = torch.from_numpy(test_backdoor_images[1]).float().to(args.device)
                backdoor_labels = torch.from_numpy(test_backdoor_labels).long().to(args.device)

                N = backdoor_labels.shape[0]
                
                loss_list = []
                logits_list = []
                vote_list = []
                z_up = model_list[0](backdoor_X_up)
                for m in range(args.M):
                    z_down = model_list[1](backdoor_X_down)
                    z_down = utils.ClipAndPerturb(z_down,device,args.epochs*0.1+2,args.sigma)

                    ########## backdoor metric
                    logits_backdoor = active_model(z_up, z_down)
                    loss_backdoor = criterion(logits_backdoor, backdoor_labels)
                    logits_list.append(logits_backdoor)
                    loss_list.append(loss_backdoor)

                    vote = utils.vote(logits_backdoor,topk=(1,))
                    vote_list.append(vote.cpu().numpy())
                #vote_list.shape=(m,N) in cpu as numpy.array
                # print("N=",N,"vote_list.shape=",(np.asarray(vote_list)).shape) #N=64, vote_list.shape=(m,N)
                vote_list = np.transpose(np.asarray(vote_list))
                # print("vote_list.shape =",vote_list.shape, " logits.shape =", logits_backdoor.shape, "N =",N)
                vote_result = np.apply_along_axis(lambda x: np.bincount(x, minlength=logits.shape[1]), axis=1, arr=vote_list)
                # print(vote_result.shape) #(N,logits.shape[1]=#class)
                vote_result = torch.from_numpy(vote_result)
                vote_result.to(device)
                
                prec1 = utils.accuracy2(vote_result, backdoor_labels[0:vote_list.shape[0]], args.M, device, topk=(1,2,))

                losses_backdoor = loss_backdoor.item()
                top1_backdoor = prec1[0]
        else:
            print("validation withiout voting")

            with torch.no_grad():
                # test accuracy
                for step, (val_X, val_y) in enumerate(valid_loader):
                    val_X = [x.float().to(args.device) for x in val_X]
                    target = val_y.view(-1).long().to(args.device)
                    N = target.size(0)

                    z_up = model_list[0](val_X[0])
                    z_down = model_list[1](val_X[1])

                    if args.rvfr:
                        z_up = F.softmax(z_up,dim=-1)
                        z_down = F.softmax(z_down,dim=-1)
                        recover_z, _ = utils.RAE_purify(args, torch.cat([z_up,z_down],dim=1), args.RAE, None, is_train=False)
                        # z_up = recover_z[0].detach().clone().to(args.device)
                        z_down = recover_z[1].detach().clone().to(args.device)
                        logits = active_model(z_up, z_down)
                    else:
                        logits = active_model(z_up, z_down)
                    # logits = active_model(z_up, z_down)
                    loss = criterion(logits, target)
                    if args.mid == 1 and args.mid_lambda > 0.0:
                        epsilon = torch.empty(z_down.size())
                        torch.nn.init.normal_(epsilon, mean=0, std=1) # epsilon is initialized
                        epsilon = epsilon.to(args.device)
                        pred_a_double = args.mid_enlarge_model(z_down)
                        mu, std = pred_a_double[:,:NUM_CLASSES], pred_a_double[:,NUM_CLASSES:]
                        std = F.softplus(std-5, beta=1) # ? F.softplus(std-5)
                        pred_Z = mu+std*epsilon
                        assert(pred_Z.size()==z_down.size())
                        pred_Z = pred_Z.to(args.device)
                        pred_Z = args.mid_model(pred_Z)
                        logits = active_model(z_up, pred_Z)
                        loss = criterion(logits, target) + args.mid_lambda * torch.mean(torch.sum((-0.5)*(1+2*torch.log(std)-mu**2 - std**2),1))

                        # # print("before mid")
                        # t_samples = args.mid_model(z_down)
                        # positive = torch.zeros_like(t_samples)
                        # prediction_1 = t_samples.unsqueeze(1)  # [nsample,1,dim]
                        # t_samples_1 = t_samples.unsqueeze(0)  # [1,nsample,dim]
                        # negative = - ((t_samples_1 - prediction_1) ** 2).mean(dim=1) / 2.   # [nsample, dim]
                        # logits = active_model(z_up, t_samples)
                        # loss = criterion(logits, target) + args.mid_lambda * (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()

                        # # new version of mid
                        # ########################### v3 #############################################
                        # epsilon = torch.empty((z_down.size()[0],z_down.size()[1]))
                        # torch.nn.init.normal_(epsilon, mean=0, std=1) # epsilon is initialized
                        # epsilon = epsilon.to(args.device)
                        # mu = torch.mean(z_down)
                        # std = torch.std(z_down, unbiased=False)
                        # std = F.softplus(std-5, beta=1)
                        # # mu, std = norm.fit(z_down.cpu().detach().numpy())
                        # _samples = mu + std * epsilon
                        # _samples = _samples.to(args.device)
                        # t_samples = args.mid_model(_samples)
                        # logits = active_model(z_up, t_samples)
                        # loss = criterion(logits, target) + args.mid_lambda * (-0.5)*(1+2*torch.log(std)-mu**2 - std**2)

                        
                    prec1 = utils.accuracy(logits, target, topk=(1,))

                    losses_valid.update(loss.item(), N)
                    top1_valid.update(prec1[0].item(), N)

                # backdoor related metrics
                backdoor_X_up = torch.from_numpy(test_backdoor_images[0]).float().to(args.device)
                backdoor_X_down = torch.from_numpy(test_backdoor_images[1]).float().to(args.device)
                backdoor_labels = torch.from_numpy(test_backdoor_labels).long().to(args.device)

                N = backdoor_labels.shape[0]

                z_up = model_list[0](backdoor_X_up)
                z_down = model_list[1](backdoor_X_down)

                ########## backdoor metric
                if args.rvfr:
                    z_up = F.softmax(z_up,dim=-1)
                    z_down = F.softmax(z_down,dim=-1)
                    recover_z, _ = utils.RAE_purify(args, torch.cat([z_up,z_down],dim=1), args.RAE, None, is_train=False)
                    # z_up = recover_z[0].detach().clone().to(args.device)
                    z_down = recover_z[1].detach().clone().to(args.device)
                    logits_backdoor = active_model(z_up, z_down)
                else:
                    logits_backdoor = active_model(z_up, z_down)
                # logits_backdoor = active_model(z_up, z_down)
                loss_backdoor = criterion(logits_backdoor, backdoor_labels)

                print(logits_backdoor.size(),backdoor_labels.size())
                prec1 = utils.accuracy(logits_backdoor, backdoor_labels, topk=(1,))

                losses_backdoor = loss_backdoor.item()
                top1_backdoor = prec1[0]

        if args.writer == 1:
            writer.add_scalar('val/loss', losses_valid.avg, cur_step)
            writer.add_scalar('val/top1_valid', top1_valid.avg, cur_step)
            writer.add_scalar('backdoor/loss', losses_backdoor, cur_step)
            writer.add_scalar('backdoor/top1_valid', top1_backdoor, cur_step)


        template = 'Epoch {}, Poisoned {}/{}, Loss: {:.4f}, Accuracy: {:.2f}, Test Loss: {:.4f}, Test Accuracy: {:.2f}, ' \
                   'Backdoor Loss: {:.4f}, Backdoor Accuracy: {:.2f}\n ' \

        logging.info(template.format(epoch + 1,
                      output_replace_count,
                      gradient_replace_count,
                      losses.avg,
                      top1.avg,
                      losses_valid.avg,
                      top1_valid.avg,
                      losses_backdoor,
                      top1_backdoor.item()))

        if losses_valid.avg > 1e8 or np.isnan(losses_valid.avg):
            logging.info('********* INSTABLE TRAINING, BREAK **********')
            break

        valid_acc_top1 = top1_valid.avg
        # save
        if valid_acc_top1 > best_acc_top1:
            best_acc_top1 = valid_acc_top1
        logging.info('best_acc_top1 %f', best_acc_top1)


        # update scheduler
        for scheduler in scheduler_list:
            scheduler.step()
    
    # after all iterations of epochs
    file.close()


def get_poisoned_matrix(passive_matrix, need_poison, poison_grad, amplify_rate):
    poisoned_matrix = passive_matrix
    poisoned_matrix[need_poison] = poison_grad * amplify_rate
    return poisoned_matrix


def copy_grad(passive_matrix, need_copy):
    poison_grad = passive_matrix[need_copy]
    return poison_grad


if __name__ == '__main__':
    main()
