import logging

import torch
import time
import argparse
import pprint
from models.autoencoder import AutoEncoder
from vfl_dlg import *
import vfl_dlg
import vfl_dlg_mid
import marvell_scoring_attack, marvell_scoring_main_auc
from models.vision import *
from utils import *

import wandb

from dataset.NuswideDataset import NUSWIDEDataset
from dataset.nuswide_dataset import SimpleDataset

tp = transforms.ToTensor()
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
     ])

transform_fn = transforms.Compose([
    transforms.ToTensor()
])

models_dict = {"mnist": 'MLP2',
               "cifar10": 'resnet18',
               "cifar100": 'resnet18',
               "nuswide": 'MLP2',
               "classifier": None}
epochs_dict = {"mnist": 100000,
              "cifar10": 2000,
              "cifar100": 2000,
              "nuswide": 20000}

def set_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_class_i(dataset, label_set):
    gt_data = []
    gt_labels = []
    # num_cls = len(label_set)
    for j in range(len(dataset)):
        img, label = dataset[j]
        if label in label_set:
            label_new = label_set.index(label)
            gt_data.append(img if torch.is_tensor(img) else tp(img))
            gt_labels.append(label_new)
            # gt_labels.append(label_to_onehot(torch.Tensor([label_new]).long(),num_classes=num_cls))
    # gt_labels = label_to_onehot(torch.Tensor(gt_labels).long(), num_classes=num_cls)
    gt_data = torch.stack(gt_data)
    return gt_data, gt_labels

def fetch_classes(num_classes):
    return np.arange(num_classes).tolist()

def fetch_data_and_label(dataset, num_classes):
    classes = fetch_classes(num_classes)
    return get_class_i(dataset, classes)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_type', default='attack', type=str, help='exp_type should be "attack" or "main_task"')
    parser.add_argument('--dataset', default='mnist', type=str, help='the dataset which the experiment is based on')
    parser.add_argument('--num_exp', default=10, type=int , help='the number of random experiments')
    parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
    parser.add_argument('--epochs', default=100, type=int, help='')
    parser.add_argument('--batch_size', default=2048, type=int, help='')
    parser.add_argument('--early_stop', default=False, type=bool, help='whether to use early stop')
    parser.add_argument('--early_stop_param', default=0.0001, type=float, help='stop training when the loss <= early_stop_param')
    parser.add_argument('--seed', default=100, type=int, help='random seed')
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--gpu', default=0, type=int, help='gpu id')
    # defense
    parser.add_argument('--apply_trainable_layer', default=False, type=bool, help='whether to use trainable layer in active party')
    parser.add_argument('--apply_laplace', default=False, type=bool, help='whether to use dp-laplace')
    parser.add_argument('--apply_gaussian', default=False, type=bool, help='whether to use dp-gaussian')
    parser.add_argument('--dp_strength', default=0, type=float, help='the parameter of dp defense')
    parser.add_argument('--apply_grad_spar', default=False, type=bool, help='whether to use gradient sparsification')
    parser.add_argument('--grad_spars', default=0, type=float, help='the parameter of gradient sparsification')
    parser.add_argument('--apply_encoder', default=False, type=bool, help='whether to use CoAE')
    # parser.add_argument('--apply_random_encoder', default=False, type=bool, help='whether to use CoAE')
    # parser.add_argument('--apply_adversarial_encoder', default=False, type=bool, help='whether to use AAE')

    # parser.add_argument('--apply_certify', default=0, type=int, help='whether to use certify')
    # parser.add_argument('--certify_M', default=1000, type=int, help='number of voters in CertifyFL')
    # parser.add_argument('--certify_start_epoch', default=0, type=int, help='number of epoch that start certify process')

    parser.add_argument('--apply_marvell', default=False, type=bool, help='whether to use marvell(optimal gaussian noise)')
    parser.add_argument('--marvell_s', default=1, type=int, help='scaler of bound in MARVELL')

    # # defense methods given in MC
    # parser.add_argument('--apply_ppdl', help='turn_on_privacy_preserving_deep_learning', type=bool, default=False)
    # parser.add_argument('--ppdl_theta_u', help='theta-u parameter for defense privacy-preserving deep learning', type=float, default=0.5)
    # # parser.add_argument('--apply_gc', help='turn_on_gradient_compression', type=bool, default=False)
    # parser.add_argument('--gc_preserved_percent', help='preserved-percent parameter for defense gradient compression', type=float, default=0.1)
    # parser.add_argument('--apply_lap_noise', help='turn_on_lap_noise', type=bool, default=False)
    # parser.add_argument('--noise_scale', help='noise-scale parameter for defense noisy gradients', type=float, default=1e-3)
    parser.add_argument('--apply_discrete_gradients', default=False, type=bool, help='whether to use Discrete Gradients')
    parser.add_argument('--discrete_gradients_bins', default=12, type=int, help='number of bins for discrete gradients')
    parser.add_argument('--discrete_gradients_bound', default=3e-4, type=float, help='value of bound for discrete gradients')
    
    parser.add_argument('--apply_mi', default=False, type=bool, help='wheather to use MutualInformation-loss instead of CrossEntropy-loss')
    parser.add_argument('--mi_loss_lambda', default=1.0, type=float, help='the parameter for MutualInformation-loss')
    parser.add_argument('--apply_mid', default=False, type=bool, help='wheather to use MID for protection')
    parser.add_argument('--mid_tau', default=0.1, type=float, help='the parameter for MID')
    parser.add_argument('--mid_loss_lambda', default=0.003, type=float, help='the parameter for MID')
    parser.add_argument('--apply_distance_correlation', default=False, type=bool, help='wheather to use Distance Correlation for protection')
    parser.add_argument('--distance_correlation_lambda', default=0.003, type=float, help='the parameter for Distance Correlation')
    parser.add_argument('--apply_grad_perturb', default=False, type=bool, help='wheather to use GradPerturb for protection')
    parser.add_argument('--perturb_epsilon', default=1.0, type=float, help='the parameter DP-epsilon for GradPerturb')
    parser.add_argument('--apply_RRwithPrior', default=False, type=bool, help='wheather to use RRwithPrior for protection')
    parser.add_argument('--RRwithPrior_epsilon', default=1.0, type=float, help='the parameter DP-epsilon for RRwithPrior')

    args = parser.parse_args()
    if args.exp_type == 'attack':
        print("attack type")
        set_seed(args.seed)
        args.model = models_dict[args.dataset]
        args.epochs = epochs_dict[args.dataset]
        args.encoder = None
        args.ae_lambda = None

        if args.device == 'cuda':
            cuda_id = args.gpu
            torch.cuda.set_device(cuda_id)
        print(f'running on cuda{torch.cuda.current_device()}')

        if args.dataset == 'cifar100':
            args.dst = datasets.CIFAR100("../../../share_dataset/", download=True)
        elif args.dataset == 'cifar10':
            args.dst = datasets.CIFAR10("../../../share_dataset/", download=True)
        elif args.dataset == 'mnist':
            args.dst = datasets.MNIST("~/.torch", download=True)
        elif args.dataset == 'nuswide':
            args.dst = None
        
        # attention, change to binary classification
        args.num_class_list = [2]

        args.batch_size_list = [2048]
        # args.batch_size_list = [1]
        
        if args.num_class_list[0] == 2:
            ae_name_list = ['autoencoder_2_1.0_1636175704','autoencoder_2_0.5_1636175420',\
                            'autoencoder_2_0.1_1636175237','autoencoder_2_0.0_1636174878']

        args.exp_res_dir = f'exp_result_norm_scoring/{args.dataset}/'
        temp = f'exp_result_direction_scoring/{args.dataset}/'
        # all the route can be concatenated
        if args.apply_trainable_layer:
            args.exp_res_dir += '_top_model/'
            temp += '_top_model/'
        if args.apply_mid:
            args.exp_res_dir += 'MID/'
            temp += 'MID/'
        if args.apply_mi:
            args.exp_res_dir += 'MI/'
            temp += 'MI/'
        if args.apply_RRwithPrior:
            args.exp_res_dir += 'RRwithPrior/'
            temp += 'RRwithPrior/'
        if args.apply_distance_correlation:
            args.exp_res_dir += 'DistanceCorrelation/'
            temp += 'DistanceCorrelation/'
        if args.apply_grad_perturb:
            args.exp_res_dir += 'GradientPerturb/'
            temp += 'GradientPerturb/'
        if args.apply_laplace:
            args.exp_res_dir += 'Laplace/'
            temp += 'Laplace/'
        elif args.apply_gaussian:
            args.exp_res_dir += 'Gaussian/'
            temp += 'Gaussian/'
        elif args.apply_grad_spar:
            args.exp_res_dir += 'GradientSparsification/'
            temp += 'GradientSparsification/'
        if args.apply_encoder:
            if args.apply_discrete_gradients:
                args.exp_res_dir += 'DCAE/'
                temp += 'DCAE/'
            else:
                args.exp_res_dir += 'CAE/'
                temp += 'CAE/'
        elif args.apply_discrete_gradients:
            args.exp_res_dir += 'DiscreteGradients/'
            temp += 'DiscreteGradients/'
        if args.apply_marvell:
            args.exp_res_dir += 'MARVELL/'
            temp += 'MARVELL/'
        if not os.path.exists(args.exp_res_dir):
            os.makedirs(args.exp_res_dir)
        if not os.path.exists(temp):
            os.makedirs(temp)
        filename = f'attack_task_acc.txt'
        args.exp_res_path = args.exp_res_dir + filename
        temp = temp + filename
        args.exp_res_path = [args.exp_res_path, temp]
        
        config_text = f'model={args.model},lr={args.lr},epochs={args.epochs},early_stop={args.early_stop},batch_size={args.batch_size_list[0]},num_classes={args.num_class_list[0]}'
        append_exp_res(args.exp_res_path[0], config_text)
        append_exp_res(args.exp_res_path[1], config_text)
        if args.apply_encoder:
            for ae_name in ae_name_list:
                args.ae_lambda = ae_name.split('_')[2]
                dim = args.num_class_list[0]
                encoder = AutoEncoder(input_dim=dim, encode_dim=2+dim * 6).to(args.device)
                encoder.load_model(f"./trained_models/{ae_name}", target_device=args.device)
                args.encoder = encoder
                label_leakage = marvell_scoring_attack.ScoringAttack(args)
                label_leakage.train()
        elif args.apply_laplace or args.apply_gaussian:
            # dp_strength_list = [0.00005, 0.0001, 0.0005, 0.001, 0.01, 0.1]
            dp_strength_list = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1,0.5, 1.0]
            for dp_strength in dp_strength_list:
                args.dp_strength = dp_strength
                label_leakage = marvell_scoring_attack.ScoringAttack(args)
                label_leakage.train()
        elif args.apply_grad_spar:
            gradient_sparsification_list = [90, 95, 96, 97, 98, 99]
            for grad_spars in gradient_sparsification_list:
                args.grad_spars = grad_spars
                label_leakage = marvell_scoring_attack.ScoringAttack(args)
                label_leakage.train()
        elif args.apply_mid:
            mid_lambda_list = [0.0,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,10,20]
            # mid_lambda_list = [1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,2,5,10,20,20,10,5,2,1,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9]
            for mid_loss_lambda in mid_lambda_list:
                args.mid_loss_lambda = mid_loss_lambda
                label_leakage = marvell_scoring_attack.ScoringAttack(args)
                label_leakage.train()
        elif args.apply_marvell:
            marvell_s_list = [10,5,2,1,0.1]
            for marvell_s in marvell_s_list:
                args.marvell_s = marvell_s
                label_leakage = marvell_scoring_attack.ScoringAttack(args)
                label_leakage.train()
        elif args.apply_RRwithPrior:
            epsilon_list = [8.0,7.0,6.0,5.0,4.0,3.0,2.0,1.0]
            for RRwithPrior_epsilon in epsilon_list:
                args.RRwithPrior_epsilon = RRwithPrior_epsilon
                label_leakage = marvell_scoring_attack.ScoringAttack(args)
                label_leakage.train()
        elif args.apply_distance_correlation:
            distance_correlation_lambda_list = [1e-1,1e-2,3e-3,1e-3,1e-4,1e-5]
            for distance_correlation_lambda in distance_correlation_lambda_list:
                args.distance_correlation_lambda = distance_correlation_lambda
                label_leakage = marvell_scoring_attack.ScoringAttack(args)
                label_leakage.train()
        elif args.apply_grad_perturb:
            perturb_list = [0.1,0.3,1.0,3.0,10.0]
            for perturb_epsilon in perturb_list:
                args.perturb_epsilon = perturb_epsilon
                label_leakage = marvell_scoring_attack.ScoringAttack(args)
                label_leakage.train()
        else:
            if args.apply_mi:
                label_leakage = marvell_scoring_attack.ScoringAttack(args)
                label_leakage.train()
            else:
                label_leakage = marvell_scoring_attack.ScoringAttack(args)
                label_leakage.train()
    
    elif args.exp_type == 'main_task':
        print("main_task type")
        set_seed(args.seed)

        if args.device == 'cuda':
            cuda_id = args.gpu
            torch.cuda.set_device(cuda_id)
        print(f'running on cuda{torch.cuda.current_device()}')

        if args.dataset == "cifar100":
            half_dim = 16
            num_classes = 2
            train_dst = datasets.CIFAR100("../../../share_dataset/", download=True, train=True, transform=transform)
            data, label = fetch_data_and_label(train_dst, num_classes)
            train_dst = SimpleDataset(data, label)
            test_dst = datasets.CIFAR100("../../../share_dataset/", download=True, train=False, transform=transform)
            data, label = fetch_data_and_label(test_dst, num_classes)
            test_dst = SimpleDataset(data, label)
        elif args.dataset == "cifar10":
            half_dim = 16
            num_classes = 2
            train_dst = datasets.CIFAR10("../../../share_dataset/", download=True, train=True, transform=transform)
            data, label = fetch_data_and_label(train_dst, num_classes)
            train_dst = SimpleDataset(data, label)
            test_dst = datasets.CIFAR10("../../../share_dataset/", download=True, train=False, transform=transform)
            data, label = fetch_data_and_label(test_dst, num_classes)
            test_dst = SimpleDataset(data, label)
        elif args.dataset == "mnist":
            half_dim = 14
            num_classes = 2
            train_dst = datasets.MNIST("~/.torch", download=True, train=True, transform=transform_fn)
            data, label = fetch_data_and_label(train_dst, num_classes)
            train_dst = SimpleDataset(data, label)
            test_dst = datasets.MNIST("~/.torch", download=True, train=False, transform=transform_fn)
            data, label = fetch_data_and_label(test_dst, num_classes)
            test_dst = SimpleDataset(data, label)
        elif args.dataset == 'nuswide':
            half_dim = [634, 1000]
            num_classes = 2
            train_dst = NUSWIDEDataset('../../../share_dataset/NUS_WIDE', 'train')
            test_dst = NUSWIDEDataset('../../../share_dataset/NUS_WIDE', 'test')
        
        args.train_dataset = train_dst
        args.val_dataset = test_dst
        args.half_dim = half_dim
        args.num_classes = num_classes

        args.models_dict = {"mnist": MLP2,
                "cifar100": resnet18,
                "cifar10": resnet18,
                "nuswide": MLP2,
                "classifier": None}

        # load trained encoder
        # filename format: autoencoder_<num-classes>_<lambda>_<timestamp>    
        ae_name_list = ['autoencoder_2_1.0_1636175704','autoencoder_2_0.5_1636175420',\
                        'autoencoder_2_0.1_1636175237','autoencoder_2_0.0_1636174878']

        path = f'exp_result_norm_scoring/{args.dataset}/'
        temp = f'exp_result_direction_scoring/{args.dataset}/'
        if args.apply_trainable_layer:
            path += '_top_model/'
            temp += '_top_model/'
        if args.apply_mid:
            path += 'MID/'
            temp += 'MID/'
        if args.apply_mi:
            path += 'MI/'
            temp += 'MI/'
        if args.apply_RRwithPrior:
            path += 'RRwithPrior/'
            temp += 'RRwithPrior/'
        if args.apply_distance_correlation:
            path += 'DistanceCorrelation/'
            temp += 'DistanceCorrelation/'
        if args.apply_grad_perturb:
            path += 'GradientPerturb/'
            temp += 'GradientPerturb/'
        if args.apply_laplace:
            path += 'Laplace/'
            temp += 'Laplace/'
        elif args.apply_gaussian:
            path += 'Gaussian/'
            temp += 'Gaussian/'
        elif args.apply_grad_spar:
            path += 'GradientSparsification/'
            temp += 'GradientSparsification/'
        if args.apply_encoder:
            if args.apply_discrete_gradients:
                path += 'DCAE/'
                temp += 'DCAE/'
            else:
                path += 'CAE/'
                temp += 'CAE/'
        elif args.apply_discrete_gradients:
            path += 'DiscreteGradients/'
            temp += 'DiscreteGradients/'
        if args.apply_marvell:
            path += 'MARVELL/'
            temp += 'MARVELL/'
        if not os.path.exists(path):
            os.makedirs(path)
        if not os.path.exists(temp):
            os.makedirs(temp)
        path += 'main_task_acc.txt'
        temp += 'main_task_acc.txt'
        path = [path, temp]
        num_exp = 10
        num_exp = 1

        args.encoder = None
        # Model(pred_Z) for mid
        args.mid_model = None
        args.mid_enlarge_model = None
        if args.apply_mid:
            args.mid_model = MID_layer(args.num_classes, args.num_classes)
            args.mid_enlarge_model = MID_enlarge_layer(args.num_classes, args.num_classes*2)

        if args.apply_encoder:
            for ae_name in ae_name_list:
                test_auc_list = []
                test_acc_list = []
                for i in range(num_exp):
                    dim = args.num_classes
                    encoder = AutoEncoder(input_dim=dim, encode_dim=2 + dim * 6).to(args.device)
                    encoder.load_model(f"./trained_models/{ae_name}", target_device=args.device)
                    args.encoder = encoder
                    _lambda = ae_name.split('_')[2]
                    print(f'num_exp:{i + 1}, epochs:{args.epochs}, batchsize:{args.batch_size}, lambda:{_lambda}')
                    marvell_vfl = marvell_scoring_main_auc.VFLmodel_AUC(args)
                    test_auc, test_acc = marvell_vfl.train()
                    test_auc_list.append(test_auc)
                    test_acc_list.append(test_acc)
                append_exp_res(path[0], str(_lambda) + ' ' + str(np.mean(test_auc_list))+ ' AUC ' + str(test_auc_list) + ' ' + str(np.max(test_auc_list)))
                append_exp_res(path[1], str(_lambda) + ' ' + str(np.mean(test_auc_list))+ ' AUC ' + str(test_auc_list) + ' ' + str(np.max(test_auc_list)))
                append_exp_res(path[0], str(_lambda) + ' ' + str(np.mean(test_acc_list))+ ' ACC ' + str(test_acc_list) + ' ' + str(np.max(test_acc_list)))
                append_exp_res(path[1], str(_lambda) + ' ' + str(np.mean(test_acc_list))+ ' ACC ' + str(test_acc_list) + ' ' + str(np.max(test_acc_list)))
        elif args.apply_laplace or args.apply_gaussian:
            # dp_strength_list = [0.00005, 0.0001, 0.0005, 0.001, 0.01, 0.1]
            dp_strength_list = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1,0.5, 1.0]
            for dp_strength in dp_strength_list:
                test_auc_list = []
                test_acc_list = []
                for i in range(num_exp):
                    args.dp_strength = dp_strength
                    marvell_vfl = marvell_scoring_main_auc.VFLmodel_AUC(args)
                    # test_auc = marvell_vfl.train()
                    # test_auc_list.append(test_auc)
                    test_auc, test_acc = marvell_vfl.train()
                    test_auc_list.append(test_auc)
                    test_acc_list.append(test_acc)
                append_exp_res(path[0], str(dp_strength) + ' ' + str(np.mean(test_auc_list))+ ' AUC ' + str(test_auc_list) + ' ' + str(np.max(test_auc_list)))
                append_exp_res(path[1], str(dp_strength) + ' ' + str(np.mean(test_auc_list))+ ' AUC ' + str(test_auc_list) + ' ' + str(np.max(test_auc_list)))
                append_exp_res(path[0], str(dp_strength) + ' ' + str(np.mean(test_acc_list))+ ' ACC ' + str(test_acc_list) + ' ' + str(np.max(test_acc_list)))
                append_exp_res(path[1], str(dp_strength) + ' ' + str(np.mean(test_acc_list))+ ' ACC ' + str(test_acc_list) + ' ' + str(np.max(test_acc_list)))
        elif args.apply_grad_spar:
            # gradient_sparsification_list = [90, 95, 96, 97, 98, 99]
            # for grad_spars in gradient_sparsification_list:
            #     test_auc_list = []
            #     test_acc_list = []
            #     for i in range(num_exp):
            #         args.grad_spars = grad_spars
            #         marvell_vfl = marvell_scoring_main_auc.VFLmodel_AUC(args)
            #         # test_auc = marvell_vfl.train()
            #         # test_auc_list.append(test_auc)
            #         test_auc, test_acc = marvell_vfl.train()
            #         test_auc_list.append(test_auc)
            #         test_acc_list.append(test_acc)
            #     append_exp_res(path[0], str(grad_spars) + ' ' + str(np.mean(test_auc_list))+ ' AUC ' + str(test_auc_list) + ' ' + str(np.max(test_auc_list)))
            #     append_exp_res(path[1], str(grad_spars) + ' ' + str(np.mean(test_auc_list))+ ' AUC ' + str(test_auc_list) + ' ' + str(np.max(test_auc_list)))
            #     append_exp_res(path[0], str(grad_spars) + ' ' + str(np.mean(test_acc_list))+ ' ACC ' + str(test_acc_list) + ' ' + str(np.max(test_acc_list)))
            #     append_exp_res(path[1], str(grad_spars) + ' ' + str(np.mean(test_acc_list))+ ' ACC ' + str(test_acc_list) + ' ' + str(np.max(test_acc_list)))
            test_auc_list = []
            test_acc_list = []
            grad_spars = args.grad_spars
            for i in range(num_exp):
                # args.grad_spars = grad_spars
                # grad_spars = args.grad_spars
                marvell_vfl = marvell_scoring_main_auc.VFLmodel_AUC(args)
                # test_auc = marvell_vfl.train()
                # test_auc_list.append(test_auc)
                test_auc, test_acc = marvell_vfl.train()
                test_auc_list.append(test_auc)
                test_acc_list.append(test_acc)
            append_exp_res(path[0], str(grad_spars) + ' ' + str(np.mean(test_auc_list))+ ' AUC ' + str(test_auc_list) + ' ' + str(np.max(test_auc_list)))
            append_exp_res(path[1], str(grad_spars) + ' ' + str(np.mean(test_auc_list))+ ' AUC ' + str(test_auc_list) + ' ' + str(np.max(test_auc_list)))
            append_exp_res(path[0], str(grad_spars) + ' ' + str(np.mean(test_acc_list))+ ' ACC ' + str(test_acc_list) + ' ' + str(np.max(test_acc_list)))
            append_exp_res(path[1], str(grad_spars) + ' ' + str(np.mean(test_acc_list))+ ' ACC ' + str(test_acc_list) + ' ' + str(np.max(test_acc_list)))
        elif args.apply_mid:
            # mid_lambda_list = [1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1]
            # mid_lambda_list = [1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1]
            mid_lambda_list = [0.0,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,10]
            # mid_lambda_list = [1e-1,1,2,5,10,20]
            for mid_loss_lambda in mid_lambda_list:
                print("mid_loss_lambda", mid_loss_lambda)
                test_auc_list = []
                test_acc_list = []
                for i in range(num_exp):
                    args.mid_loss_lambda = mid_loss_lambda
                    marvell_vfl = marvell_scoring_main_auc.VFLmodel_AUC(args)
                    # test_auc = marvell_vfl.train()
                    # test_auc_list.append(test_auc)
                    test_auc, test_acc = marvell_vfl.train()
                    test_auc_list.append(test_auc)
                    test_acc_list.append(test_acc)
                append_exp_res(path[0], str(args.mid_loss_lambda) + ' ' + str(np.mean(test_auc_list))+ ' AUC ' + str(test_auc_list) + ' ' + str(np.max(test_auc_list)))
                append_exp_res(path[1], str(args.mid_loss_lambda) + ' ' + str(np.mean(test_auc_list))+ ' AUC ' + str(test_auc_list) + ' ' + str(np.max(test_auc_list)))
                append_exp_res(path[0], str(args.mid_loss_lambda) + ' ' + str(np.mean(test_acc_list))+ ' ACC ' + str(test_acc_list) + ' ' + str(np.max(test_acc_list)))
                append_exp_res(path[1], str(args.mid_loss_lambda) + ' ' + str(np.mean(test_acc_list))+ ' ACC ' + str(test_acc_list) + ' ' + str(np.max(test_acc_list)))
        elif args.apply_marvell:
            marvell_s_list = [10,5,2,1,0.1]
            for marvell_s in marvell_s_list:
                test_auc_list = []
                test_acc_list = []
                for i in range(num_exp):
                    args.marvell_s = marvell_s
                    marvell_vfl = marvell_scoring_main_auc.VFLmodel_AUC(args)
                    # test_auc = marvell_vfl.train()
                    # test_auc_list.append(test_auc)
                    test_auc, test_acc = marvell_vfl.train()
                    test_auc_list.append(test_auc)
                    test_acc_list.append(test_acc)
                append_exp_res(path[0], str(args.marvell_s) + ' ' + str(np.mean(test_auc_list))+ ' AUC ' + str(test_auc_list) + ' ' + str(np.max(test_auc_list)))
                append_exp_res(path[1], str(args.marvell_s) + ' ' + str(np.mean(test_auc_list))+ ' AUC ' + str(test_auc_list) + ' ' + str(np.max(test_auc_list)))
                append_exp_res(path[0], str(args.marvell_s) + ' ' + str(np.mean(test_acc_list))+ ' ACC ' + str(test_acc_list) + ' ' + str(np.max(test_acc_list)))
                append_exp_res(path[1], str(args.marvell_s) + ' ' + str(np.mean(test_acc_list))+ ' ACC ' + str(test_acc_list) + ' ' + str(np.max(test_acc_list)))
        elif args.apply_RRwithPrior:
            epsilon_list = [8.0,7.0,6.0,5.0,4.0,3.0,2.0,1.0]
            for RRwithPrior_epsilon in epsilon_list:
                test_auc_list = []
                test_acc_list = []
                for i in range(num_exp):
                    args.RRwithPrior_epsilon = RRwithPrior_epsilon
                    marvell_vfl = marvell_scoring_main_auc.VFLmodel_AUC(args)
                    test_auc, test_acc = marvell_vfl.train()
                    test_auc_list.append(test_auc)
                    test_acc_list.append(test_acc)
                append_exp_res(path[0], str(args.RRwithPrior_epsilon) + ' ' + str(np.mean(test_auc_list))+ ' AUC ' + str(test_auc_list) + ' ' + str(np.max(test_auc_list)))
                append_exp_res(path[1], str(args.RRwithPrior_epsilon) + ' ' + str(np.mean(test_auc_list))+ ' AUC ' + str(test_auc_list) + ' ' + str(np.max(test_auc_list)))
                append_exp_res(path[0], str(args.RRwithPrior_epsilon) + ' ' + str(np.mean(test_acc_list))+ ' ACC ' + str(test_acc_list) + ' ' + str(np.max(test_acc_list)))
                append_exp_res(path[1], str(args.RRwithPrior_epsilon) + ' ' + str(np.mean(test_acc_list))+ ' ACC ' + str(test_acc_list) + ' ' + str(np.max(test_acc_list)))
        elif args.apply_distance_correlation:
            distance_correlation_lambda_list = [1e-1,1e-2,3e-3,1e-3,1e-4,1e-5]
            for distance_correlation_lambda in distance_correlation_lambda_list:
                test_auc_list = []
                test_acc_list = []
                for i in range(num_exp):
                    args.distance_correlation_lambda = distance_correlation_lambda
                    marvell_vfl = marvell_scoring_main_auc.VFLmodel_AUC(args)
                    # test_auc = marvell_vfl.train()
                    # test_auc_list.append(test_auc)
                    test_auc, test_acc = marvell_vfl.train()
                    test_auc_list.append(test_auc)
                    test_acc_list.append(test_acc)
                append_exp_res(path[0], str(args.distance_correlation_lambda) + ' ' + str(np.mean(test_auc_list))+ ' AUC ' + str(test_auc_list) + ' ' + str(np.max(test_auc_list)))
                append_exp_res(path[1], str(args.distance_correlation_lambda) + ' ' + str(np.mean(test_auc_list))+ ' AUC ' + str(test_auc_list) + ' ' + str(np.max(test_auc_list)))
                append_exp_res(path[0], str(args.distance_correlation_lambda) + ' ' + str(np.mean(test_acc_list))+ ' ACC ' + str(test_acc_list) + ' ' + str(np.max(test_acc_list)))
                append_exp_res(path[1], str(args.distance_correlation_lambda) + ' ' + str(np.mean(test_acc_list))+ ' ACC ' + str(test_acc_list) + ' ' + str(np.max(test_acc_list)))
        elif args.apply_grad_perturb:
            perturb_list = [0.1,0.3,1.0,3.0,10.0]
            for perturb_epsilon in perturb_list:
                test_auc_list = []
                test_acc_list = []
                for i in range(num_exp):
                    args.perturb_epsilon = perturb_epsilon
                    marvell_vfl = marvell_scoring_main_auc.VFLmodel_AUC(args)
                    # test_auc = marvell_vfl.train()
                    # test_auc_list.append(test_auc)
                    test_auc, test_acc = marvell_vfl.train()
                    test_auc_list.append(test_auc)
                    test_acc_list.append(test_acc)
                append_exp_res(path[0], str(args.perturb_epsilon) + ' ' + str(np.mean(test_auc_list))+ ' AUC ' + str(test_auc_list) + ' ' + str(np.max(test_auc_list)))
                append_exp_res(path[1], str(args.perturb_epsilon) + ' ' + str(np.mean(test_auc_list))+ ' AUC ' + str(test_auc_list) + ' ' + str(np.max(test_auc_list)))
                append_exp_res(path[0], str(args.perturb_epsilon) + ' ' + str(np.mean(test_acc_list))+ ' ACC ' + str(test_acc_list) + ' ' + str(np.max(test_acc_list)))
                append_exp_res(path[1], str(args.perturb_epsilon) + ' ' + str(np.mean(test_acc_list))+ ' ACC ' + str(test_acc_list) + ' ' + str(np.max(test_acc_list)))
        else:
            test_auc_list = []
            test_acc_list = []
            for _ in range(num_exp):
                marvell_vfl = marvell_scoring_main_auc.VFLmodel_AUC(args)
                # test_auc = marvell_vfl.train()
                # test_auc_list.append(test_auc)
                test_auc, test_acc = marvell_vfl.train()
                test_auc_list.append(test_auc)
                test_acc_list.append(test_acc)
            if args.apply_mi:
                append_exp_res(path[0], str(args.mi_loss_lambda) + ' ' + str(np.mean(test_auc_list))+ ' AUC ' + str(test_auc_list) + ' ' + str(np.max(test_auc_list)))
                append_exp_res(path[1], str(args.mi_loss_lambda) + ' ' + str(np.mean(test_auc_list))+ ' AUC ' + str(test_auc_list) + ' ' + str(np.max(test_auc_list)))
                append_exp_res(path[0], str(args.mi_loss_lambda) + ' ' + str(np.mean(test_acc_list))+ ' ACC ' + str(test_acc_list) + ' ' + str(np.max(test_acc_list)))
                append_exp_res(path[1], str(args.mi_loss_lambda) + ' ' + str(np.mean(test_acc_list))+ ' ACC ' + str(test_acc_list) + ' ' + str(np.max(test_acc_list)))
            # elif args.apply_distance_correlation:
            #     append_exp_res(path[0], str(args.distance_correlation_lambda) + ' ' + str(np.mean(test_auc_list))+ ' AUC ' + str(test_auc_list) + ' ' + str(np.max(test_auc_list)))
            #     append_exp_res(path[1], str(args.distance_correlation_lambda) + ' ' + str(np.mean(test_auc_list))+ ' AUC ' + str(test_auc_list) + ' ' + str(np.max(test_auc_list)))
            #     append_exp_res(path[0], str(args.distance_correlation_lambda) + ' ' + str(np.mean(test_acc_list))+ ' ACC ' + str(test_acc_list) + ' ' + str(np.max(test_acc_list)))
            #     append_exp_res(path[1], str(args.distance_correlation_lambda) + ' ' + str(np.mean(test_acc_list))+ ' ACC ' + str(test_acc_list) + ' ' + str(np.max(test_acc_list)))
            elif args.apply_discrete_gradients:
                append_exp_res(path[0], str(args.discrete_gradients_bins) + ' ' + str(np.mean(test_auc_list))+ ' AUC ' + str(test_auc_list) + ' ' + str(np.max(test_auc_list)))
                append_exp_res(path[1], str(args.discrete_gradients_bins) + ' ' + str(np.mean(test_auc_list))+ ' AUC ' + str(test_auc_list) + ' ' + str(np.max(test_auc_list)))
                append_exp_res(path[0], str(args.discrete_gradients_bins) + ' ' + str(np.mean(test_acc_list))+ ' ACC ' + str(test_acc_list) + ' ' + str(np.max(test_acc_list)))
                append_exp_res(path[1], str(args.discrete_gradients_bins) + ' ' + str(np.mean(test_acc_list))+ ' ACC ' + str(test_acc_list) + ' ' + str(np.max(test_acc_list)))
            else:
                append_exp_res(path[0], "bs|num_class|recovery_rate," + str(args.batch_size) + '|' + str(num_classes) + ' ' + str(np.mean(test_auc_list))+ ' AUC ' + str(test_auc_list) + ' ' + str(np.max(test_auc_list)))
                append_exp_res(path[1], "bs|num_class|recovery_rate," + str(args.batch_size) + '|' + str(num_classes) + ' ' + str(np.mean(test_auc_list))+ ' AUC ' + str(test_auc_list) + ' ' + str(np.max(test_auc_list)))
                append_exp_res(path[0], "bs|num_class|recovery_rate," + str(args.batch_size) + '|' + str(num_classes) + ' ' + str(np.mean(test_acc_list))+ ' ACC ' + str(test_acc_list) + ' ' + str(np.max(test_acc_list)))
                append_exp_res(path[1], "bs|num_class|recovery_rate," + str(args.batch_size) + '|' + str(num_classes) + ' ' + str(np.mean(test_acc_list))+ ' ACC ' + str(test_acc_list) + ' ' + str(np.max(test_acc_list)))
                