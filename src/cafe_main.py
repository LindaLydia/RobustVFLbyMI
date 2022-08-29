import logging

import torch
import time
import argparse
import pprint
from models.autoencoder import AutoEncoder
from vfl_dlg import *
import cafe_attacker
from models.vision import *
from utils import *

models_dict = {"mnist": 'MLP2',
               "cifar10": 'resnet18',
               "cifar100": 'resnet18',
               "nuswide": 'MLP2',
               "classifier": None}
epochs_dict = {"mnist": 200,
              "cifar10": 100,
              "cifar100": 100,
              "nuswide": 200}

def set_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='mnist', type=str, help='the dataset which the experiment is based on')
    parser.add_argument('--num_exp', default=10, type=int , help='the number of random experiments')
    parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
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


    args = parser.parse_args()
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
        args.num_class_list = [20] #[5, 10, 15, 20, 40, 60, 80, 100]
    elif args.dataset == 'cifar10':
        args.dst = datasets.CIFAR10("../../../share_dataset/", download=True)
        args.num_class_list = [10] #[5, 10, 15, 20, 40, 60, 80, 100]
    elif args.dataset == 'mnist':
        args.dst = datasets.MNIST("~/.torch", download=True)
        args.num_class_list = [10] #[2, 3, 4, 5, 6, 7, 8, 9, 10]
    elif args.dataset == 'nuswide':
        args.dst = None
        args.num_class_list = [5] #[2, 4, 8, 16, 20, 40, 60, 81]
    
    # # attention, change to binary classification
    # args.num_class_list = [2]

    args.batch_size_list = [256]
    # args.batch_size_list = [1]
    
    if args.num_class_list[0] == 2:
        ae_name_list = ['autoencoder_2_1.0_1636175704','autoencoder_2_0.5_1636175420',\
                        'autoencoder_2_0.1_1636175237','autoencoder_2_0.0_1636174878']
    elif args.dataset == 'nuswide':
        ae_name_list = ['autoencoder_5_1.0_1630879452', 'autoencoder_5_0.5_1630891447',\
                        'autoencoder_5_0.1_1630895642', 'autoencoder_5_0.05_1630981788', 'autoencoder_5_0.0_1631059899']
        # ae_name_list = ['negative/autoencoder_5_0.5_1647017897', 'negative/autoencoder_5_1.0_1647017945']
    elif args.dataset == 'mnist' or args.dataset == 'cifar10':
        ae_name_list = ['autoencoder_10_1.0_1642396548', 'autoencoder_10_0.5_1642396797',\
                        'autoencoder_10_0.1_1642396928', 'autoencoder_10_0.0_1631093149']
        # ae_name_list = ['negative/autoencoder_10_0.5_1645982479','negative/autoencoder_10_1.0_1645982367']
    elif args.dataset == 'cifar100':
        ae_name_list = ['autoencoder_20_1.0_1645374675','autoencoder_20_0.5_1645374585',\
                        'autoencoder_20_0.1_1645374527','autoencoder_20_0.05_1645374482','autoencoder_20_0.0_1645374739']
        # ae_name_list = ['negative/autoencoder_20_0.5_1647127262', 'negative/autoencoder_20_1.0_1647127164']



    args.exp_res_dir = f'exp_result_data_recovery_2048/{args.dataset}/'
    # args.exp_res_dir = f'exp_result_data_recovery_binary/{args.dataset}/'
    # all the route can be concatenated
    if args.apply_trainable_layer:
        args.exp_res_dir += '_top_model/'
    if args.apply_mid:
        args.exp_res_dir += 'MID/'
    if args.apply_mi:
        args.exp_res_dir += 'MI/'
    if args.apply_distance_correlation:
        args.exp_res_dir += 'DistanceCorrelation/'
    if args.apply_laplace:
        args.exp_res_dir += 'Laplace/'
    elif args.apply_gaussian:
        args.exp_res_dir += 'Gaussian/'
    elif args.apply_grad_spar:
        args.exp_res_dir += 'GradientSparsification/'
    if args.apply_encoder:
        if args.apply_discrete_gradients:
            args.exp_res_dir += 'DCAE/'
        else:
            args.exp_res_dir += 'CAE/'
    elif args.apply_discrete_gradients:
        args.exp_res_dir += 'DiscreteGradients/'
    if not os.path.exists(args.exp_res_dir):
        os.makedirs(args.exp_res_dir)
    filename = f'attack_task_acc.txt'
    args.exp_res_path = args.exp_res_dir + filename
    
    config_text = f'model={args.model},lr={args.lr},epochs={args.epochs},early_stop={args.early_stop},batch_size={args.batch_size_list[0]},num_classes={args.num_class_list[0]}'
    append_exp_res(args.exp_res_path, config_text)
    if args.apply_encoder:
        for ae_name in ae_name_list:
            args.ae_lambda = ae_name.split('_')[2]
            dim = args.num_class_list[0]
            encoder = AutoEncoder(input_dim=dim, encode_dim=2+dim * 6).to(args.device)
            encoder.load_model(f"./trained_models/{ae_name}", target_device=args.device)
            args.encoder = encoder
            label_leakage = cafe_attacker.CAFEattacker(args)
            label_leakage.train()
    elif args.apply_laplace or args.apply_gaussian:
        # dp_strength_list = [0.00005, 0.0001, 0.0005, 0.001, 0.01, 0.1]
        dp_strength_list = [0.0001, 0.001, 0.01, 0.1]
        for dp_strength in dp_strength_list:
            args.dp_strength = dp_strength
            label_leakage = cafe_attacker.CAFEattacker(args)
            label_leakage.train()
    elif args.apply_grad_spar:
        gradient_sparsification_list = [90, 95, 96, 97, 98, 99]
        for grad_spars in gradient_sparsification_list:
            args.grad_spars = grad_spars
            label_leakage = cafe_attacker.CAFEattacker(args)
            label_leakage.train()
    elif args.apply_mid:
        mid_lambda_list = [1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1]
        for mid_loss_lambda in mid_lambda_list:
            args.mid_loss_lambda = mid_loss_lambda
            label_leakage = cafe_attacker.CAFEattacker(args)
            label_leakage.train()
    else:
        label_leakage = cafe_attacker.CAFEattacker(args)
        label_leakage.train()

