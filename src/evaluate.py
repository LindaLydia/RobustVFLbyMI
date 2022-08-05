import math
import os
from matplotlib import projections
from matplotlib.legend import Legend

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib.ticker as mtick
import numpy as np

from utils import draw_line_chart, remove_exponent

NO_DEFENSE = 'no defense'
EPSILON = 1e-3

def read_values(attack_path,main_path,defense_name):
    _rec_rate_list, _acc_list, _param_list = [], [], []
    with open(attack_path, 'r') as f:
        with open(main_path, 'r') as f1:
            for line in f.readlines():
                line_split = line.strip('\n').split(' ')
                if len(line_split) == 1:
                    continue
                param = line_split[0]
                rec_rate = line_split[1]
                rec_rate = float(rec_rate) * 100
                _param_list.append(param)
                _rec_rate_list.append(rec_rate)
            _counter = 0
            for line1 in f1.readlines():
                line1_split = line1.strip('\n').split(' ')
                if len(line1_split) == 1:
                    continue
                param = line1_split[0]
                if defense_name != NO_DEFENSE:
                    assert param == _param_list[_counter]
                    _counter += 1
                acc = line1_split[1]
                acc = float(acc) * 100
                _acc_list.append(acc)
    return _rec_rate_list, _acc_list, _param_list


def defense_evaluation(dir, defense_name_list):
    file_name_list = ['attack_task_acc.txt', 'main_task_acc.txt']
    # for "no defense"
    dir0 = dir 
    attack_path = os.path.join(dir0, file_name_list[0])
    main_path = os.path.join(dir0, file_name_list[1])
    _rec_rate, _acc, _ = read_values(attack_path, main_path, NO_DEFENSE)
    assert len(_rec_rate)==1 and len(_acc)==1
    _rec_rate, _acc = _rec_rate[0], _acc[0]
    print(f"no defense :: [attackACC, mainACC] = [{_rec_rate}, {_acc}]")
    # for "defense"
    for defense in defense_name_list:
        dir1 = os.path.join(dir, defense) if defense != NO_DEFENSE else dir
        attack_path = os.path.join(dir1, file_name_list[0])
        main_path = os.path.join(dir1, file_name_list[1])
        _rec_rate_list, _acc_list, _param_list = read_values(attack_path, main_path, defense)
        assert len(_rec_rate_list)==len(_param_list) and len(_acc_list)==len(_param_list)
        evaluation = []
        for i in range(len(_rec_rate_list)):
            evaluation.append(np.log((_rec_rate-_rec_rate_list[i])/_rec_rate)/np.log((_acc-_acc_list[i])/_acc))
        print(defense, ":: parameters", _param_list)
        print(defense, ":: attackACC", _rec_rate_list)
        print(defense, ":: mianACC", _acc_list)
        print(defense, ":: evaluation", evaluation, np.mean(evaluation), np.std(evaluation))


def defense_evaluation_AUC(dir, defense_name_list):
    file_name_list = ['attack_task_acc.txt', 'main_task_acc.txt']
    # for "no defense"
    dir0 = dir 
    attack_path = os.path.join(dir0, file_name_list[0])
    main_path = os.path.join(dir0, file_name_list[1])
    _rec_rate, _acc, _ = read_values(attack_path, main_path, NO_DEFENSE)
    assert len(_rec_rate)==1 and len(_acc)==1
    _rec_rate, _acc = _rec_rate[0], _acc[0]
    print(f"no defense :: [attackACC, mainACC] = [{_rec_rate}, {_acc}]")
    # for "defense"
    for defense in defense_name_list:
        dir1 = os.path.join(dir, defense) if defense != NO_DEFENSE else dir
        attack_path = os.path.join(dir1, file_name_list[0])
        main_path = os.path.join(dir1, file_name_list[1])
        _rec_rate_list, _acc_list, _param_list = read_values(attack_path, main_path, defense)
        assert len(_rec_rate_list)==len(_param_list) and len(_acc_list)==len(_param_list)
        evaluation = 0.0
        if _acc_list[0] <= _acc_list[-1]:
            # print("increasing")
            last_x = 0.0
            last_y = 0.0
            for i in range(len(_acc_list)):
                evaluation += abs(_acc_list[i]-last_x)*(last_y+_rec_rate_list[i])
                # print(_acc_list[i], last_x, last_y, _rec_rate_list[i], _acc_list[i]-last_x, last_y+_rec_rate_list[i], evaluation)
                last_x = _acc_list[i]
                last_y = _rec_rate_list[i]
            evaluation += abs(_acc-last_x)*(last_y+_rec_rate)
            # print(_acc, last_x, last_y, _rec_rate, _acc-last_x, last_y+_rec_rate, evaluation)
        else:
            assert _acc_list[0] > _acc_list[-1]
            # print("decreasing")
            last_x = _acc
            last_y = _rec_rate
            for i in range(len(_acc_list)):
                evaluation += abs(last_x-_acc_list[i])*(last_y+_rec_rate_list[i])
                # print(last_x, _acc_list[i], last_y, _rec_rate_list[i], last_x-_acc_list[i], last_y+_rec_rate_list[i], evaluation)
                last_x = _acc_list[i]
                last_y = _rec_rate_list[i]
            evaluation += abs(last_x-0.0)*(last_y+0.0)
            # print(last_x, 0.0, last_y, 0.0, last_x-0.0, last_y+0.0, evaluation)
        evaluation *= 0.5
        evaluation /= (_rec_rate * _acc)
        # print(defense, ":: parameters", _param_list)
        # print(defense, ":: attackACC", _rec_rate_list)
        # print(defense, ":: mianACC", _acc_list)
        print(defense, ":: evaluation", evaluation)


def draw_defense_on_main_and_dlg_task_using_scatter(dir, x_limit, y_limit, x_major_locator, mark):
    # plt.style.use('ggplot')
    fig, ax = plt.subplots()

    # defense_name_list = ['gaussian', 'laplace', 'grad_spars', 'marvell', 'ppdl', 'laplace_noise', 'gradient_compression', 'discrete_gradients', 'autoencoder', 'autoencoder/discreteGradients', 'autoencoder/random', 'no defense']
    defense_name_list = ['CAE', 'MID', 'marvell', 'no defense']
    # defense_list = ['DP-G', 'DP-L', 'GS', 'Marvell', 'PPDL', 'LN', 'GC', 'DG', 'CAE', 'CAE+DG', 'RCAE', 'w/o defense']
    defense_list = ['CAE', 'MID', 'Marvel', 'w/o defense']
    file_name_list = ['attack_task_acc.txt', 'main_task_acc.txt']
    rec_rate_list = []
    acc_list = []
    param_list = []
    label_x = 'Main task accuracy'
    label_y = 'Label recovery accuracy'

    for defense in defense_name_list:
        # if defense != 'marvell' and defense != 'autoencoder':
        if defense == 'marvell':
            rec_rate_list.append([])
            acc_list.append([])
            param_list.append([])
            continue
        if defense != defense_name_list[-1]:
            dir1 = os.path.join(dir, defense)
        else:
            dir1 = dir
        attack_path = os.path.join(dir1, file_name_list[0])
        main_path = os.path.join(dir1, file_name_list[1])
        _rec_rate_list, _acc_list, _param_list = [], [], []
        with open(attack_path, 'r') as f:
            with open(main_path, 'r') as f1:
                for line in f.readlines():
                    line_split = line.strip('\n').split(' ')
                    if len(line_split) == 1:
                        continue
                    param = line_split[0]
                    rec_rate = line_split[1]
                    rec_rate = float(rec_rate) * 100
                    _param_list.append(param)
                    _rec_rate_list.append(rec_rate)
                print("[debug]: parameter_list =", _param_list)
                _counter = 0
                for line1 in f1.readlines():
                    line1_split = line1.strip('\n').split(' ')
                    if len(line1_split) == 1:
                        continue
                    param = line1_split[0]
                    if defense != defense_name_list[-1]:
                        assert param == _param_list[_counter]
                        _counter += 1
                    acc = line1_split[1]
                    acc = float(acc) * 100
                    _acc_list.append(acc)
        rec_rate_list.append(_rec_rate_list)
        acc_list.append(_acc_list)
        param_list.append(_param_list)
        if defense == defense_name_list[-1]: # without defese result should be unique
            assert len(param_list[-1])==1 and len(rec_rate_list[-1])==1 and len(acc_list[-1])==1
        print(defense, _rec_rate_list,_acc_list,_param_list)

# dataset = 'nuswide'
# dataset = 'cifar100'
dataset = 'mnist'

exp_type = 'multi_no_top_model'
exp_type = 'multi_top_model'
# exp_type = 'binary_no_top_model'
# exp_type = 'binary_top_model'

if __name__ == '__main__':

    main_task_x_limit_dict = {
        'cifar100':{},
        'mnist':{'multi_no_top_model':[80,98],'multi_top_model':[63,98],'binary_no_top_model':[99.9,100],'binary_top_model':[-1,101]},
        'nuswide':{}
    }
    attack_task_y_limit_dict = {
        'cifar100':{},
        'mnist':{'multi_no_top_model':[-1,101],'multi_top_model':[-1,101],'binary_no_top_model':[-1,101],'binary_top_model':[-1,101]},
        'nuswide':{}
    }
    x_major_locator_dict = {
        'cifar100':{'multi_no_top_model':5},
        'mnist':{'multi_no_top_model':3,'multi_top_model':5,'binary_no_top_model':0.02,'binary_top_model':2},
        'nuswide':{'multi_no_top_model':3}
    }

    exp_dir = f'./exp_result_2048/{dataset}/'
    if not ('_no_top_model' in exp_type):
        exp_dir += '_top_model/'
    x_limit = main_task_x_limit_dict[dataset][exp_type]
    y_limit = attack_task_y_limit_dict[dataset][exp_type]
    x_major_locator = x_major_locator_dict[dataset][exp_type]


    # draw_defense_on_main_and_dlg_task_using_scatter(exp_dir, x_limit, y_limit, x_major_locator, True)
    # draw_defense_on_main_and_dlg_task_using_scatter(exp_dir, x_limit, y_limit, x_major_locator, False)

    # defense_evaluation(exp_dir,['CAE', 'MID'])
    defense_evaluation_AUC(exp_dir,['Gaussian', 'Laplace', 'GradientSparsification', 'CAE', 'MID'])

    # fig = plt.figure()
    # axes = plt.axes(projection='3d')
    
    # if exp_type == 'multi_top_model':
    #     x0 = 94.05
    #     y0 = 11.44
    # elif exp_type == 'multi_no_top_model':
    #     x0 = 95.95
    #     y0 = 93.71
    # else:
    #     x0 = 100
    #     y0 = 100
    # X = np.linspace(EPSILON,x0-x0*0.03,1000)
    # Y = np.linspace(EPSILON,y0-y0*0.03,1000)
    # X, Y = np.meshgrid(X,Y) # generate (x,y) paires
    # Z = ((y0-Y)/(x0-X))
    # Z = np.sqrt((y0-Y)/(x0-X))
    # Z = (((y0-Y)/y0)/((x0-X)/x0))
    # Z = np.sqrt(((y0-Y)/y0)/((x0-X)/x0))
    # surf0 = axes.plot_surface(X,Y,Z, cmap=plt.get_cmap("plasma"))
    # plt.colorbar(surf0)

    # axes.scatter3D([x0],[y0],[0])
    # plt.xlim(x0,0)
    # plt.ylim(0,y0)
    # plt.tight_layout()
    # plt.savefig(exp_dir + 'evaluation_matrix.png', dpi = 200)
    # plt.show()
