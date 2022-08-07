import math
import os
from matplotlib.legend import Legend

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib.ticker as mtick
import numpy as np

from utils import draw_line_chart, remove_exponent


def draw_recovery_rate(path, dataset):
    name = []
    x = []
    y = []
    title = f'exp_on_{dataset}'
    label_x = 'Number of classes'
    label_y = 'Label recovery accuracy'

    with open(path, 'r') as f:
        _name, _x, _y = [], [], []
        for line in f.readlines():
            if '||' in line:
                bs, _, num_class, rate = line.strip('\n').split(',')[-1].split('|')
            else:
                bs, num_class, rate = line.strip('\n').split(',')[-1].split('|')
            num_class = float(num_class)
            rate = float(rate)
            if len(_name) == 0 or bs != _name[-1]:
                name.append('batchsize=' + bs)
                if len(_x) != 0:
                    x.append(_x)
                    y.append(_y)
                    _x, _y = [], []
            _name.append(bs)
            _x.append(num_class)
            _y.append(rate)
        if len(_x) != 0:
            x.append(_x)
            y.append(_y)
    # x_scale = (max(x[0]) - min(x[0])) // 7
    if dataset == 'cifar100':
        x_scale = 5
    elif dataset == 'mnist':
        x_scale = 2
    else:
        x_scale = 10
    y_scale = 0.1
    draw_line_chart(title=title, note_list=name, x=x, y=y, x_scale=x_scale, y_scale=y_scale, label_x=label_x,
                    label_y=label_y, path=path)

def draw_rec_rate_change(path, dataset, num_class, max_epoch):
    name = []
    x = []
    y = []
    title = f'exp_on_{dataset}'
    label_x = 'epochs'
    label_y = 'recovery_rate'

    with open(path, 'r') as f:
        _name = []
        for line in f.readlines():
            bs, epochs, rate, abc = line.strip('\n').split(' ')
            epochs = int(epochs)
            rate = float(rate)
            if epochs > 20000:
                continue
            if epochs == 0:
                name.append(f'batchsize={bs},num_class={num_class}')
                _x, _y = [], []
            _x.append(epochs)
            _y.append(rate)
            if epochs == max_epoch:
                x.append(_x)
                y.append(_y)
    draw_line_chart(title=title, note_list=name, x=x, y=y, x_scale=max_epoch // 5,
                    y_scale=0.1, label_x=label_x, label_y=label_y, path=path)

def draw_rec_rate_vs_numclass_div_batchsize(path, dataset):
    name = []
    x = []
    y = []
    title = f'exp_on_{dataset}'
    label_x = 'class_num/batchsize'
    label_y = 'recovery_rate'
    with open(path, 'r') as f:
        _name, _x, _y = [], [], []
        for line in f.readlines():
            bs, num_class, rate = line.strip('\n').split(',')[-1].split('|')
            num_class = float(num_class)
            rate = float(rate)
            if len(_name) == 0 or bs != _name[-1]:
                name.append('batchsize=' + bs)
                if len(_x) != 0:
                    x.append(_x)
                    y.append(_y)
                    _x, _y = [], []
            _name.append(bs)
            _x.append(num_class / float(bs))
            _y.append(rate)
        if len(_x) != 0:
            x.append(_x)
            y.append(_y)
    x_scale = max(x[0]) / 10
    y_scale = 0.1
    print(title)
    draw_line_chart(title=title, note_list=name, x=x, y=y, x_scale=x_scale, y_scale=y_scale, label_x=label_x,
                    label_y=label_y, path=path)

def draw_label_leakage_defense(path, dataset):
    name = []
    x = []
    y = []
    title = f'exp on {dataset}'
    label_x = 'class_num'
    label_y = 'recovery_rate'
    with open(path, 'r') as f:
        _name, _x, _y = [], [], []
        for line in f.readlines():
            bs, defense_param, num_class, rate = line.strip('\n').split(',')[-1].split('|')
            defense_type = line.strip('\n').split(',')[0].split('|')[1]
            num_class = float(num_class)
            rate = float(rate)
            new_name = f'{defense_type}={remove_exponent(defense_param)}' if defense_type != 'no_defense' else defense_type
            if len(_name) == 0 or new_name != _name[-1]:
                name.append(new_name)
                if len(_x) != 0:
                    x.append(_x)
                    y.append(_y)
                    _x, _y = [], []
            _name.append(new_name)
            _x.append(num_class)
            _y.append(rate)
        if len(_x) != 0:
            x.append(_x)
            y.append(_y)
    if dataset == 'mnist' or dataset == 'cifar10':
        x_scale = 1
    else:
        x_scale = 10
    y_scale = 0.1
    title += f'(batchsize={bs})'
    draw_line_chart(title=title, note_list=name, x=x, y=y, x_scale=x_scale, y_scale=y_scale, label_x=label_x,
                    label_y=label_y, path=path)

def draw_defense_on_main_and_dlg_task(dir, defense):
    file_name_list = ['attack_task_acc.txt', 'main_task_acc.txt']
    name = ['batch label inference attack task', 'main task']
    x = []
    y = []
    title = f'main task and attack task with {defense}'
    label_x = 'dp strength' if defense in ['gaussian', 'laplace'] else 'gradient sparsification'
    label_y = 'recovery_rate/accuracy'
    dir = os.path.join(dir, defense)

    for file_name in file_name_list:
        path = os.path.join(dir, file_name)
        _x, _y = [], []
        with open(path, 'r') as f:
            for line in f.readlines():
                strength, rate = line.strip('\n').split(' ')
                _x.append(float(strength))
                _y.append(float(rate))
            x.append(_x)
            y.append(_y)
    if defense == 'grad_spars':
        x_scale = 1
    else:
        x_scale = 0.0002
    y_scale = 0.1
    res_path = os.path.join(dir, f'main_task_and_attack_task_with_{defense}.png')
    print(x)
    print(y)
    draw_line_chart(title=title, note_list=name, x=x, y=y, x_scale=x_scale, y_scale=y_scale, label_x=label_x,
                    label_y=label_y, path=res_path)

def draw_defense_on_main_and_dlg_task_using_scatter(dir, x_limit, y_limit, x_major_locator, mark):
    # plt.style.use('ggplot')
    fig, ax = plt.subplots()

    # defense_name_list = ['gaussian', 'laplace', 'grad_spars', 'marvell', 'ppdl', 'laplace_noise', 'gradient_compression', 'discrete_gradients', 'autoencoder', 'autoencoder/discreteGradients', 'autoencoder/random', 'no defense']
    defense_name_list = ['Gaussian', 'Laplace', 'GradientSparsification', 'CAE', 'DCAE', 'MARVELL', 'MID', 'no defense']
    # defense_list = ['DP-G', 'DP-L', 'GS', 'Marvell', 'PPDL', 'LN', 'GC', 'DG', 'CAE', 'CAE+DG', 'RCAE', 'w/o defense']
    defense_list = ['DP-G', 'DP-L', 'GS', 'CAE', 'DCAE', 'MARVELL', 'MID', 'w/o defense']
    file_name_list = ['attack_task_acc.txt', 'main_task_acc.txt']
    rec_rate_list = []
    acc_list = []
    param_list = []
    label_x = 'Main task accuracy'
    label_y = 'Label recovery accuracy'

    for defense in defense_name_list:
        # # if defense != 'MARVELL' and defense != 'CAE' and defense != 'MID' and defense != 'no defense':
        # if defense == 'MARVELL':
        #     rec_rate_list.append([])
        #     acc_list.append([])
        #     param_list.append([])
        #     continue
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
                        print(param, _param_list[_counter])
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



    # print(acc_list)
    # print(rec_rate_list)
    # print(param_list)

    # fontsize
    # tick
    # legend
    # labelsize
    # linewidth
    # linemarker_size


    # defense ['DP-G', 'DP-L', 'GS', 'Marvell',|| 'PPDL', 'LN(DP-L)', 'GC(GS)', 'DG', 'CAE', 'CAE+DG', 'RCAE', 'w/o defense']
    # marker_list = ['o', 'v', '^', 'x',|| 'h'(PPDL), 'D'(DG), '+'(CAE+DG), '*'(CAE), 's'(w/o defense), '1'(RCAE), '2', '3', '4'(MID)]
    # marker_list = ['o', 'v', '^', 'x', '*', '+', 's', '1', '2', '3', '4']
    marker_list = ['o', 'v', '^', '*', '+', 'x', '4', 's']
    # color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'(CAE),|| '#9467bd'(Marvell), '#8c564b'(RCAE), '#e377c2'(PPDL), '#7f7f7f'(DG), '#bcbd22'(CAE+DG), '#17becf'(MID)] # the same as the default colors
    # color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#d62728', '#bcbd22', '#8c564b', '#e377c2', '#7f7f7f', '#17becf'] # the same as the default colors
    color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#bcbd22', '#9467bd', '#17becf'] # the same as the default colors
    # offset = [0, -0.08, 0, 0, 0, 0, 0, 0, 0] #cifar10--2
    # offset = [0, -3, 0, 0, 0, 0, 0, 0, 0]
    offset = [-0.3, -0.3, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(len(defense_list)):
        # print(param_list[i])
        if i == len(defense_list)-1:
            ax.scatter(acc_list[i], rec_rate_list[i], label=defense_list[i], marker=marker_list[i], s=60, color='black')
        elif len(acc_list[i])>0:
            ax.scatter(acc_list[i], rec_rate_list[i], label=defense_list[i], marker=marker_list[i], s=60, color=color_list[i])
            if mark:
                for j, txt in enumerate(param_list[i]):
                    print(i,j,txt)
                    if dataset == 'cifar100': # and float(txt)>0.4
                        # # ax.annotate(txt, (acc_list[i][j] + offset[j], rec_rate_list[i][j] + offset[j]), fontsize=9)
                        # if j == len(param_list[i])-2 and (i==4 or i==5):
                        #     ax.annotate(txt, (acc_list[i][j]-0.3, rec_rate_list[i][j]), fontsize=9)
                        # else:
                        #     ax.annotate(txt, (acc_list[i][j], rec_rate_list[i][j]), fontsize=9)
                        ax.annotate(txt, (acc_list[i][j], rec_rate_list[i][j]), fontsize=9)
                    elif dataset == 'mnist':
                        # # ax.annotate(txt, (acc_list[i][j] + offset[i], rec_rate_list[i][j] + offset[i]), fontsize=9)
                        # # if i == 0:
                        # #     ax.annotate(txt, (acc_list[i][j], rec_rate_list[i][j] - 5), fontsize=9)
                        # # elif i == 1:
                        # #     ax.annotate(txt, (acc_list[i][j], rec_rate_list[i][j] + 5), fontsize=9)
                        # # elif i == 3:
                        # #     ax.annotate(txt, (acc_list[i][j], rec_rate_list[i][j]+(3-j*4)), fontsize=9)
                        # # elif i == 4:
                        # #     ax.annotate(txt, (acc_list[i][j], rec_rate_list[i][j]+(2-j*3), fontsize=9)
                        # if i == 0:
                        #     ax.annotate(txt, (acc_list[i][j]-0.5, rec_rate_list[i][j]), fontsize=9)
                        # elif i < 4:
                        #     ax.annotate(txt, (acc_list[i][j], rec_rate_list[i][j]+(-2*(int)(j%2==0))), fontsize=9)
                        # else:
                        #     ax.annotate(txt, (acc_list[i][j], rec_rate_list[i][j]), fontsize=9)
                        ax.annotate(txt, (acc_list[i][j], rec_rate_list[i][j]), fontsize=9)
                    elif dataset == 'nuswide':
                        # if i == 4 and j == 0:
                        #     ax.annotate(txt, (acc_list[i][j] + offset[j], rec_rate_list[i][j]), fontsize=9)
                        # elif i == 5:
                        #     ax.annotate(txt, (acc_list[i][j] , rec_rate_list[i][j] + offset[j]), fontsize=9)
                        # # if i == 5:
                        # #     ax.annotate(txt, (acc_list[i][j] , rec_rate_list[i][j] + offset[j]), fontsize=9)
                        # else:
                        #     ax.annotate(txt, (acc_list[i][j], rec_rate_list[i][j]), fontsize=9)
                        ax.annotate(txt, (acc_list[i][j], rec_rate_list[i][j]), fontsize=9)
        if len(acc_list[i]) > 1:
            ax.plot(acc_list[i], rec_rate_list[i], '--', linewidth=2, color=color_list[i])
            # ax.plot(rec_rate_list[i], acc_list[i], '--',  mec='r', mfc='w', label=defense_list[i])

    ax.set_xlabel(label_x, fontsize=16)
    ax.set_ylabel(label_y, fontsize=16)
    # ax.set_xlabel(label_x, fontsize=16, fontdict={'family' : 'SimSun', 'weight':800})
    # ax.set_ylabel(label_y, fontsize=16, fontdict={'family' : 'SimSun', 'weight':800})
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.legend(fontsize=12)  
    ax.set_xlim(x_limit)
    ax.set_ylim(y_limit)
    x_major_locator = mtick.MultipleLocator(x_major_locator)
    y_major_locator = mtick.MultipleLocator(5)
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    
    fig.patch.set_facecolor('white')
    # fig.patch.set_alpha(0)
    
    plt.tight_layout()
    if mark:
        plt.savefig(dir + 'defense_on_attack_and_main_task.png', dpi = 200)
    else:
        plt.savefig(dir + 'defense_on_attack_and_main_task_nomarker.png', dpi = 200)
    plt.show()


dataset = 'nuswide'
dataset = 'cifar100'
dataset = 'mnist'

# exp_type = 'multi_no_top_model'
# exp_type = 'multi_top_model'
exp_type = 'binary_no_top_model'
# exp_type = 'binary_top_model'

if __name__ == '__main__':

    # if dataset == 'cifar100':
    #     max_num_classes = 2 # 100
    # elif dataset == 'mnist':
    #     max_num_classes = 2 # 10
    # else:
    #     max_num_classes = 2 # 81
    # label_leakage_cifar20_path = 'exp_result/cifar100/dataset=cifar100,model=resnet18,lr=0.05,num_exp=10,epochs=200,early_stop=False.txt'
    # label_leakage_mnist_path = 'exp_result/mnist/dataset=mnist,model=MLP2,lr=0.05,num_exp=30,epochs=10000,early_stop=False.txt'
    # label_leakage_nuswide_path = 'exp_result/nuswide/dataset=nuswide,model=MLP2,lr=0.05,num_exp=20,epochs=20000,early_stop=False.txt'

    # draw_recovery_rate(label_leakage_nuswide_path, dataset)
    # draw_rec_rate_change(f'exp_result/{dataset}/exp_on_{dataset}_rec_rate_change.txt', dataset, 81, 20000)
    # draw_rec_rate_vs_numclass_div_batchsize(f'exp_result/nuswide/rec_rate_vs_numclass_div_batchsize.txt', nuswide)
    # draw_label_leakage_defense('exp_result/nuswide/dataset=nuswide,defense=gradient_sparsification,model=MLP2,num_exp=10,epochs=5000.txt', dataset)
    # draw_defense_on_main_and_dlg_task('exp_result/cifar100', 'laplace')

    main_task_x_limit_dict = {
        'cifar100':{'multi_no_top_model':[34,60],'multi_top_model':[-1,101],'binary_no_top_model':[85,95],'binary_top_model':[-1,101]},
        'mnist':{'multi_no_top_model':[82,98],'multi_top_model':[63,98],'binary_no_top_model':[99.84,100],'binary_top_model':[99.8,100]},
        'nuswide':{'multi_no_top_model':[82,90],'multi_top_model':[63,98],'binary_no_top_model':[77,90],'binary_top_model':[-1,101]}
    }
    attack_task_y_limit_dict = {
        'cifar100':{'multi_no_top_model':[-1,101],'multi_top_model':[-1,101],'binary_no_top_model':[-1,101],'binary_top_model':[-1,101]},
        'mnist':{'multi_no_top_model':[-1,101],'multi_top_model':[-1,101],'binary_no_top_model':[30,70],'binary_top_model':[30,70]},
        'nuswide':{'multi_no_top_model':[-1,101],'multi_top_model':[-1,101],'binary_no_top_model':[-1,101],'binary_top_model':[-1,101]}
    }
    x_major_locator_dict = {
        'cifar100':{'multi_no_top_model':2,'binary_no_top_model':2,'binary_top_model':2},
        'mnist':{'multi_no_top_model':3,'multi_top_model':5,'binary_no_top_model':0.02,'binary_top_model':2},
        'nuswide':{'multi_no_top_model':2,'binary_no_top_model':2,'binary_top_model':2}
    }

    exp_dir = f'./exp_result/{dataset}/'
    exp_dir = f'./exp_result_2048/{dataset}/'
    exp_dir = f'./exp_result_binary/{dataset}/'
    exp_dir = f'./exp_result_direction_scoring/{dataset}/'
    exp_dir = f'./exp_result_norm_scoring/{dataset}/'
    if not ('_no_top_model' in exp_type):
        exp_dir += '_top_model/'
    x_limit = main_task_x_limit_dict[dataset][exp_type]
    y_limit = attack_task_y_limit_dict[dataset][exp_type]
    x_major_locator = x_major_locator_dict[dataset][exp_type]
    draw_defense_on_main_and_dlg_task_using_scatter(exp_dir, x_limit, y_limit, x_major_locator, True)
    draw_defense_on_main_and_dlg_task_using_scatter(exp_dir, x_limit, y_limit, x_major_locator, False)
