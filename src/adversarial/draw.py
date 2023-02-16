import os
import numpy as np
import matplotlib.pyplot as plt
import json

from pyparsing import line
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

EPOCH_NUM = 50
EPOCH_NUM = 100

linestyle_tuple = [
     ('solid', 'solid'),      # Same as (0, ()) or '-'
     ('densely dotted',        (0, (1, 1))),
     ('densely dashdotted',    (0, (3, 1, 1, 1))),
     ('densely dashed',        (0, (5, 1))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1))),
     ('dashdotted',            (0, (3, 5, 1, 5))),
     ('dashed',                (0, (5, 5))),
     ('dashdotted',            (0, (3, 5, 1, 5))),
     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5)))
]


# log reading function for non-defense experiments
def read_log(log_path):
    with open(log_path, 'r') as f:
        lines = f.readlines()

    target_label = 'none'
    for line in lines:
        if 'target label:' in line:
            target_label = int(line.split('target label:')[-1].strip())
            break

    #20-mnist-mlp2-64-perround-1-10-16-0-0-0.01-1-0.1-20211031-215159
    #new: epoch - dataset - model - batch_size - name - \
    #     backdoor - amplify_rate - amplify_rate_output - dp_type - dp_strength - \
    #     gradient_sparsification - certify - sigma - autoencoder - lba - \
    #     seed - use_project_head - random_output - learning_rate - timestamp
    # parse information
    path = os.path.normpath(log_path)
    path_temps = path.split(os.sep)
    basename = path_temps[-2]
    # print(path_temps, basename)
    temps = basename.split('-')
    dataset = temps[1]
    model = temps[2]
    backdoor = int(temps[5])
    amplify_rate = float(temps[6])
    amplify_rate_output = float(temps[7])
    if backdoor == 0:
        amplify_rate = 0
    info = [dataset, model, backdoor, amplify_rate, basename, amplify_rate_output, target_label]

    # parse results
    train_acc_list = []
    test_acc_list = []
    backdoor_acc_list = []
    backdoor_acc_target_list = []
    train_loss_list = []
    test_loss_list = []
    backdoor_loss_list = []
    backdoor_loss_target_list = []
    for line in lines:
        if 'Epoch' in line:
            # 2021-10-31 21:52:09,743 Epoch 1, Poisoned 928/928, Loss: 2.2992, Accuracy: 4.32, Test Loss: 0.0000, Test Accuracy: 30.18, Backdoor Loss: 0.0000, Backdoor Accuracy: 52.00
            temps = line.split(',')
            train_loss = float(temps[3].strip().split(':')[-1].strip())
            train_acc = float(temps[4].strip().split(':')[-1].strip())
            test_loss = float(temps[5].strip().split(':')[-1].strip())
            test_acc = float(temps[6].strip().split(':')[-1].strip())
            backdoor_loss = float(temps[7].strip().split(':')[-1].strip())
            backdoor_acc = float(temps[8].strip().split(':')[-1].strip())
            # if defense_model == 'autoencoder' and backdoor_acc >= 99.9:
            #     return [],[],[],[],[],[],[],[],[]
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            backdoor_acc_list.append(backdoor_acc)
            train_loss_list.append(train_loss)
            test_loss_list.append(test_loss)
            backdoor_loss_list.append(backdoor_loss)
        if 'Backdoor Loss target down' in line:
            temps = line.split(',')
            backdoor_loss_target = float(temps[4].strip().split(':')[-1].strip())
            backdoor_acc_target = float(temps[5].strip().split(':')[-1].strip())
            backdoor_loss_target_list.append(backdoor_loss_target)
            backdoor_acc_target_list.append(backdoor_acc_target)

    epoch_count = len(train_acc_list)
    append_count = 100 - len(train_acc_list)
    epoch_count -= 1
    if epoch_count >= 49:
        for _ in range(append_count):
            train_acc_list.append(train_acc_list[epoch_count])
            test_acc_list.append(test_acc_list[epoch_count])
            backdoor_acc_list.append(backdoor_acc_list[epoch_count])
            train_loss_list.append(train_loss_list[epoch_count])
            test_loss_list.append(test_loss_list[epoch_count])
            backdoor_loss_list.append(backdoor_loss_list[epoch_count])
    
    epoch_count = len(backdoor_loss_target_list)
    append_count = 100 - len(backdoor_loss_target_list)
    epoch_count -= 1
    if epoch_count >= 49:
        for _ in range(append_count):
            backdoor_loss_target_list.append(backdoor_loss_target_list[epoch_count])
            backdoor_acc_target_list.append(backdoor_acc_target_list[epoch_count])

    return train_acc_list, train_loss_list, test_acc_list, test_loss_list, backdoor_acc_list,  backdoor_loss_list, \
           backdoor_acc_target_list, backdoor_loss_target_list, info


# log reading function for defense experiments
def read_log_defense(log_path):
    with open(log_path, 'r') as f:
        lines = f.readlines()

    # epoch - dataset - model - batch_size - name - \
    # backdoor - amplify_rate - amplify_rate_output - dp_type - dp_strength - \
    # gradient_sparsification - certify - sigma - autoencoder - lba - \
    # seed - use_project_head - random_output - learning_rate - timestamp
    
    # parse information
    path = os.path.normpath(log_path)
    path_temps = path.split(os.sep)
    basename = path_temps[-2]
    temps = basename.split('-')
    dataset = temps[1]
    model = temps[2]
    backdoor = int(temps[5])
    amplify_rate = float(temps[6])
    amplify_rate_output = float(temps[7])
    defense_model = temps[8]
    dp_strength = temps[9]
    sparsification = float(temps[10])
    sigma = float(temps[12])
    autoencoder_coef = float(temps[14])
    if sparsification != 0:
        defense_model = 'gradient_sparsification'
    if sigma != 0:
        defense_model = 'certifyFL'
    if autoencoder_coef != 0:
        defense_model = 'autoencoder'

    if backdoor == 0:
        amplify_rate = 0
    info = [dataset, model, backdoor, amplify_rate, basename, amplify_rate_output, defense_model, dp_strength, sparsification, sigma, autoencoder_coef]

    # parse results
    train_acc_list = []
    test_acc_list = []
    backdoor_acc_list = []
    backdoor_acc_target_list = []
    train_loss_list = []
    test_loss_list = []
    backdoor_loss_list = []
    backdoor_loss_target_list = []
    for line in lines:
        if 'Epoch' in line:
            temps = line.split(',')
            train_loss = float(temps[3].strip().split(':')[-1].strip())
            train_acc = float(temps[4].strip().split(':')[-1].strip())
            test_loss = float(temps[5].strip().split(':')[-1].strip())
            test_acc = float(temps[6].strip().split(':')[-1].strip())
            backdoor_loss = float(temps[7].strip().split(':')[-1].strip())
            backdoor_acc = float(temps[8].strip().split(':')[-1].strip())
            # if defense_model == 'autoencoder' and backdoor_acc >= 85.0:
            #     print("removed")
            #     return [],[],[],[],[],[],[],[],[]
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            backdoor_acc_list.append(backdoor_acc)
            train_loss_list.append(train_loss)
            test_loss_list.append(test_loss)
            backdoor_loss_list.append(backdoor_loss)
        if 'Backdoor Loss target down' in line:
            temps = line.split(',')
            backdoor_loss_target = float(temps[4].strip().split(':')[-1].strip())
            backdoor_acc_target = float(temps[5].strip().split(':')[-1].strip())
            backdoor_loss_target_list.append(backdoor_loss_target)
            backdoor_acc_target_list.append(backdoor_acc_target)

    epoch_count = len(train_acc_list)
    append_count = 100 - len(train_acc_list)
    epoch_count -= 1
    if epoch_count >= 49:
        for _ in range(append_count):
            train_acc_list.append(train_acc_list[epoch_count])
            test_acc_list.append(test_acc_list[epoch_count])
            backdoor_acc_list.append(backdoor_acc_list[epoch_count])
            train_loss_list.append(train_loss_list[epoch_count])
            test_loss_list.append(test_loss_list[epoch_count])
            backdoor_loss_list.append(backdoor_loss_list[epoch_count])
    
    epoch_count = len(backdoor_loss_target_list)
    append_count = 100 - len(backdoor_loss_target_list)
    epoch_count -= 1
    if epoch_count >= 49:
        for _ in range(append_count):
            backdoor_loss_target_list.append(backdoor_loss_target_list[epoch_count])
            backdoor_acc_target_list.append(backdoor_acc_target_list[epoch_count])

    return train_acc_list, train_loss_list, test_acc_list, test_loss_list, backdoor_acc_list,  backdoor_loss_list, \
           backdoor_acc_target_list, backdoor_loss_target_list, info


def plot_per_dataset_and_model(target_dataset, target_model, exp_dir, target_dir, filter_dict={'test_loss': 3, 'backdoor_acc':70}):
    res_to_plot = {}
    for item in os.listdir(exp_dir):
        sub_dir = os.path.join(exp_dir, item)
        if os.path.isdir(sub_dir):
            temps = item.split('-')
            dataset = temps[1]
            model = temps[2]
            backdoor = int(temps[5])
            amplify_rate = float(temps[6])
            amplify_rate_output = float(temps[7])
            if backdoor == 0:
                amplify_rate = 0
            exp_key = '{}-{}-{}-{}'.format(dataset, model, backdoor, amplify_rate)
            if dataset == target_dataset and model == target_model and amplify_rate_output==1:
                # print(os.path.join(sub_dir, 'log.txt'))
                res = read_log(os.path.join(sub_dir, 'log.txt'))
                # remove unstable results
                valid_flag = True
                if len(res[3]) != 100:
                    valid_flag = False
                for item in filter_dict:
                    # print(item)
                    if 'loss' in item:
                        if res[3][-1] > filter_dict[item]:
                            valid_flag = False
                    if 'acc' in item:
                        if res[4][-1] < filter_dict[item]:
                            valid_flag = False
                    if 'target' in item:
                        if res[6][-1] < filter_dict[item]:
                            valid_flag = False
                if not valid_flag:
                    continue
                if exp_key not in res_to_plot:
                    res_to_plot[exp_key] = {'raw': [], 'count': 1, 'backdoor': backdoor, 'amplify_rate': amplify_rate,
                                            'amplify_rate_output': amplify_rate_output}
                    res_to_plot[exp_key]['raw'].append(res)
                    if 'x' not in res_to_plot[exp_key]:
                        res_to_plot[exp_key]['x'] = list(range(len(res[0])))
                else:
                    res_to_plot[exp_key]['raw'].append(res)
                    res_to_plot[exp_key]['count'] = res_to_plot[exp_key]['count'] + 1

    for item in res_to_plot:
        res_to_plot[item]['mean'] = []
        ref = res_to_plot[item]['raw']
        for i in range(8):
            temp_mean = np.zeros(len(ref[0][0]))
            count = 0
            for j in range(len(ref)):
                # print(i,j, len(ref[j][i]))
                temp_mean = temp_mean + ref[j][i]
                count = count + 1
            res_to_plot[item]['mean'].append(np.array(temp_mean/count).tolist())

    res_to_plot = {k: v for k, v in sorted(res_to_plot.items(), key=lambda item: int(float(item[0].split('-')[-1])))}
    # plot
    fig, axs = plt.subplots(2,2)
    fig.set_size_inches(12, 8)
    if len(filter_dict) == 0:
        filter_str = 'all'
    else:
        filter_str = '-'.join(['{}-{}'.format(x, filter_dict[x]) for x in filter_dict])

    # plot main task test accuracy
    for item in res_to_plot:
        ref = res_to_plot[item]
        if len(ref['mean']) > 0:
            label = 'amplify_rate_{}'.format(int(ref['amplify_rate'])) if ref['backdoor'] == 1 else 'normal training'
            axs[0,0].plot(ref['x'], ref['mean'][2], label=label, linewidth=4)
    axs[0,0].set_xlabel('Number of epochs', fontsize=14)
    axs[0,0].set_ylabel('Main task test accuracy', fontsize=14)
    axs[0,0].legend()

    # plot main task test loss
    for item in res_to_plot:
        ref = res_to_plot[item]
        if len(ref['mean']) > 0:
            label = 'amplify_rate_{}'.format(int(ref['amplify_rate'])) if ref['backdoor'] == 1 else 'normal training'
            axs[1,0].plot(ref['x'], np.log10(np.array(ref['mean'][3])+1), label=label, linewidth=4)
    axs[1,0].set_xlabel('Number of epochs', fontsize=14)
    axs[1,0].set_ylabel('Main task test loss [log10]', fontsize=14)
    axs[1,0].legend()

    # plot backdoor task accuracy
    for item in res_to_plot:
        ref = res_to_plot[item]
        if len(ref['mean']) > 0:
            label = 'amplify_rate_{}'.format(int(ref['amplify_rate'])) if ref['backdoor'] == 1 else 'normal training'
            axs[0,1].plot(ref['x'], ref['mean'][4], label=label, linewidth=4)
    axs[0,1].set_xlabel('Number of epochs', fontsize=14)
    axs[0,1].set_ylabel('Backdoor task accuracy', fontsize=14)
    if target_dataset == 'mnist':
        axs[0, 1].legend(loc="center right", bbox_to_anchor=(1.0,0.58))
    elif target_dataset == 'nuswide':
        axs[0, 1].legend(loc="center right", bbox_to_anchor=(1.0,0.43))
    else:
        axs[0, 1].legend()

    # plot backdoor task loss
    for item in res_to_plot:
        ref = res_to_plot[item]
        if len(ref['mean']) > 0:
            label = 'amplify_rate_{}'.format(int(ref['amplify_rate'])) if ref['backdoor'] == 1 else 'normal'
            axs[1,1].plot(ref['x'], np.log10(np.array(ref['mean'][5])+1), label=label, linewidth=4)
    axs[1,1].set_xlabel('Number of epochs', fontsize=14)
    axs[1,1].set_ylabel('Backdoor task loss [log10]')
    axs[1,1].legend()

    plt.tight_layout()
    fig.subplots_adjust(top=0.90)

    plt.savefig('./{}/{}_{}_{}.png'.format(target_dir, target_dataset, target_model, filter_str),dpi=200)
    plt.close()


def plot_per_dataset_and_model_defense(target_dataset, target_model, exp_dir, target_dir, target_method='laplace'):
    res_to_plot = {}
    for item in os.listdir(exp_dir):
        sub_dir = os.path.join(exp_dir, item)
        if os.path.isdir(sub_dir):
            temps = item.split('-')
            dataset = temps[1]
            model = temps[2]
            backdoor = int(temps[5])
            amplify_rate = float(temps[6])
            amplify_rate_output = float(temps[7])

            defense_model = temps[8]
            dp_strength = temps[9]
            sparsification = float(temps[10])

            if sparsification != 0:
                defense_model = 'gradient_sparsification'
            if backdoor == 0:
                amplify_rate = 0

            exp_key = '{}-{}-{}-{}-{}-{}'.format(dataset, model, backdoor, defense_model, dp_strength, sparsification)
            if dataset == target_dataset and model == target_model and amplify_rate_output==1 and defense_model == target_method:
                # print(os.path.join(sub_dir, 'log.txt'))
                res = read_log_defense(os.path.join(sub_dir, 'log.txt'))
                # remove unstable results
                valid_flag = True
                if len(res[3]) != 100:
                    valid_flag = False
                if not valid_flag:
                    continue
                if exp_key not in res_to_plot:
                    res_to_plot[exp_key] = {'raw': [], 'count': 1, 'backdoor': backdoor, 'defense_model': defense_model,
                                            'dp_strength': dp_strength, 'sparsification':sparsification}
                    res_to_plot[exp_key]['raw'].append(res)
                    if 'x' not in res_to_plot[exp_key]:
                        res_to_plot[exp_key]['x'] = list(range(len(res[0])))
                else:
                    res_to_plot[exp_key]['raw'].append(res)
                    res_to_plot[exp_key]['count'] = res_to_plot[exp_key]['count'] + 1

    for item in res_to_plot:
        res_to_plot[item]['mean'] = []
        ref = res_to_plot[item]['raw']
        for i in range(8):
            temp_mean = np.zeros(len(ref[0][0]))
            count = 0
            for j in range(len(ref)):
                # print(i,j, len(ref[j][i]))
                temp_mean = temp_mean + ref[j][i]
                count = count + 1
            res_to_plot[item]['mean'].append(np.array(temp_mean/count).tolist())

    # plot
    fig, ax = plt.subplots()
    fig.suptitle('{}-{}-{}'.format(target_dataset, target_model, target_method), fontsize=14)

    x = []
    y_main = []
    y_backdoor = []

    tag = []
    for i, item in enumerate(res_to_plot):
        ref = res_to_plot[item]
        if len(ref['mean']) > 0:
            x.append(i)
            y_main.append(ref['mean'][2][-1])
            y_backdoor.append(ref['mean'][4][-1])

            if target_method in ['gaussian', 'laplace']:
                temps = item.split('-')
                tag.append(temps[-2])
            else:
                temps = item.split('-')
                tag.append(temps[-1])
    ax.plot(x, y_main, marker='o', label='main task', color='tab:red', linewidth=4, markersize=8)
    ax.plot(x, y_backdoor, marker='o', label='backdoor task', color='tab:blue', linewidth=4, markersize=8)

    if target_method in ['gaussian', 'laplace']:
        ax.set_xlabel('Noise level', fontsize=14)
    else:
        ax.set_xlabel('Drop rate', fontsize=14)

    ax.set_ylabel('Test accuracy', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(tag, fontsize=14)#, rotation=45, ha="right")
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.legend(fontsize=14)
    plt.tight_layout()
    fig.subplots_adjust(top=0.90)
    plt.savefig('./{}/{}_{}_{}_inone.png'.format(target_dir, target_dataset, target_model, target_method), dpi=200)
    plt.close()


def save_defense_data(target_dataset, target_model, exp_dir):
    res_to_plot = {}
    for item in os.listdir(exp_dir):
        sub_dir = os.path.join(exp_dir, item)
        if os.path.isdir(sub_dir):
            temps = item.split('-')

            # epoch - dataset - model - batch_size - name - \
            # backdoor - amplify_rate - amplify_rate_output - dp_type - dp_strength - \
            # gradient_sparsification - certify - sigma - autoencoder - lba - \
            # seed - use_project_head - random_output - learning_rate - timestamp

            epoch = int(temps[0])
            if epoch != EPOCH_NUM:
                continue
            dataset = temps[1]
            model = temps[2]
            backdoor = int(temps[5])
            amplify_rate = float(temps[6])
            amplify_rate_output = float(temps[7])
            defense_model = temps[8]
            dp_strength = temps[9]
            sparsification = float(temps[10])
            sigma = float(temps[12])
            autoencoder_coef = float(temps[14])

            # timestamp_date = int(temps[19])
            # # print(timestamp_date)
            # if timestamp_date != 20211104:
            #     continue

            if sparsification != 0:
                defense_model = 'gradient_sparsification'
            if backdoor == 0:
                amplify_rate = 0
            if sigma != 0:
                defense_model = 'certifyFL'
            if defense_model=='none' and autoencoder_coef != -0.1:
                defense_model = 'autoencoder'
            # else:
            #     autoencoder_coef = 0
            if dataset == target_dataset and model == target_model: #?? and amplify_rate==10

                print(os.path.join(sub_dir, 'log.txt'))
                temp_sub_dir = sub_dir
                while os.path.isdir(temp_sub_dir):
                    found_tb = False
                    for item in os.listdir(temp_sub_dir):
                        print(item)
                        if item == 'tb' or item=='log.txt':
                            found_tb = True
                            break
                        if os.path.isdir(os.path.join(temp_sub_dir, item)):
                            temp_sub_dir = os.path.join(temp_sub_dir, item)
                    if found_tb:
                        break
                sub_dir = temp_sub_dir
                res = read_log_defense(os.path.join(sub_dir, 'log.txt'))

                exp_key = '{}-{}-{}-{}-{}-{}-{}-{}'.format(dataset, model, backdoor, defense_model, dp_strength,
                                                     sparsification, sigma, autoencoder_coef) #[4,5,6,7]
                # print("exp_key is:", exp_key)

                # remove unstable results
                valid_flag = True
                if len(res[3]) != 100:
                    valid_flag = False
                print("valid_flag = ",valid_flag)
                if not valid_flag:
                    continue
                if exp_key not in res_to_plot:
                    res_to_plot[exp_key] = {'raw': [], 'count': 1, 'backdoor': backdoor, 'defense_model': defense_model,
                                            'dp_strength': dp_strength, 'sparsification': sparsification, 'sigma': sigma,
                                            'autoencoder_coef': autoencoder_coef}
                    res_to_plot[exp_key]['raw'].append(res)
                    if 'x' not in res_to_plot[exp_key]:
                        res_to_plot[exp_key]['x'] = list(range(len(res[0])))
                else:
                    res_to_plot[exp_key]['raw'].append(res)
                    res_to_plot[exp_key]['count'] = res_to_plot[exp_key]['count'] + 1

    for item in res_to_plot:
        # print("item is:", item)
        res_to_plot[item]['mean'] = []
        ref = res_to_plot[item]['raw']
        # print(type(ref),len(ref),len(ref[0]))
        for i in range(8):
            # temp_mean = np.zeros(len(ref[0][i]))
            temp_mean = np.zeros(1)
            count = 0
            for j in range(len(ref)):
                # print(i, j, len(ref[j][i]))
                temp_mean = temp_mean + ref[j][i]
                count = count + 1
            res_to_plot[item]['mean'].append(np.array(temp_mean / count).tolist())
        res_to_plot[item]['raw'] = []

    # print(res_to_plot.keys())

    dict_to_save = {}
    for item in res_to_plot:
        # print("item in res_to_plot is" , item)
        dict_to_save[item] = {}
        dict_to_save[item]['backdoor'] = res_to_plot[item]['mean'][4][-1]
        dict_to_save[item]['main'] = res_to_plot[item]['mean'][2][-1]
    temp_l = list(dict_to_save.items())
    temp_l.sort(reverse=True)
    dict_to_save = dict(temp_l)
    with open('{}_{}.json'.format(target_dataset, target_model), 'w') as f:
        json.dump(dict_to_save, f, indent=4)


def plot_per_dataset_and_model_for_paper(target_dataset, target_model, exp_dir, target_dir):
    res_to_plot = {}
    for item in os.listdir(exp_dir):
        sub_dir = os.path.join(exp_dir, item)
        if os.path.isdir(sub_dir):
            temps = item.split('-')
            epoch = int(temps[0])
            if epoch != EPOCH_NUM:
                continue
            dataset = temps[1]
            model = temps[2]
            backdoor = int(temps[5])
            amplify_rate = float(temps[6])
            amplify_rate_output = float(temps[7])
            if backdoor == 0:
                amplify_rate = 0

            # set unique keys for experiments
            exp_key = '{}-{}-{}-{}'.format(dataset, model, backdoor, amplify_rate)

            # filter condition
            # if dataset == target_dataset and model == target_model and amplify_rate_output == 10:
            if dataset == target_dataset and model == target_model and amplify_rate_output == 1:
                # print(os.path.join(sub_dir, 'log.txt'))
                temp_sub_dir = sub_dir
                while os.path.isdir(temp_sub_dir):
                    found_tb = False
                    for item in os.listdir(temp_sub_dir):
                        if item == 'tb' or item=='log.txt':
                            found_tb = True
                            break
                        if os.path.isdir(os.path.join(temp_sub_dir, item)):
                            temp_sub_dir = os.path.join(temp_sub_dir, item)
                    if found_tb:
                        break
                sub_dir = temp_sub_dir
                res = read_log(os.path.join(sub_dir, 'log.txt'))
                # print(type(res[3]),len(res[3]))
                # remove unstable results
                valid_flag = True
                if len(res[3]) != 100:
                    valid_flag = False
                if not valid_flag:
                    continue

                # add results to dict
                if exp_key not in res_to_plot:
                    res_to_plot[exp_key] = {'raw': [], 'count': 1, 'backdoor': backdoor, 'amplify_rate': amplify_rate,
                                            'amplify_rate_output': amplify_rate_output}
                    res_to_plot[exp_key]['raw'].append(res)
                    if 'x' not in res_to_plot[exp_key]:
                        res_to_plot[exp_key]['x'] = list(range(len(res[0])))
                else:
                    res_to_plot[exp_key]['raw'].append(res)
                    res_to_plot[exp_key]['count'] = res_to_plot[exp_key]['count'] + 1

    # compute mean for each experimental key
    for item in res_to_plot:
        res_to_plot[item]['mean'] = []
        ref = res_to_plot[item]['raw']
        print(len(ref))
        for i in range(6):
            # for 6 different kind of values (train_acc_list, train_loss_list, test_acc_list, test_loss_list, backdoor_acc_list,  backdoor_loss_list)
            print(len(ref[0]),len(ref[0][0]))
            temp_mean = np.zeros(len(ref[0][0]))
            count = 0
            for j in range(len(ref)):
                print(i, j, len(ref[j][i]))
                temp_mean = temp_mean + ref[j][i]
                count = count + 1
            res_to_plot[item]['mean'].append(np.array(temp_mean / count).tolist())

    res_to_plot = {k: v for k, v in sorted(res_to_plot.items(), key=lambda item: int(float(item[0].split('-')[-1])))}

    # save final test results
    dict_to_save = {}
    for item in res_to_plot:
        dict_to_save[item] = {}
        dict_to_save[item]['backdoor'] = res_to_plot[item]['mean'][4][-1]
        dict_to_save[item]['main'] = res_to_plot[item]['mean'][2][-1]
    with open('{}_{}_training.json'.format(target_dataset, target_model), 'w') as f:
        json.dump(dict_to_save, f, indent=4)

    # plot main task accuracy
    fig, ax = plt.subplots()
    _i = 0
    for item in res_to_plot:
        ref = res_to_plot[item]
        if len(ref['mean']) > 0:
            label = 'amplify_rate_{}'.format(int(ref['amplify_rate'])) if ref['backdoor'] == 1 else 'normal training'
            ax.plot(ref['x'], ref['mean'][2], label=label, linewidth=4, linestyle=linestyle_tuple[_i][1])
            _i += 1
    ax.set_xlabel('Number of epochs', fontsize=16)
    ax.set_ylabel('Main task accuracy', fontsize=16)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_ylim([0, 100])
    ax.legend(fontsize=14)

    plt.tight_layout()
    plt.savefig('./{}/{}_{}_{}.png'.format(target_dir, target_dataset, target_model, 'main_task'), dpi=200)

    # plot backdoor task accuracy
    fig, ax = plt.subplots()
    _i = 0
    for item in res_to_plot:
        ref = res_to_plot[item]
        if len(ref['mean']) > 0:
            label = 'amplify_rate_{}'.format(int(ref['amplify_rate'])) if ref['backdoor'] == 1 else 'normal training'
            ax.plot(ref['x'], ref['mean'][4], label=label, linewidth=4, linestyle=linestyle_tuple[_i][1])
            _i += 1
    ax.set_xlabel('Number of epochs', fontsize=16)
    ax.set_ylabel('Backdoor task accuracy', fontsize=16)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_ylim([0, 100])
    if target_dataset == 'mnist':
        ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.42), fontsize=14)
    elif target_dataset == 'nuswide':
        ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.36), fontsize=14)
    else:
        ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.25), fontsize=14)

    plt.tight_layout()
    plt.savefig('./{}/{}_{}_{}.png'.format(target_dir, target_dataset, target_model, 'backdoor_task'), dpi=200)
    plt.close()


def plot_scatter_plot_for_paper(input_file, target_dir, marker=False):
    with open(input_file, 'r') as f:
        data = json.load(f)
    x = {}
    y = {}
    label = {}
    for key in data:
        # print("key is",key)
        temps = key.split('-')
        dataset = temps[0]
        defense_method = temps[3]
        defense_param = -100
        if defense_method in ['gaussian', 'laplace']:
            defense_param = temps[4]
            print(defense_param)
            if dataset == 'mnist':
                if defense_param not in ['1.0', '0.1', '0.05', '0.01']:
                # if defense_param not in ['1.0', '0.5', '0.1', '0.01', '0.05', '0.001', '0.005', '0.0001', '0.00001']:
                    continue
            elif dataset == 'cifar20' or dataset == 'cifar10':
                if defense_param not in ['1.0', '0.5', '0.1', '0.01', '0.001']:
                # if defense_param not in ['1.0', '0.5', '0.1', '0.01', '0.05', '0.001', '0.005', '0.0001', '0.00001']:
                    continue
        elif defense_method == 'gradient_sparsification':
            defense_param = temps[5]
            # if ('cifar10_' not in input_file) and defense_param not in ['99.9', '99.5', '99.0']:
            #     continue
            # elif 'cifar10_' in input_file and defense_param not in ['99.9', '99.5', '95.0']:
            #     continue
        elif defense_method == 'certifyFL':
            defense_param = temps[6]
            if defense_param not in ['0.1', '0.01', '0.001', '0.0001']:
                continue
        elif defense_method == 'autoencoder' or defense_method == 'autoencoder+dsgd':
            defense_param = temps[7]
            if 'nuswide' in input_file:
                if defense_param not in ['0.0', '0.1', '0.5', '1.0']:
                    continue
            else:
                if defense_param not in ['0.0', '0.1', '0.5', '1.0']:
                    continue
        elif defense_method == 'dsgd':
            defense_param = temps[9]
        elif defense_method == 'mid':
            defense_param = temps[8]
        elif defense_method == 'rvfr':
            defense_param = temps[10]
        if defense_method not in x:
            x[defense_method] = []
            y[defense_method] = []
            label[defense_method] = []
        x[defense_method].append(data[key]['main'])
        y[defense_method].append(data[key]['backdoor'])
        # y[defense_method].append(100-data[key]['backdoor'])
        label[defense_method].append(defense_param)

    x.pop('none', None)
    y.pop('none', None)
    print(x)
    fig, ax = plt.subplots() #default_figsize=(6.4,4.8)
    ax.set_xscale("log")
    # marker_list = ['o'(DP-G), 'v'(DP-L), '^'(GS), 'D'(DG), '*'(CAE), '1'(DCAE), '4'(MID)]
    # color_list = ['#1f77b4'(DP-G), '#ff7f0e'(DP-L), '#2ca02c', '#7f7f7f', '#d62728', '#bcbd22', '#17becf']
    marker_list = ['o', 'v', '^', 'h', '4']
    color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#e377c2', '#17becf']
    method_name_dict = {'gaussian': 'DP-G', 'laplace': 'DP-L', 'gradient_sparsification': 'GS', 'certifyFL': 'CFL', 'dsgd': 'DG', 'autoencoder': 'CAE', 'autoencoder+dsgd': 'DCAE', 'mid': 'MID', 'rvfr': 'RVFR'}
    for i, key in enumerate(x):
        # ax.scatter(x[key], y[key], label=method_name_dict[key], marker=marker_list[i], color=color_list[i], s=150)
        # ax.plot(x[key], y[key], '--', linewidth=4, color=color_list[i])
        temp = list(map(lambda a: a[0]-a[1], zip(x[key],y[key])))
        ax.scatter(x[key], temp, label=method_name_dict[key], marker=marker_list[i], color=color_list[i], s=150)
        ax.plot(x[key], temp, '--', linewidth=4, color=color_list[i])
        if marker:
            for j, txt in enumerate(label[key]):
                # ax.annotate(txt, (x[key][j], y[key][j]))
                ax.annotate(txt, (x[key][j], temp[j]))

    # add baseline point
    if 'mnist' in input_file:
        # ax.scatter([93.2],[42.6], label='w/o defense', marker='s', color='k', s=150)
        ax.scatter([93.2],[93.2-42.6], label='w/o defense', marker='s', color='k', s=150)
        # ax.scatter([93.2],[100-42.6], label='w/o defense', marker='s', color='k', s=150)
    elif 'nuswide' in input_file:
        # ax.scatter([61.1],[22.8], label='w/o defense', marker='s', color='k', s=150)
        ax.scatter([61.1],[61.1-22.8], label='w/o defense', marker='s', color='k', s=150)
        # ax.scatter([61.1],[100-22.8], label='w/o defense', marker='s', color='k', s=150)
    elif 'cifar10_' in input_file:
        # ax.scatter([75.96],[71.0], label='w/o defense', marker='s', color='k', s=150)
        ax.scatter([75.96],[75.96-71.0], label='w/o defense', marker='s', color='k', s=150)
        # ax.scatter([75.96],[100-71.0], label='w/o defense', marker='s', color='k', s=150)
    elif 'cifar100' in input_file:
        # ax.scatter([42.42],[10.5], label='w/o defense', marker='s', color='k', s=150)
        ax.scatter([42.42],[42.41-10.5], label='w/o defense', marker='s', color='k', s=150)
        # ax.scatter([42.42],[100-10.5], label='w/o defense', marker='s', color='k', s=150)
    else:
        if not 'multi_party' in target_dir:
            # ax.scatter([55.7],[13.6], label='w/o defense', marker='s', color='k', s=150)
            ax.scatter([55.7],[55.7-13.6], label='w/o defense', marker='s', color='k', s=150)
            # ax.scatter([55.7],[100-13.6], label='w/o defense', marker='s', color='k', s=150)
        else:
            # ax.scatter([0.5],[5.0], label='w/o defense', marker='s', color='k', s=150)
            ax.scatter([0.5],[0.5-5.0], label='w/o defense', marker='s', color='k', s=150)
            # ax.scatter([0.5],[100-5.0], label='w/o defense', marker='s', color='k', s=150)

    ax.set_xlabel('Main task accuracy', fontsize=16)
    # ax.set_ylabel('Adversarial sample main task accuracy', fontsize=15)
    ax.set_ylabel('Noisy-sample main task difference', fontsize=15)
    # ax.set_ylabel('Adversarial sample main task error', fontsize=15)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.legend(fontsize=14)

    if 'cifar100' in input_file:
        # axins = inset_axes(ax, loc='center', bbox_to_anchor=[0.6, 0.3, 0.2, 0.2]) #, width=1.2, height=1.2, loc='center',bbox_to_anchor=[left, bottom, width, height], bbox_to_anchor=[0.6, 0.7, 0.2, 0.2]
        # axins = inset_axes((0.6, 0.3, 0.2, 0.2)) #, width=1.2, height=1.2, loc='center',bbox_to_anchor=[left, bottom, width, height], bbox_to_anchor=[0.6, 0.7, 0.2, 0.2]
        axins = inset_axes(ax, width="20%", height="20%",loc='lower left',bbox_to_anchor=(0.45, 0.75, 1, 1),bbox_transform=ax.transAxes)
        for i, key in enumerate(x):
            # axins.scatter(x[key], y[key], label=method_name_dict[key], marker=marker_list[i], color=color_list[i], s=150)
            # axins.plot(x[key], y[key], '--', linewidth=4, color=color_list[i])
            temp = list(map(lambda a: a[0]-a[1], zip(x[key],y[key])))
            axins.scatter(x[key], temp, label=method_name_dict[key], marker=marker_list[i], color=color_list[i], s=150)
            axins.plot(x[key], temp, '--', linewidth=4-0.5, color=color_list[i])
            if marker:
                for j, txt in enumerate(label[key]):
                    # axins.annotate(txt, (x[key][j], y[key][j]))
                    axins.annotate(txt, (x[key][j], temp[j]))
        if 'cifar100' in input_file:
            axins.set_xlim(40.6, 42.8)
            axins.set_ylim(28.7,33.3)
        # axins.scatter([42.42],[10.5], label='w/o defense', marker='s', color='k', s=150)
        axins.scatter([42.42],[42.41-10.5], label='w/o defense', marker='s', color='k', s=150)
        # axins.scatter([42.42],[100-10.5], label='w/o defense', marker='s', color='k', s=150)

        axins.patch.set_alpha(0.4)
        mark_inset(ax, axins, loc1=1, loc2=4, fc="none", ec='k', lw=0.5) # 1(upper right) 2 (upper left) 3(lower left) 4(lower right)



    plt.tight_layout()
    if marker:
        plt.savefig(os.path.join(target_dir, input_file.split('/')[-1].split('.')[0] + '_reference.png'), dpi=200)
    else:
        plt.savefig(os.path.join(target_dir, input_file.split('/')[-1].split('.')[0] + '_nomarker.png'), dpi=200)
    plt.close()


def plot_scatter_plot_for_paper_smaller(input_file, target_dir, marker=False):
    with open(input_file, 'r') as f:
        data = json.load(f)
    x = {}
    y = {}
    label = {}
    for key in data:
        # print("key is",key)
        temps = key.split('-')
        defense_dataset = temps[0]
        defense_method = temps[3]
        defense_param = -100
        if defense_method in ['gaussian', 'laplace']:
            defense_param = temps[4]
            print(defense_param)
            if dataset == 'mnist':
                if defense_param not in ['1.0', '0.1', '0.05', '0.01']:
                # if defense_param not in ['1.0', '0.5', '0.1', '0.01', '0.05', '0.001', '0.005', '0.0001', '0.00001']:
                    continue
            elif dataset == 'cifar20' or dataset == 'cifar10':
                if defense_param not in ['1.0', '0.5', '0.1', '0.01', '0.001']:
                # if defense_param not in ['1.0', '0.5', '0.1', '0.01', '0.05', '0.001', '0.005', '0.0001', '0.00001']:
                    continue
        elif defense_method == 'gradient_sparsification':
            defense_param = temps[5]
            # if defense_param not in ['99.9', '99.5', '99.0']:
            # # if defense_param not in ['99.9', '99.5', '99.0', '95.0']:
            #     continue
        elif defense_method == 'certifyFL':
            defense_param = temps[6]
            if defense_param not in ['0.1', '0.01', '0.001', '0.0001']:
                continue
        elif defense_method == 'autoencoder' or defense_method == 'autoencoder+dsgd':
            defense_param = temps[7]
            if 'nuswide' in input_file:
                if defense_param not in ['0.0', '0.1', '0.5', '1.0']:
                    continue
            else:
                if defense_param not in ['0.0', '0.1', '0.5', '1.0']:
                    continue
        elif defense_method == 'dsgd':
            defense_param = temps[9]
        elif defense_method == 'mid':
            defense_param = temps[8]
        elif defense_method == 'rvfr':
            defense_param = temps[10]
        if defense_method not in x:
            x[defense_method] = []
            y[defense_method] = []
            label[defense_method] = []
        x[defense_method].append(data[key]['main'])
        y[defense_method].append(data[key]['backdoor'])
        label[defense_method].append(defense_param)

    none_main = x['none'][0]
    none_attack = y['none'][0]

    x.pop('none', None)
    y.pop('none', None)
    print(x)
    fig, ax = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False, squeeze=False, figsize=(15,6))
    # ax[0][1] = fig.add_axes([7.5,0,0.5,0.5])
    # marker_list = ['o'(DP-G), 'v'(DP-L), '^'(GS), 'D'(DG), '*'(CAE), '1'(DCAE), '4'(MID)]
    # color_list = ['#1f77b4'(DP-G), '#ff7f0e'(DP-L), '#2ca02c', '#7f7f7f', '#d62728', '#bcbd22', '#17becf']
    # marker_list = ['o', 'v', '^', 'D', '*', '1', '4']
    # color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#7f7f7f', '#d62728', '#bcbd22', '#17becf']
    marker_list = ['o', 'v', '^', 'h', '4']
    color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#e377c2', '#17becf']
    method_name_dict = {'gaussian': 'DP-G', 'laplace': 'DP-L', 'gradient_sparsification': 'GS', 'certifyFL': 'CFL', 'dsgd': 'DG', 'autoencoder': 'CAE', 'autoencoder+dsgd': 'DCAE', 'mid': 'MID', 'rvfr': 'RVFR'}
    print(type(x))
    for i, key in enumerate(x):
        print("key is", key)
        if key == 'none':
            continue
        # ax.scatter(x[key], y[key], label=method_name_dict[key], marker=marker_list[i], color=color_list[i], s=150)
        # ax.plot(x[key], y[key], '--', linewidth=4, color=color_list[i])
        temp = list(map(lambda a: max(a[0]-a[1],0.0), zip(x[key],y[key])))
        # temp = list(map(lambda a: a[0]-a[1], zip(x[key],y[key])))
        ax[0][0].scatter(x[key], temp, label=method_name_dict[key], marker=marker_list[i], color=color_list[i], s=150)
        ax[0][0].plot(x[key], temp, '--', linewidth=4, color=color_list[i])
        if marker:
            for j, txt in enumerate(label[key]):
                # ax[0][0].annotate(txt, (x[key][j], y[key][j]))
                ax[0][0].annotate(txt, (x[key][j], temp[j]))
        # ax[0][1] = fig.add_axes([7.5,0,0.5,0.5])
        if key == 'mid' or key== 'rvfr':
            ax[0][1].scatter(x[key], temp, label=method_name_dict[key], marker=marker_list[i], color=color_list[i], s=150)
            ax[0][1].plot(x[key], temp, '--', linewidth=4, color=color_list[i])
            if marker:
                for j, txt in enumerate(label[key]):
                    # ax[0][1].annotate(txt, (x[key][j], y[key][j]))
                    ax[0][1].annotate(txt, (x[key][j], temp[j]))

    # add baseline point
    # if 'mnist' in input_file:
    #     ax.scatter([90.6],[100.0], label='w/o defense', marker='s', color='k', s=150)
    # elif 'nuswide' in input_file:
    #     ax.scatter([61.8],[54.2], label='w/o defense', marker='s', color='k', s=150)
    # else:
    #     if not 'multi_party' in target_dir:
    #         ax.scatter([52.5],[71.4], label='w/o defense', marker='s', color='k', s=150)
    #     else:
    #         ax.scatter([0.5],[5.0], label='w/o defense', marker='s', color='k', s=150)
    # ax.scatter([none_main],[none_attack], label='w/o defense', marker='s', color='k', s=150)
    # add baseline point
    if 'mnist' in input_file:
        # ax.scatter([93.2],[42.6], label='w/o defense', marker='s', color='k', s=150)
        ax[0][0].scatter([93.2],[93.2-42.6], label='w/o defense', marker='s', color='k', s=150)
        # ax.scatter([93.2],[100-42.6], label='w/o defense', marker='s', color='k', s=150)
    elif 'nuswide' in input_file:
        # ax.scatter([61.1],[22.8], label='w/o defense', marker='s', color='k', s=150)
        ax[0][0].scatter([61.1],[61.1-22.8], label='w/o defense', marker='s', color='k', s=150)
        # ax.scatter([61.1],[100-22.8], label='w/o defense', marker='s', color='k', s=150)
    elif 'cifar10_' in input_file:
        # ax.scatter([75.96],[71.0], label='w/o defense', marker='s', color='k', s=150)
        ax[0][0].scatter([75.96],[75.96-71.0], label='w/o defense', marker='s', color='k', s=150)
        # ax.scatter([75.96],[100-71.0], label='w/o defense', marker='s', color='k', s=150)
    elif 'cifar100' in input_file:
        # ax.scatter([42.42],[10.5], label='w/o defense', marker='s', color='k', s=150)
        ax[0][0].scatter([42.42],[42.41-10.5], label='w/o defense', marker='s', color='k', s=150)
        # ax.scatter([42.42],[100-10.5], label='w/o defense', marker='s', color='k', s=150)
    else:
        if not 'multi_party' in target_dir:
            # ax.scatter([55.7],[13.6], label='w/o defense', marker='s', color='k', s=150)
            ax[0][0].scatter([55.7],[55.7-13.6], label='w/o defense', marker='s', color='k', s=150)
            # ax.scatter([55.7],[100-13.6], label='w/o defense', marker='s', color='k', s=150)
        else:
            # ax.scatter([0.5],[5.0], label='w/o defense', marker='s', color='k', s=150)
            ax[0][0].scatter([0.5],[0.5-5.0], label='w/o defense', marker='s', color='k', s=150)
            # ax.scatter([0.5],[100-5.0], label='w/o defense', marker='s', color='k', s=150)


    ax[0][0].set_xlabel('Main task accuracy', fontsize=16)
    # ax[0][0].set_ylabel('Missing sample main task accuracy', fontsize=16)
    ax[0][0].set_ylabel('Missing sample main task difference', fontsize=16)
    ax[0][0].tick_params(axis='x', labelsize=14)
    ax[0][0].tick_params(axis='y', labelsize=14)
    ax[0][0].legend(fontsize=14)
    ax[0][1].set_xlabel('Main task accuracy', fontsize=16)
    # ax[0][1].set_ylabel('Missing sample main task accuracy', fontsize=16)
    ax[0][1].set_ylabel('Missing sample main task difference', fontsize=16)
    ax[0][1].tick_params(axis='x', labelsize=14)
    ax[0][1].tick_params(axis='y', labelsize=14)
    ax[0][1].legend(fontsize=14)

    plt.tight_layout()
    if marker:
        plt.savefig(os.path.join(target_dir, input_file.split('/')[-1].split('.')[0] + '_small_reference.png'), dpi=200)
    else:
        plt.savefig(os.path.join(target_dir, input_file.split('/')[-1].split('.')[0] + '_small_nomarker.png'), dpi=200)
    plt.close()


def CoAE_distribution(target_dataset, target_model, exp_dir):
    # mnist-mlp2-1-autoencoder-0-0.0-0.0-1.0
    res_to_plot = {}
    for item in os.listdir(exp_dir):
        sub_dir = os.path.join(exp_dir, item)
        if os.path.isdir(sub_dir):
            temps = item.split('-')

            # epoch - dataset - model - batch_size - name - \
            # backdoor - amplify_rate - amplify_rate_output - dp_type - dp_strength - \
            # gradient_sparsification - certify - sigma - autoencoder - lba - \
            # seed - use_project_head - random_output - learning_rate - timestamp

            epoch = int(temps[0])
            if epoch != EPOCH_NUM:
                continue
            dataset = temps[1]
            model = temps[2]
            backdoor = int(temps[5])
            amplify_rate = float(temps[6])
            amplify_rate_output = float(temps[7])
            defense_model = temps[8]
            dp_strength = temps[9]
            sparsification = float(temps[10])
            sigma = float(temps[12])
            autoencoder_coef = float(temps[14])
            
            # timestamp_date = int(temps[19])
            # # print(timestamp_date)
            # if timestamp_date != 20211104:
            #     continue

            if sparsification != 0:
                defense_model = 'gradient_sparsification'
            if backdoor == 0:
                amplify_rate = 0
            if sigma != 0:
                defense_model = 'certifyFL'
            if defense_model=='none' and autoencoder_coef != -0.1:
                defense_model = 'autoencoder'
            # else:
            #     autoencoder_coef = 0
            if dataset == target_dataset and model == target_model: #?? and amplify_rate==10

                print(os.path.join(sub_dir, 'log.txt'))
                temp_sub_dir = sub_dir
                while os.path.isdir(temp_sub_dir):
                    found_tb = False
                    for item in os.listdir(temp_sub_dir):
                        if item == 'tb' or item=='log.txt':
                            found_tb = True
                            break
                        if os.path.isdir(os.path.join(temp_sub_dir, item)):
                            temp_sub_dir = os.path.join(temp_sub_dir, item)
                    if found_tb:
                        break
                sub_dir = temp_sub_dir
                res = read_log_defense(os.path.join(sub_dir, 'log.txt'))

                exp_key = '{}-{}-{}-{}-{}-{}-{}-{}'.format(dataset, model, backdoor, defense_model, dp_strength,
                                                     sparsification, sigma, autoencoder_coef) #[4,5,6,7]

                # remove unstable results
                valid_flag = True
                if len(res[3]) != 100:
                    valid_flag = False
                if not valid_flag:
                    continue
                if exp_key not in res_to_plot:
                    res_to_plot[exp_key] = {'raw': [], 'count': 1, 'backdoor': backdoor, 'defense_model': defense_model,
                                            'dp_strength': dp_strength, 'sparsification': sparsification, 'sigma': sigma,
                                            'autoencoder_coef': autoencoder_coef}
                    res_to_plot[exp_key]['raw'].append(res)
                    if 'x' not in res_to_plot[exp_key]:
                        res_to_plot[exp_key]['x'] = list(range(len(res[0])))
                else:
                    res_to_plot[exp_key]['raw'].append(res)
                    res_to_plot[exp_key]['count'] = res_to_plot[exp_key]['count'] + 1

    for item in res_to_plot:
        if item == "mnist-mlp2-1-autoencoder-0-0.0-0.0-1.0":
            ref = res_to_plot[item]['raw']
            # print(type(ref),len(ref),len(ref[0]))
            backdoor_acc_list = []
            main_acc_list = []
            for j in range(len(ref)):
                backdoor_acc_list.append(np.asarray(ref[j][4]).mean())
                main_acc_list.append(np.asarray(ref[j][2]).mean())
            backdoor_acc_list = np.asarray(backdoor_acc_list)
            main_acc_list = np.asarray(main_acc_list)
            # print(backdoor_acc_list.shape, main_acc_list.shape)
            plt.hist(np.asarray(backdoor_acc_list),bins=20)
            plt.xlabel='backdoor accuracy'
            plt.show()
            plt.savefig(os.path.join(target_dir, 'distribution.png'), dpi=200)
            plt.close()
            # break

if __name__ == '__main__':

    # dataset = 'nuswide'
    # model = 'mlp2'
    # exp_dir = './experiment_defense5'
    # target_dir = 'images5'

    # for dataset, model in zip(['mnist', 'nuswide', 'cifar20'], ['mlp2', 'mlp2', 'resnet18']):
    #     plot_per_dataset_and_model_for_paper(dataset, model, exp_dir, target_dir)
    #     plot_per_dataset_and_model_for_paper(dataset, model, exp_dir, target_dir)
    #     plot_per_dataset_and_model_for_paper(dataset, model, exp_dir, target_dir)

    # for dataset, model in zip(['mnist', 'nuswide', 'cifar20'], ['mlp2', 'mlp2', 'resnet18']):
    #     save_defense_data(dataset, model, exp_dir)
    #     plot_scatter_plot_for_paper('{}_{}.json'.format(dataset, model), target_dir, True)
    #     plot_scatter_plot_for_paper('{}_{}.json'.format(dataset, model), target_dir, False)


    # ########################### code for vfl defending paper (IEEE journal) ###########################
    # dataset = 'mnist'
    # model = 'mlp2'
    # # exp_dir = './experiment_plot_model_DSGD_50_experiment_plot_model_new_negativeCoAE_50_defense'
    # # exp_dir = './experiment_plot_model_DSGD_50_defense'
    # # exp_dir = './experiment_plot_model_new_negativeCoAE_50_defense'
    # exp_dir = './Backdoor_exp_result/experiment_maintask'
    # # exp_dir = './experiment_plot_50_CoAE_defense'
    # # exp_dir = './experiment_defense'
    # # exp_dir = './experiment_plot_model_right_50_defense'
    # # exp_dir = './experiment_plot_nuswide_50_defense'
    # target_dir = 'Backdoor_exp_result'

    # for dataset, model in zip(['mnist', 'nuswide', 'cifar20'], ['mlp2', 'mlp2', 'resnet18']):
    # # for dataset, model in zip(['mnist', 'nuswide', 'cifar100'], ['mlp2', 'mlp2', 'resnet18']):
    #     print("[dataset, model] is [{}, {}]".format(dataset, model))
    #     plot_per_dataset_and_model_for_paper(dataset, model, exp_dir, target_dir)
    #     # plot_per_dataset_and_model_for_paper(dataset, model, exp_dir, target_dir)
    #     # plot_per_dataset_and_model_for_paper(dataset, model, exp_dir, target_dir)

    # for dataset, model in zip(['mnist', 'nuswide', 'cifar20'], ['mlp2', 'mlp2', 'resnet18']):
    # # for dataset, model in zip(['mnist', 'nuswide', 'cifar100'], ['mlp2', 'mlp2', 'resnet18']):
    #     save_defense_data(dataset, model, exp_dir)
    #     plot_scatter_plot_for_paper('{}_{}.json'.format(dataset, model), target_dir, True)
    #     plot_scatter_plot_for_paper('{}_{}.json'.format(dataset, model), target_dir, False)
    
    # for dataset, model in zip(['mnist'], ['mlp2']):
    #     CoAE_distribution(dataset, model, exp_dir)
    # ########################### code for vfl defending paper (IEEE journal) ###########################


    target_dir = 'images'
    # for dataset, model in zip(['mnist', 'nuswide', 'cifar10', 'cifar20', 'cifar100'], ['mlp2', 'mlp2', 'resnet18', 'resnet18', 'resnet18']):
    for dataset, model in zip(['mnist', 'cifar100'], ['mlp2', 'resnet18']):
    # for dataset, model in zip(['mnist', 'nuswide', 'cifar20'], ['mlp2', 'mlp2', 'resnet18']):
    # # for dataset, model in zip(['mnist', 'nuswide', 'cifar100'], ['mlp2', 'mlp2', 'resnet18']):
        # plot_scatter_plot_for_paper_smaller('images/data_use/{}_{}.json'.format(dataset, model), target_dir, True)
        # plot_scatter_plot_for_paper_smaller('images/data_use/{}_{}.json'.format(dataset, model), target_dir, False)
        plot_scatter_plot_for_paper('images/data_use/{}_{}.json'.format(dataset, model), target_dir, True)
        plot_scatter_plot_for_paper('images/data_use/{}_{}.json'.format(dataset, model), target_dir, False)

    # for dataset, model in zip(['cifar20'], ['resnet18']):
    #     plot_scatter_plot_for_paper('../reviewer paper/plotting/{}_{}.json'.format(dataset, model), target_dir+'/multi_party', True)
    #     plot_scatter_plot_for_paper('../reviewer paper/plotting/{}_{}.json'.format(dataset, model), target_dir+'/multi_party', False)