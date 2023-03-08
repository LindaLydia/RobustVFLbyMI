import math
import os
from matplotlib import projections
from matplotlib.legend import Legend
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib.ticker as mtick
import numpy as np

from utils import draw_line_chart, remove_exponent

# NO_DEFENSE = 'no defense'
# EPSILON = 1e-3

# def defense_evaluation_AUC():
#     _rec_rate, _acc = 90, 95
#     print(f"no defense :: [attackACC, mainACC] = [{_rec_rate}, {_acc}]")
#     _rec_rate_list, _acc_list = [60, 50, 40], [90,80,70]
#     assert len(_rec_rate_list)==len(_acc_list)
#     evaluation = 0.0
#     if _acc_list[0] <= _acc_list[-1]:
#         print("increasing")
#         last_x = 0.0
#         last_y = 0.0
#         for i in range(len(_acc_list)):
#             evaluation += abs(_acc_list[i]-last_x)*(last_y+_rec_rate_list[i])
#             print(_acc_list[i], last_x, last_y, _rec_rate_list[i], '||', _acc_list[i]-last_x, last_y+_rec_rate_list[i], evaluation)
#             last_x = _acc_list[i]
#             last_y = _rec_rate_list[i]
#         evaluation += abs(_acc-last_x)*(last_y+_rec_rate)
#         print(_acc, last_x, last_y, _rec_rate, '||', _acc-last_x, last_y+_rec_rate, evaluation)
#     else:
#         assert _acc_list[0] > _acc_list[-1]
#         print("decreasing")
#         last_x = _acc
#         last_y = _rec_rate
#         for i in range(len(_acc_list)):
#             evaluation += abs(last_x-_acc_list[i])*(last_y+_rec_rate_list[i])
#             print(last_x, _acc_list[i], last_y, _rec_rate_list[i], '||', last_x-_acc_list[i], last_y+_rec_rate_list[i], evaluation)
#             last_x = _acc_list[i]
#             last_y = _rec_rate_list[i]
#         evaluation += abs(last_x-0.0)*(last_y+0.0)
#         print(last_x, 0.0, last_y, 0.0, '||', last_x-0.0, last_y+0.0, evaluation)
#     evaluation *= 0.5
#     evaluation /= (_rec_rate * _acc)
#     print(":: evaluation", evaluation)

def feature_reconstructed_plot(path):
    dummy_data = [np.load(path+'b_generate_dummy.npy')]
    print(dummy_data[0].shape)

    for i in range(10):
        fig, ax=plt.subplots(1,1)
        ############# CIFAR10 #############
        dummy = dummy_data[0][i]
        dummy = dummy.reshape(3,16,32)
        ax.imshow(np.transpose(dummy, (1,2,0)))
        print(np.transpose(dummy, (1,2,0)).shape)

        # ############# MNIST #############
        # plt.gray()
        # dummy = dummy_data[0][i]
        # ax.imshow(dummy.reshape(14,28))

        ax.axis('off')
        fig.patch.set_alpha(1)
        plt.tight_layout()
        plt.savefig(f'dravl_color_plot_{i}.png', dpi=300)
        # plt.savefig(f'dravl_gray_plot_{i}.png', dpi=300)

if __name__ == '__main__':
    feature_reconstructed_plot('./exp_result_2048/cifar10/recover_image/')
    # feature_reconstructed_plot('./exp_result_2048/mnist/recover_image/')
    # defense_evaluation_AUC()
