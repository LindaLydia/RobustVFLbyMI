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

def defense_evaluation_AUC():
    _rec_rate, _acc = 90, 95
    print(f"no defense :: [attackACC, mainACC] = [{_rec_rate}, {_acc}]")
    _rec_rate_list, _acc_list = [60, 50, 40], [90,80,70]
    assert len(_rec_rate_list)==len(_acc_list)
    evaluation = 0.0
    if _acc_list[0] <= _acc_list[-1]:
        print("increasing")
        last_x = 0.0
        last_y = 0.0
        for i in range(len(_acc_list)):
            evaluation += abs(_acc_list[i]-last_x)*(last_y+_rec_rate_list[i])
            print(_acc_list[i], last_x, last_y, _rec_rate_list[i], '||', _acc_list[i]-last_x, last_y+_rec_rate_list[i], evaluation)
            last_x = _acc_list[i]
            last_y = _rec_rate_list[i]
        evaluation += abs(_acc-last_x)*(last_y+_rec_rate)
        print(_acc, last_x, last_y, _rec_rate, '||', _acc-last_x, last_y+_rec_rate, evaluation)
    else:
        assert _acc_list[0] > _acc_list[-1]
        print("decreasing")
        last_x = _acc
        last_y = _rec_rate
        for i in range(len(_acc_list)):
            evaluation += abs(last_x-_acc_list[i])*(last_y+_rec_rate_list[i])
            print(last_x, _acc_list[i], last_y, _rec_rate_list[i], '||', last_x-_acc_list[i], last_y+_rec_rate_list[i], evaluation)
            last_x = _acc_list[i]
            last_y = _rec_rate_list[i]
        evaluation += abs(last_x-0.0)*(last_y+0.0)
        print(last_x, 0.0, last_y, 0.0, '||', last_x-0.0, last_y+0.0, evaluation)
    evaluation *= 0.5
    evaluation /= (_rec_rate * _acc)
    print(":: evaluation", evaluation)

if __name__ == '__main__':

    defense_evaluation_AUC()
