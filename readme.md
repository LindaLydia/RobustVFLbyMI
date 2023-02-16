# Mutual Information Regularization for Vertical Federated Learning

This repository contains codes for paper [Mutual Information Regularization for Vertical Federated Learning](https://arxiv.org/abs/2301.01142). 

For the multi-classification task main task evalution, please use `python vfl_main_task_no_defense.py` while for Batch-level Label Inference Attack, please use `python vfl_dlg_no_defense.py`. 

For binary classification please use `python marvell_main.py --exp_type main_task` or  `python marvell_main.py --exp_type attack` for main task evaluation and direction scoring attack evaluation separately.

Config parameters can be specified like `python vfl_dlg_no_defense.py --dataset cifar10`.