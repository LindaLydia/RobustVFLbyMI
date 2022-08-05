# # No Active Party Top Model
python marvell_main.py
# python marvell_main.py --apply_marvell True
# python marvell_main.py --apply_mid True
# python marvell_main.py --apply_encoder True
# python marvell_main.py --apply_laplace True
# python marvell_main.py --apply_gaussian True
# python marvell_main.py --apply_grad_spar True
# python marvell_main.py --apply_discrete_gradients True --discrete_gradients_bins 6
# python marvell_main.py --apply_discrete_gradients True --discrete_gradients_bins 12
# python marvell_main.py --apply_discrete_gradients True --discrete_gradients_bins 18

python marvell_main.py --apply_trainable_layer True
# python marvell_main.py --apply_marvell True --apply_trainable_layer True
# python marvell_main.py --apply_mid True --apply_trainable_layer True
# python marvell_main.py --apply_encoder True --apply_trainable_layer True
# python marvell_main.py --apply_laplace True --apply_trainable_layer True
# python marvell_main.py --apply_gaussian True --apply_trainable_layer True
# python marvell_main.py --apply_grad_spar True --apply_trainable_layer True
# python marvell_main.py --apply_discrete_gradients True --discrete_gradients_bins 6 --apply_trainable_layer True
# python marvell_main.py --apply_discrete_gradients True --discrete_gradients_bins 12 --apply_trainable_layer True
# python marvell_main.py --apply_discrete_gradients True --discrete_gradients_bins 18 --apply_trainable_layer True

# # MID
# python vfl_main_task_no_defense.py --apply_mid True --mid_loss_lambda 1e-7 --epochs 20
# python vfl_dlg_no_defense.py --apply_mid True --mid_loss_lambda 1e-7
# python vfl_main_task_no_defense.py --apply_mid True --mid_loss_lambda 1e-6 --epochs 20
# python vfl_dlg_no_defense.py --apply_mid True --mid_loss_lambda 1e-6
# python vfl_main_task_no_defense.py --apply_mid True --mid_loss_lambda 1e-5 --epochs 20
# python vfl_dlg_no_defense.py --apply_mid True --mid_loss_lambda 1e-5
# python vfl_main_task_no_defense.py --apply_mid True --mid_loss_lambda 1e-4 --epochs 20
# python vfl_dlg_no_defense.py --apply_mid True --mid_loss_lambda 1e-4

# # Confusional Auto Encoder
# python vfl_main_task_no_defense.py --apply_encoder True --epochs 20
# python vfl_dlg_no_defense.py --apply_encoder True

# # No Derense
# python vfl_main_task_no_defense.py --epochs 20
# python vfl_dlg_no_defense.py

# Baseline defense
# python vfl_main_task_no_defense.py --apply_laplace True --epochs 20
# python vfl_dlg_no_defense.py --apply_laplace True
# python vfl_main_task_no_defense.py --apply_gaussian True --epochs 20
# python vfl_dlg_no_defense.py --apply_gaussian Tru
# python vfl_main_task_no_defense.py --apply_grad_spar True --epochs 20
# python vfl_dlg_no_defense.py --apply_grad_spar True
# python vfl_main_task_no_defense.py --apply_discrete_gradients True --discrete_gradients_bins 6 --epochs 20
# python vfl_dlg_no_defense.py --apply_discrete_gradients True --discrete_gradients_bins 6
# python vfl_main_task_no_defense.py --apply_discrete_gradients True --discrete_gradients_bins 12 --epochs 20
# python vfl_dlg_no_defense.py --apply_discrete_gradients True --discrete_gradients_bins 12
# python vfl_main_task_no_defense.py --apply_discrete_gradients True --discrete_gradients_bins 12 --epochs 20
# python vfl_dlg_no_defense.py --apply_discrete_gradients True --discrete_gradients_bins 12

# python vfl_main_task_no_defense.py --dataset_name cifar10 --apply_mid True --mid_loss_lambda 1e-9 --epochs 60
# python vfl_dlg_no_defense.py --dataset cifar10 --apply_mid True --mid_loss_lambda 1e-9
# python vfl_main_task_no_defense.py --dataset_name cifar10 --apply_mid True --mid_loss_lambda 1e-8 --epochs 60
# python vfl_dlg_no_defense.py --dataset cifar10 --apply_mid True --mid_loss_lambda 1e-8
# python vfl_main_task_no_defense.py --dataset_name cifar10 --apply_mid True --mid_loss_lambda 1e-7 --epochs 60
# python vfl_dlg_no_defense.py --dataset cifar10 --apply_mid True --mid_loss_lambda 1e-7
# python vfl_main_task_no_defense.py --dataset_name cifar10 --apply_mid True --mid_loss_lambda 1e-6 --epochs 60
# python vfl_dlg_no_defense.py --dataset cifar10 --apply_mid True --mid_loss_lambda 1e-6
# python vfl_main_task_no_defense.py --dataset_name cifar10 --apply_mid True --mid_loss_lambda 1e-5 --epochs 60
# python vfl_dlg_no_defense.py --dataset cifar10 --apply_mid True --mid_loss_lambda 1e-5
# python vfl_main_task_no_defense.py --dataset_name cifar10 --apply_mid True --mid_loss_lambda 1e-4 --epochs 60
# python vfl_dlg_no_defense.py --dataset cifar10 --apply_mid True --mid_loss_lambda 1e-4
# python vfl_main_task_no_defense.py --dataset_name cifar10 --apply_mid True --mid_loss_lambda 1e-3 --epochs 60
# python vfl_dlg_no_defense.py --dataset cifar10 --apply_mid True --mid_loss_lambda 1e-3
# python vfl_main_task_no_defense.py --dataset_name cifar10 --apply_mid True --mid_loss_lambda 1e-2 --epochs 60
# python vfl_dlg_no_defense.py --dataset cifar10 --apply_mid True --mid_loss_lambda 1e-2
# python vfl_main_task_no_defense.py --dataset_name cifar10 --apply_mid True --mid_loss_lambda 1e-1 --epochs 60
# python vfl_dlg_no_defense.py --dataset cifar10 --apply_mid True --mid_loss_lambda 1e-1
# python vfl_main_task_no_defense.py --dataset_name cifar10 --apply_mid True --mid_loss_lambda 1 --epochs 60
# python vfl_dlg_no_defense.py --dataset cifar10 --apply_mid True --mid_loss_lambda 1
# python vfl_main_task_no_defense.py --dataset_name cifar10 --apply_mid True --mid_loss_lambda 2 --epochs 60
# python vfl_dlg_no_defense.py --dataset cifar10 --apply_mid True --mid_loss_lambda 2


# python vfl_dlg_no_defense.py --dataset mnist --apply_mid True
# python vfl_main_task_no_defense.py --dataset_name mnist --apply_mid True --epochs 30
# python vfl_dlg_no_defense.py --dataset nuswide --apply_mid True
# python vfl_main_task_no_defense.py --dataset_name nuswide --apply_mid True --epochs 30
# python vfl_dlg_no_defense.py --dataset cifar100 --apply_mid True
# python vfl_main_task_no_defense.py --dataset_name cifar100 --apply_mid True --epochs 160



# # With Active Party Top Model
# # MID
# python vfl_main_task_no_defense.py --apply_mid True --mid_loss_lambda 1e-7 --epochs 30 --apply_trainable_layer True
# python vfl_dlg_no_defense.py --apply_mid True --mid_loss_lambda 1e-7 --apply_trainable_layer True
# python vfl_main_task_no_defense.py --apply_mid True --mid_loss_lambda 1e-6 --epochs 30 --apply_trainable_layer True
# python vfl_dlg_no_defense.py --apply_mid True --mid_loss_lambda 1e-6 --apply_trainable_layer True
# python vfl_main_task_no_defense.py --apply_mid True --mid_loss_lambda 1e-5 --epochs 30 --apply_trainable_layer True
# python vfl_dlg_no_defense.py --apply_mid True --mid_loss_lambda 1e-5 --apply_trainable_layer True
# python vfl_main_task_no_defense.py --apply_mid True --mid_loss_lambda 1e-4 --epochs 30 --apply_trainable_layer True
# python vfl_dlg_no_defense.py --apply_mid True --mid_loss_lambda 1e-4 --apply_trainable_layer True

# # Confusional Auto Encoder
# python vfl_main_task_no_defense.py --apply_encoder True --epochs 30 --apply_trainable_layer True
# python vfl_dlg_no_defense.py --apply_encoder True --apply_trainable_layer True

# # No Derense
# python vfl_main_task_no_defense.py --epochs 30 --apply_trainable_layer True
# python vfl_dlg_no_defense.py --apply_trainable_layer True

# Baseline defense
# python vfl_main_task_no_defense.py --apply_laplace True --epochs 30 --apply_trainable_layer True
# python vfl_dlg_no_defense.py --apply_laplace True --apply_trainable_layer True
# python vfl_main_task_no_defense.py --apply_gaussian True --epochs 30 --apply_trainable_layer True
# python vfl_dlg_no_defense.py --apply_gaussian Tru --apply_trainable_layer True
# python vfl_main_task_no_defense.py --apply_grad_spar True --epochs 30 --apply_trainable_layer True
# python vfl_dlg_no_defense.py --apply_grad_spar True --apply_trainable_layer True
# python vfl_main_task_no_defense.py --apply_discrete_gradients True --discrete_gradients_bins 6 --epochs 30 --apply_trainable_layer True
# python vfl_dlg_no_defense.py --apply_discrete_gradients True --discrete_gradients_bins 6 --apply_trainable_layer True
# python vfl_main_task_no_defense.py --apply_discrete_gradients True --discrete_gradients_bins 12 --epochs 30 --apply_trainable_layer True
# python vfl_dlg_no_defense.py --apply_discrete_gradients True --discrete_gradients_bins 12 --apply_trainable_layer True
# python vfl_main_task_no_defense.py --apply_discrete_gradients True --discrete_gradients_bins 12 --epochs 30 --apply_trainable_layer True
# python vfl_dlg_no_defense.py --apply_discrete_gradients True --discrete_gradients_bins 12 --apply_trainable_layer True
