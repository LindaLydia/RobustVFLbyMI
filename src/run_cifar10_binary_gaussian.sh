# # No Active Party Top Model
# python vfl_main_task_no_defense.py --apply_mid True --mid_loss_lambda 1e-7 --epochs 20
# python vfl_dlg_no_defense.py --apply_mid True --mid_loss_lambda 1e-7
# python vfl_main_task_no_defense.py --apply_mid True --mid_loss_lambda 1e-6 --epochs 20
# python vfl_dlg_no_defense.py --apply_mid True --mid_loss_lambda 1e-6
# python vfl_main_task_no_defense.py --apply_mid True --mid_loss_lambda 1e-5 --epochs 20
# python vfl_dlg_no_defense.py --apply_mid True --mid_loss_lambda 1e-5
# python vfl_main_task_no_defense.py --apply_mid True --mid_loss_lambda 1e-4 --epochs 20
# python vfl_dlg_no_defense.py --apply_mid True --mid_loss_lambda 1e-4
# python vfl_main_task_no_defense.py --apply_encoder True --epochs 30
# python vfl_dlg_no_defense.py --apply_encoder True


# With Active Party Top Model
# python vfl_main_task_no_defense.py --apply_mid True --mid_loss_lambda 1e-7 --epochs 300 --apply_trainable_layer True --batch_size 1
# python vfl_dlg_no_defense.py --apply_mid True --mid_loss_lambda 1e-7 --apply_trainable_layer True --gpu 1
# python vfl_main_task_no_defense.py --apply_mid True --mid_loss_lambda 1e-6 --epochs 300 --apply_trainable_layer True --batch_size 1
# python vfl_dlg_no_defense.py --apply_mid True --mid_loss_lambda 1e-6 --apply_trainable_layer True --gpu 1
# python vfl_main_task_no_defense.py --apply_mid True --mid_loss_lambda 1e-5 --epochs 300 --apply_trainable_layer True --batch_size 1
# python vfl_dlg_no_defense.py --apply_mid True --mid_loss_lambda 1e-5 --apply_trainable_layer True --gpu 1
# python vfl_main_task_no_defense.py --apply_mid True --mid_loss_lambda 1e-4 --epochs 300 --apply_trainable_layer True --batch_size 1
# python vfl_dlg_no_defense.py --apply_mid True --mid_loss_lambda 1e-4 --apply_trainable_layer True --gpu 1
# python vfl_main_task_no_defense.py --apply_encoder True --epochs 300 --apply_trainable_layer True --batch_size 1
# python vfl_dlg_no_defense.py --apply_encoder True --apply_trainable_layer True --gpu 1

# python vfl_main_task_no_defense_binary.py --dataset_name cifar10 --epochs 160 --gpu 0
# python vfl_dlg_no_defense.py --dataset cifar10 --gpu 0
# python vfl_main_task_no_defense_binary.py --dataset_name cifar10 --apply_mid True --epochs 160 --gpu 0
# python vfl_dlg_no_defense.py --dataset cifar10 --apply_mid True --gpu 0
# python vfl_main_task_no_defense_binary.py --dataset_name cifar10 --apply_gaussian True --epochs 160 --gpu 0
# python vfl_dlg_no_defense.py --dataset cifar10 --apply_gaussian True --gpu 0
# python vfl_main_task_no_defense_binary.py --dataset_name cifar10 --apply_laplace True --epochs 160 --gpu 0
# python vfl_dlg_no_defense.py --dataset cifar10 --apply_laplace True --gpu 0
# python vfl_main_task_no_defense_binary.py --dataset_name cifar10 --apply_grad_spar True --epochs 160 --gpu 0
# python vfl_dlg_no_defense.py --dataset cifar10 --apply_grad_spar True --gpu 0
python vfl_main_task_no_defense_binary.py --dataset_name cifar10 --apply_marvell True --marvell_s 1 --epochs 160 --gpu 0
python vfl_main_task_no_defense_binary.py --dataset_name cifar10 --apply_marvell True --marvell_s 2 --epochs 160 --gpu 0
python vfl_main_task_no_defense_binary.py --dataset_name cifar10 --apply_marvell True --marvell_s 5 --epochs 160 --gpu 0
python vfl_main_task_no_defense_binary.py --dataset_name cifar10 --apply_marvell True --marvell_s 10 --epochs 160 --gpu 0
python vfl_dlg_no_defense.py --dataset cifar10 --apply_marvell True --marvell_s 1 --gpu 0
python vfl_dlg_no_defense.py --dataset cifar10 --apply_marvell True --marvell_s 2 --gpu 0
python vfl_dlg_no_defense.py --dataset cifar10 --apply_marvell True --marvell_s 5 --gpu 0
python vfl_dlg_no_defense.py --dataset cifar10 --apply_marvell True --marvell_s 10 --gpu 0

# python vfl_main_task_no_defense.py --apply_mid True --dataset_name cifar100 --epochs 130 --gpu 1
# python vfl_main_task_no_defense.py --apply_gaussian True --dataset_name cifar100 --epochs 130 --gpu 1
# python vfl_main_task_no_defense.py --apply_laplace True --dataset_name cifar100 --epochs 130 --gpu 1
# python vfl_main_task_no_defense.py --apply_grad_spar True --dataset_name cifar100 --epochs 130 --gpu 1
# python vfl_main_task_no_defense.py --apply_mid True --dataset_name cifar10 --epochs 160 --gpu 2
# python vfl_main_task_no_defense.py --apply_gaussian True --dataset_name cifar10 --epochs 160 --gpu 2
# python vfl_main_task_no_defense.py --apply_laplace True --dataset_name cifar10 --epochs 160 --gpu 2
# python vfl_main_task_no_defense.py --apply_grad_spar True --dataset_name cifar10 --epochs 160 --gpu 2
