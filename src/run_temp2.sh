# python vfl_dlg_no_defense.py --dataset mnist --apply_encoder True --gpu 2 --k 4
# python vfl_dlg_no_defense.py --dataset mnist --apply_encoder True --gpu 2 --k 25
python vfl_dlg_no_defense.py --dataset cifar10 --apply_encoder True --gpu 2 --k 4
python vfl_dlg_no_defense.py --dataset cifar10 --apply_encoder True --gpu 2 --k 25

# python vfl_main_task_no_defense.py --dataset_name mnist --epochs 20 --apply_encoder True --gpu 2 --k 4
# python vfl_main_task_no_defense.py --dataset_name mnist --epochs 20 --apply_encoder True --gpu 2 --k 25
python vfl_main_task_no_defense.py --dataset_name cifar10 --epochs 60 --apply_encoder True --gpu 2 --k 4
python vfl_main_task_no_defense.py --dataset_name cifar10 --epochs 60 --apply_encoder True --gpu 2 --k 25

