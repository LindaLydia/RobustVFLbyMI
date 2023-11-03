
# python vfl_main_task_no_defense.py --apply_mid True --dataset_name mnist --epochs 20
# python vfl_main_task_no_defense.py --apply_mid True --dataset_name cifar10 --epochs 160
# python vfl_main_task_no_defense.py --apply_mid True --dataset_name cifar100 --epochs 160

# python vfl_recovery_no_defense.py --dataset_name mnist --lr 0.0001
# python vfl_recovery_no_defense.py --dataset_name mnist --lr 0.005
# python vfl_recovery_no_defense.py --dataset_name mnist --lr 0.001
python vfl_recovery_no_defense.py --apply_gaussian True --dataset_name mnist --lr 0.005 --gpu 7
python vfl_recovery_no_defense.py --apply_mid True --dataset_name mnist --lr 0.005 --gpu 7
python vfl_recovery_no_defense.py --apply_dravl True --dataset_name mnist --lr 0.005 --gpu 7

# python vfl_recovery_no_defense.py --dataset_name cifar10 --lr 0.005 --gpu 7
python vfl_recovery_no_defense.py --dataset_name cifar100 --lr 0.005

python vfl_recovery_no_defense.py --apply_mid True --dataset_name cifar10 --lr 0.005 --gpu 7
python vfl_recovery_no_defense.py --apply_dravl True --dataset_name cifar10 --lr 0.005 --gpu 7
python vfl_recovery_no_defense.py --apply_mid True --dataset_name cifar100 --lr 0.005 --gpu 7
python vfl_recovery_no_defense.py --apply_dravl True --dataset_name cifar100 --lr 0.005 --gpu 7
