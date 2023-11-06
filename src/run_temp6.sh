# # python vfl_dlg_no_defense.py --dataset mnist --apply_encoder True --gpu 2 --k 4
# # python vfl_dlg_no_defense.py --dataset mnist --apply_encoder True --gpu 2 --k 25
# python vfl_dlg_no_defense.py --dataset cifar10 --apply_encoder True --gpu 2 --k 4
# python vfl_dlg_no_defense.py --dataset cifar10 --apply_encoder True --gpu 2 --k 25

# # python vfl_main_task_no_defense.py --dataset_name mnist --epochs 20 --apply_encoder True --gpu 2 --k 4
# # python vfl_main_task_no_defense.py --dataset_name mnist --epochs 20 --apply_encoder True --gpu 2 --k 25
# python vfl_main_task_no_defense.py --dataset_name cifar10 --epochs 60 --apply_encoder True --gpu 2 --k 4
# python vfl_main_task_no_defense.py --dataset_name cifar10 --epochs 60 --apply_encoder True --gpu 2 --k 25

# echo "2"
# echo "w/o"
# python vfl_main_task_no_defense_time.py --epoch 20 --k 2 --gpu 5
# echo "DP-L"
# python vfl_main_task_no_defense_time.py --apply_laplace True --epoch 20 --k 2 --gpu 5
# echo "mid"
# python vfl_main_task_no_defense_time.py --apply_mid True --epoch 20 --k 2 --gpu 5
# echo "dcore"
# python vfl_main_task_no_defense_time.py --apply_distance_correlation True --epoch 20 --k 2 --gpu 5

# echo "25"
# echo "w/o"
# python vfl_main_task_no_defense_time.py --epoch 20 --k 25 --gpu 5
# echo "DP-L"
# python vfl_main_task_no_defense_time.py --apply_laplace True --epoch 20 --k 25 --gpu 5
# echo "dcore"
# python vfl_main_task_no_defense_time.py --apply_distance_correlation True --epoch 20 --k 25 --gpu 5
# echo "mid"
# python vfl_main_task_no_defense_time.py --apply_mid True --epoch 20 --k 25 --gpu 5

# echo "4"
# echo "w/o"
# python vfl_main_task_no_defense_time.py --dataset cifar10 --epoch 60 --k 4
# echo "DP-L"
# python vfl_main_task_no_defense_time.py --dataset cifar10 --apply_laplace True --epoch 60 --k 4
# echo "mid"
# python vfl_main_task_no_defense_time.py --dataset cifar10 --apply_mid True --epoch 60 --k 4
# echo "dcore"
# python vfl_main_task_no_defense_time.py --dataset cifar10 --apply_distance_correlation True --epoch 60 --k 4

# echo "25"
# echo "w/o"
# python vfl_main_task_no_defense_time.py --dataset cifar10 --epoch 10 --k 25
# echo "DP-L"
# python vfl_main_task_no_defense_time.py --dataset cifar10 --apply_laplace True --epoch 10 --k 25
# echo "mid"
# python vfl_main_task_no_defense_time.py --dataset cifar10 --apply_mid True --epoch 10 --k 25
# echo "dcore"
# python vfl_main_task_no_defense_time.py --dataset cifar10 --apply_distance_correlation True --epoch 10 --k 25

echo "2"
echo "w/o"
python vfl_main_task_no_defense_time.py --dataset cifar10 --epoch 30 --k 2
echo "DP-L"
python vfl_main_task_no_defense_time.py --dataset cifar10 --apply_laplace True --epoch 30 --k 2
echo "mid"
python vfl_main_task_no_defense_time.py --dataset cifar10 --apply_mid True --epoch 30 --k 2
echo "dcore"
python vfl_main_task_no_defense_time.py --dataset cifar10 --apply_distance_correlation True --epoch 30 --k 2
