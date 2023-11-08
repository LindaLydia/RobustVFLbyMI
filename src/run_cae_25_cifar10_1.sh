echo "===30epoch, lr=5e-5"
python vfl_main_task_no_defense.py --apply_encoder True --dataset_name cifar10 --epochs 30 --lr 0.00005 --k 25 --gpu 5
echo "===30epoch, lr=1e-4"
python vfl_main_task_no_defense.py --apply_encoder True --dataset_name cifar10 --epochs 30 --lr 0.0001 --k 25 --gpu 5
echo "===30epoch, lr=1e-3"
python vfl_main_task_no_defense.py --apply_encoder True --dataset_name cifar10 --epochs 30 --lr 0.001 --k 25 --gpu 5
