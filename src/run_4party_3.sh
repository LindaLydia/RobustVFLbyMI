python vfl_dlg_no_defense.py --dataset cifar10 --gpu 0 --k 4
python vfl_dlg_no_defense.py --dataset cifar10 --apply_mid True --gpu 0 --k 4
python vfl_dlg_no_defense.py --dataset cifar10 --apply_encoder True --gpu 0 --k 4
python vfl_dlg_no_defense.py --dataset cifar10 --apply_laplace True --gpu 0 --k 4
python vfl_dlg_no_defense.py --dataset cifar10 --apply_gaussian True --gpu 0 --k 4
python vfl_dlg_no_defense.py --dataset cifar10 --apply_grad_spar True --gpu 0 --k 4
python vfl_dlg_no_defense.py --dataset cifar10 --apply_RRwithPrior True --gpu 0 --k 4
python vfl_dlg_no_defense.py --dataset cifar10 --apply_grad_perturb True --gpu 0 --k 4
python vfl_dlg_no_defense.py --dataset cifar10 --apply_distance_correlation True --gpu 0 --k 4

