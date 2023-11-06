# # python vfl_dlg_no_defense.py --apply_encoder True --gpu 1 --k 25

# python vfl_dlg_no_defense.py --dataset cifar10 --gpu 1 --k 25
# python vfl_dlg_no_defense.py --dataset cifar10 --apply_mid True --gpu 1 --k 25
# python vfl_dlg_no_defense.py --dataset cifar10 --apply_encoder True --gpu 1 --k 25
# python vfl_dlg_no_defense.py --dataset cifar10 --apply_laplace True --gpu 1 --k 25
python vfl_dlg_no_defense.py --dataset cifar10 --apply_gaussian True --gpu 1 --k 25
python vfl_dlg_no_defense.py --dataset cifar10 --apply_grad_spar True --gpu 1 --k 25
python vfl_dlg_no_defense.py --dataset cifar10 --apply_RRwithPrior True --gpu 1 --k 25
python vfl_dlg_no_defense.py --dataset cifar10 --apply_grad_perturb True --gpu 1 --k 25
python vfl_dlg_no_defense.py --dataset cifar10 --apply_distance_correlation True --gpu 1 --k 25

python vfl_dlg_no_defense.py --dataset cifar10 --gpu 1 --k 25
python vfl_dlg_no_defense.py --dataset cifar10 --apply_mid True --gpu 1 --k 25
# python vfl_dlg_no_defense.py --dataset cifar10 --apply_encoder True --gpu 1 --k 25
python vfl_dlg_no_defense.py --dataset cifar10 --apply_laplace True --gpu 1 --k 25
python vfl_dlg_no_defense.py --dataset cifar10 --apply_encoder True --gpu 1 --k 25
