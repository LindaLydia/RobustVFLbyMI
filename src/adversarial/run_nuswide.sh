for i in `seq 1 10`; do 
#     python main.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --gpu 0
#     python main.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --dp_type laplace --dp_strength 1 --gpu 0
#     python main.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --dp_type laplace --dp_strength 0.5 --gpu 0
#     python main.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --dp_type laplace --dp_strength 0.1 --gpu 0
#     python main.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --dp_type laplace --dp_strength 0.01 --gpu 0
#     python main.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --dp_type laplace --dp_strength 0.001 --gpu 0
#     python main.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --dp_type laplace --dp_strength 0.0001 --gpu 0
#     python main.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --dp_type gaussian --dp_strength 1 --gpu 0
#     python main.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --dp_type gaussian --dp_strength 0.5 --gpu 0
#     python main.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --dp_type gaussian --dp_strength 0.1 --gpu 0
#     python main.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --dp_type gaussian --dp_strength 0.01 --gpu 0
#     python main.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --dp_type gaussian --dp_strength 0.001 --gpu 0
#     python main.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --dp_type gaussian --dp_strength 0.0001 --gpu 0
#     python main.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --gradient_sparsification 95.0 --gpu 0
#     python main.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --gradient_sparsification 99.0 --gpu 0
#     python main.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --gradient_sparsification 99.5 --gpu 0
#     python main.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --gradient_sparsification 99.9 --gpu 0
#     python main.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --apply_discrete_gradients 1 --discrete_gradients_bins 6 --gpu 0
#     python main.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --apply_discrete_gradients 1 --discrete_gradients_bins 12 --gpu 0
#     python main.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --apply_discrete_gradients 1 --discrete_gradients_bins 18 --gpu 0
#     python main.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --autoencoder 1 --lba 1.0 --gpu 0
#     python main.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --autoencoder 1 --lba 0.5 --gpu 0
#     python main.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --autoencoder 1 --lba 0.0 --gpu 0
#     python main.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --lba 1.0 --apply_discrete_gradients 1 --discrete_gradients_bins 12 --gpu 0
#     python main.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --lba 0.5 --apply_discrete_gradients 1 --discrete_gradients_bins 12 --gpu 0
#     python main.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --lba 0.0 --apply_discrete_gradients 1 --discrete_gradients_bins 12 --gpu 0
    # python main.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --mid 1 --mid_lambda 0.000001 --gpu 0
    # python main.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --mid 1 --mid_lambda 0.00001 --gpu 0 --learning_rate 0.0001
    # python main.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --mid 1 --mid_lambda 0.0001 --gpu 0 --learning_rate 0.0001
    # python main.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --mid 1 --mid_lambda 0.001 --gpu 0 --learning_rate 0.0001
    # python main.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --mid 1 --mid_lambda 0.01 --gpu 0 --learning_rate 0.0001
    # python main.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --mid 1 --mid_lambda 0.1 --gpu 0 --learning_rate 0.0001
    # python main.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --mid 1 --mid_lambda 0.5 --gpu 0 --learning_rate 0.0001
    # python main.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --mid 1 --mid_lambda 1.0 --gpu 0 --learning_rate 0.0001
    python main.py --name defense_test0.0 --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --mid 1 --mid_lambda 0.0 --gpu 0 --learning_rate 0.0001
    python main_missing.py --name defense_test0.0 --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --mid 1 --mid_lambda 0.0 --gpu 0 --learning_rate 0.0001
done

# for i in `seq 1 10`; do 
#     python main_missing.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --gpu 0
#     python main_missing.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --dp_type laplace --dp_strength 0.1 --gpu 0
#     python main_missing.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --dp_type laplace --dp_strength 0.01 --gpu 0
#     python main_missing.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --dp_type laplace --dp_strength 0.001 --gpu 0
#     python main_missing.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --dp_type laplace --dp_strength 0.0001 --gpu 0
#     python main_missing.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --dp_type gaussian --dp_strength 0.1 --gpu 0
#     python main_missing.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --dp_type gaussian --dp_strength 0.01 --gpu 0
#     python main_missing.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --dp_type gaussian --dp_strength 0.001 --gpu 0
#     python main_missing.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --dp_type gaussian --dp_strength 0.0001 --gpu 0
#     python main_missing.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --gradient_sparsification 95.0 --gpu 0
#     python main_missing.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --gradient_sparsification 99.0 --gpu 0
#     python main_missing.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --gradient_sparsification 99.5 --gpu 0
#     python main_missing.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --gradient_sparsification 99.9 --gpu 0
#     python main_missing.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --apply_discrete_gradients 1 --discrete_gradients_bins 6 --gpu 0
#     python main_missing.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --apply_discrete_gradients 1 --discrete_gradients_bins 12 --gpu 0
#     python main_missing.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --apply_discrete_gradients 1 --discrete_gradients_bins 18 --gpu 0
#     python main_missing.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --autoencoder 1 --lba 1.0 --gpu 0
#     python main_missing.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --autoencoder 1 --lba 0.5 --gpu 0
#     python main_missing.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --autoencoder 1 --lba 0.0 --gpu 0
#     python main_missing.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --lba 1.0 --apply_discrete_gradients 1 --discrete_gradients_bins 12 --gpu 0
#     python main_missing.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --lba 0.5 --apply_discrete_gradients 1 --discrete_gradients_bins 12 --gpu 0
#     python main_missing.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --lba 0.0 --apply_discrete_gradients 1 --discrete_gradients_bins 12 --gpu 0
#     python main_missing.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --mid 1 --mid_lambda 0.000001 --gpu 0
#     python main_missing.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --mid 1 --mid_lambda 0.00001 --gpu 0
#     python main_missing.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --mid 1 --mid_lambda 0.0001 --gpu 0
#     python main_missing.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --mid 1 --mid_lambda 0.001 --gpu 0
# done

# for i in `seq 1 10`; do
#     python main_missing_rvfr.py --name defense --dataset cifar20 --model resnet18 --seed $i --epoch 100 --backdoor 1 --rvfr 1 --rvfr_alpha 1.0 --quarantine_epochs 10 --rae_pretrain_epochs 100 --rae_tune_epochs 100 --missing_rate 2 --gpu 0
#     python main_missing_rvfr.py --name defense --dataset cifar20 --model resnet18 --seed $i --epoch 100 --backdoor 1 --rvfr 1 --rvfr_alpha 1.0 --quarantine_epochs 10 --rae_pretrain_epochs 100 --rae_tune_epochs 50 --missing_rate 2 --gpu 0
#     python main_missing_rvfr.py --name defense --dataset cifar20 --model resnet18 --seed $i --epoch 100 --backdoor 1 --rvfr 1 --rvfr_alpha 1.0 --quarantine_epochs 10 --rae_pretrain_epochs 100 --rae_tune_epochs 10 --missing_rate 2 --gpu 0
# done

for i in `seq 1 10`; do
    python main.py --name defense --dataset mnist --model mlp2 --seed $i --epoch 100 --backdoor 1 --mid 1 --mid_lambda 0.1 --gpu 0
    python main.py --name defense --dataset mnist --model mlp2 --seed $i --epoch 100 --backdoor 1 --mid 1 --mid_lambda 1 --gpu 0
done
