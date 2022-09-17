for i in `seq 1 10`; do 
    python main_missing.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --gpu 0 --missing_rate 4
    python main_missing.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --dp_type laplace --dp_strength 1 --missing_rate 4 --gpu 0
    python main_missing.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --dp_type laplace --dp_strength 0.1 --missing_rate 4 --gpu 0
    python main_missing.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --dp_type laplace --dp_strength 0.01 --missing_rate 4 --gpu 0
    python main_missing.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --dp_type laplace --dp_strength 0.001 --missing_rate 4 --gpu 0
    python main_missing.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --dp_type gaussian --dp_strength 1 --missing_rate 4 --gpu 0
    python main_missing.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --dp_type gaussian --dp_strength 0.1 --missing_rate 4 --gpu 0
    python main_missing.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --dp_type gaussian --dp_strength 0.01 --missing_rate 4 --gpu 0
    python main_missing.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --dp_type gaussian --dp_strength 0.001 --missing_rate 4 --gpu 0
    python main_missing.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --gradient_sparsification 95.0 --missing_rate 4 --gpu 0
    python main_missing.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --gradient_sparsification 99.0 --missing_rate 4 --gpu 0
    python main_missing.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --gradient_sparsification 99.5 --missing_rate 4 --gpu 0
    python main_missing.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --gradient_sparsification 99.9 --missing_rate 4 --gpu 0
    python main_missing.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --apply_discrete_gradients 1 --discrete_gradients_bins 6 --missing_rate 4 --gpu 0
    python main_missing.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --apply_discrete_gradients 1 --discrete_gradients_bins 12 --missing_rate 4 --gpu 0
    python main_missing.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --apply_discrete_gradients 1 --discrete_gradients_bins 18 --missing_rate 4 --gpu 0
    python main_missing.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --autoencoder 1 --lba 1.0 --missing_rate 4 --gpu 0
    python main_missing.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --autoencoder 1 --lba 0.5 --missing_rate 4 --gpu 0
    python main_missing.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --autoencoder 1 --lba 0.0 --missing_rate 4 --gpu 0
    python main_missing.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --autoencoder 1 --lba 1.0 --apply_discrete_gradients 1 --discrete_gradients_bins 12 --missing_rate 4 --gpu 0
    python main_missing.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --autoencoder 1 --lba 0.5 --apply_discrete_gradients 1 --discrete_gradients_bins 12 --missing_rate 4 --gpu 0
    python main_missing.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --autoencoder 1 --lba 0.0 --apply_discrete_gradients 1 --discrete_gradients_bins 12 --missing_rate 4 --gpu 0
    python main_missing.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --mid 1 --mid_lambda 0.00001 --missing_rate 4 --gpu 0
    python main_missing.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --mid 1 --mid_lambda 0.0001 --missing_rate 4 --gpu 0
    python main_missing.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --mid 1 --mid_lambda 0.001 --missing_rate 4 --gpu 0
    python main_missing.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --mid 1 --mid_lambda 0.01 --missing_rate 4 --gpu 0
    python main_missing.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --mid 1 --mid_lambda 0.1 --missing_rate 4 --gpu 0
    python main_missing.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --mid 1 --mid_lambda 1 --missing_rate 4 --gpu 0
done

# for i in `seq 1 10`; do 
#     python main_missing.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --gpu 0 --missing_rate 2
#     python main_missing.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --dp_type laplace --dp_strength 1 --missing_rate 2 --gpu 0
#     python main_missing.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --dp_type laplace --dp_strength 0.1 --missing_rate 2 --gpu 0
#     python main_missing.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --dp_type laplace --dp_strength 0.01 --missing_rate 2 --gpu 0
#     python main_missing.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --dp_type laplace --dp_strength 0.001 --missing_rate 2 --gpu 0
#     python main_missing.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --dp_type gaussian --dp_strength 1 --missing_rate 2 --gpu 0
#     python main_missing.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --dp_type gaussian --dp_strength 0.1 --missing_rate 2 --gpu 0
#     python main_missing.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --dp_type gaussian --dp_strength 0.01 --missing_rate 2 --gpu 0
#     python main_missing.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --dp_type gaussian --dp_strength 0.001 --missing_rate 2 --gpu 0
#     python main_missing.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --gradient_sparsification 95.0 --missing_rate 2 --gpu 0
#     python main_missing.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --gradient_sparsification 99.0 --missing_rate 2 --gpu 0
#     python main_missing.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --gradient_sparsification 99.5 --missing_rate 2 --gpu 0
#     python main_missing.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --gradient_sparsification 99.9 --missing_rate 2 --gpu 0
#     python main_missing.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --apply_discrete_gradients 1 --discrete_gradients_bins 6 --missing_rate 2 --gpu 0
#     python main_missing.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --apply_discrete_gradients 1 --discrete_gradients_bins 12 --missing_rate 2 --gpu 0
#     python main_missing.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --apply_discrete_gradients 1 --discrete_gradients_bins 18 --missing_rate 2 --gpu 0
#     python main_missing.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --autoencoder 1 --lba 1.0 --missing_rate 2 --gpu 0
#     python main_missing.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --autoencoder 1 --lba 0.5 --missing_rate 2 --gpu 0
#     python main_missing.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --autoencoder 1 --lba 0.0 --missing_rate 2 --gpu 0
#     python main_missing.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --autoencoder 1 --lba 1.0 --apply_discrete_gradients 1 --discrete_gradients_bins 12 --missing_rate 2 --gpu 0
#     python main_missing.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --autoencoder 1 --lba 0.5 --apply_discrete_gradients 1 --discrete_gradients_bins 12 --missing_rate 2 --gpu 0
#     python main_missing.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --autoencoder 1 --lba 0.0 --apply_discrete_gradients 1 --discrete_gradients_bins 12 --missing_rate 2 --gpu 0
#     python main_missing.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --mid 1 --mid_lambda 0.00001 --missing_rate 2 --gpu 0
#     python main_missing.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --mid 1 --mid_lambda 0.0001 --missing_rate 2 --gpu 0
#     python main_missing.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --mid 1 --mid_lambda 0.001 --missing_rate 2 --gpu 0
#     python main_missing.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --mid 1 --mid_lambda 0.01 --missing_rate 2 --gpu 0
#     python main_missing.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --mid 1 --mid_lambda 0.1 --missing_rate 2 --gpu 0
#     python main_missing.py --name defense --dataset nuswide --model mlp2 --seed $i --epoch 100 --backdoor 1 --mid 1 --mid_lambda 1 --missing_rate 2 --gpu 0
# done