    python main.py --name defense --dataset cifar20 --model resnet18 --seed 2 --epoch 100 --backdoor 1 --mid 1 --mid_lambda 0.000001 --gpu 0
    python main.py --name defense --dataset cifar20 --model resnet18 --seed 2 --epoch 100 --backdoor 1 --mid 1 --mid_lambda 0.00001 --gpu 0
    python main.py --name defense --dataset cifar20 --model resnet18 --seed 2 --epoch 100 --backdoor 1 --mid 1 --mid_lambda 0.0001 --gpu 0
    python main.py --name defense --dataset cifar20 --model resnet18 --seed 2 --epoch 100 --backdoor 1 --mid 1 --mid_lambda 0.001 --gpu 0
for i in `seq 3 10`; do 
    python main.py --name defense --dataset cifar20 --model resnet18 --seed $i --epoch 100 --backdoor 1 --gpu 0
    python main.py --name defense --dataset cifar20 --model resnet18 --seed $i --epoch 100 --backdoor 1 --dp_type laplace --dp_strength 0.1 --gpu 0
    python main.py --name defense --dataset cifar20 --model resnet18 --seed $i --epoch 100 --backdoor 1 --dp_type laplace --dp_strength 0.01 --gpu 0
    python main.py --name defense --dataset cifar20 --model resnet18 --seed $i --epoch 100 --backdoor 1 --dp_type laplace --dp_strength 0.001 --gpu 0
    python main.py --name defense --dataset cifar20 --model resnet18 --seed $i --epoch 100 --backdoor 1 --dp_type laplace --dp_strength 0.0001 --gpu 0
    python main.py --name defense --dataset cifar20 --model resnet18 --seed $i --epoch 100 --backdoor 1 --dp_type gaussian --dp_strength 0.1 --gpu 0
    python main.py --name defense --dataset cifar20 --model resnet18 --seed $i --epoch 100 --backdoor 1 --dp_type gaussian --dp_strength 0.01 --gpu 0
    python main.py --name defense --dataset cifar20 --model resnet18 --seed $i --epoch 100 --backdoor 1 --dp_type gaussian --dp_strength 0.001 --gpu 0
    python main.py --name defense --dataset cifar20 --model resnet18 --seed $i --epoch 100 --backdoor 1 --dp_type gaussian --dp_strength 0.0001 --gpu 0
    python main.py --name defense --dataset cifar20 --model resnet18 --seed $i --epoch 100 --backdoor 1 --gradient_sparsification 95.0 --gpu 0
    python main.py --name defense --dataset cifar20 --model resnet18 --seed $i --epoch 100 --backdoor 1 --gradient_sparsification 99.0 --gpu 0
    python main.py --name defense --dataset cifar20 --model resnet18 --seed $i --epoch 100 --backdoor 1 --gradient_sparsification 99.5 --gpu 0
    python main.py --name defense --dataset cifar20 --model resnet18 --seed $i --epoch 100 --backdoor 1 --gradient_sparsification 99.9 --gpu 0
    python main.py --name defense --dataset cifar20 --model resnet18 --seed $i --epoch 100 --backdoor 1 --apply_discrete_gradients 1 --discrete_gradients_bins 6 --gpu 0
    python main.py --name defense --dataset cifar20 --model resnet18 --seed $i --epoch 100 --backdoor 1 --apply_discrete_gradients 1 --discrete_gradients_bins 12 --gpu 0
    python main.py --name defense --dataset cifar20 --model resnet18 --seed $i --epoch 100 --backdoor 1 --apply_discrete_gradients 1 --discrete_gradients_bins 18 --gpu 0
    python main.py --name defense --dataset cifar20 --model resnet18 --seed $i --epoch 100 --backdoor 1 --autoencoder 1 --lba 1.0 --gpu 0
    python main.py --name defense --dataset cifar20 --model resnet18 --seed $i --epoch 100 --backdoor 1 --autoencoder 1 --lba 0.5 --gpu 0
    python main.py --name defense --dataset cifar20 --model resnet18 --seed $i --epoch 100 --backdoor 1 --autoencoder 1 --lba 0.0 --gpu 0
    # python main.py --name defense --dataset cifar20 --model resnet18 --seed $i --epoch 100 --backdoor 1 --lba 1.0 --gpu 0 --apply_discrete_gradients 1 --discrete_gradients_bins 12 --gpu 0
    # python main.py --name defense --dataset cifar20 --model resnet18 --seed $i --epoch 100 --backdoor 1 --lba 0.5 --gpu 0 --apply_discrete_gradients 1 --discrete_gradients_bins 12 --gpu 0
    # python main.py --name defense --dataset cifar20 --model resnet18 --seed $i --epoch 100 --backdoor 1 --lba 0.0 --gpu 0 --apply_discrete_gradients 1 --discrete_gradients_bins 12 --gpu 0
    python main.py --name defense --dataset cifar20 --model resnet18 --seed $i --epoch 100 --backdoor 1 --mid 1 --mid_lambda 0.000001 --gpu 0
    python main.py --name defense --dataset cifar20 --model resnet18 --seed $i --epoch 100 --backdoor 1 --mid 1 --mid_lambda 0.00001 --gpu 0
    python main.py --name defense --dataset cifar20 --model resnet18 --seed $i --epoch 100 --backdoor 1 --mid 1 --mid_lambda 0.0001 --gpu 0
    python main.py --name defense --dataset cifar20 --model resnet18 --seed $i --epoch 100 --backdoor 1 --mid 1 --mid_lambda 0.001 --gpu 0
done