gpu_id=${1}
# ogb-arxiv
python main.py --method erm --gnn gcn --lr 0.01 --dataset ogb-arxiv --device ${gpu_id}

python main.py --method eerm --gnn gcn --lr 0.005 --K 5 --T 5 --num_sample 1 --beta 0.5 --lr_a 0.01 --dataset ogb-arxiv --device ${gpu_id}

python main.py --method erm --gnn sage --lr 0.01 --dataset ogb-arxiv --device ${gpu_id}

python main.py --method eerm --gnn sage --lr 0.005 --K 5 --T 5 --num_sample 1 --beta 0.5 --lr_a 0.01 --dataset ogb-arxiv --device ${gpu_id}

python main.py --method erm --gnn gpr --lr 0.01 --dataset ogb-arxiv --device ${gpu_id}

python main.py --method eerm --gnn gpr --lr 0.01 --K 3 --T 5 --num_sample 1 --beta 1.0 --lr_a 0.001 --dataset ogb-arxiv --device ${gpu_id}

