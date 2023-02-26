import argparse
import numpy as np
from gtransform_adj_small import EdgeAgent
from gtransform_both import GraphAgent
from gtransform_adj import EdgeAgent as AttackAgent
from gtransform_feat import FeatAgent
from utils import *
import torch
import random
import time
st = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--hidden', type=int, default=128)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--seed', type=int, default=0, help='Random seed.')
parser.add_argument('--lr_adj', type=float, default=0.01)
parser.add_argument('--lr_feat', type=float, default=1e-3)
parser.add_argument('--nlayers', type=int, default=2)
parser.add_argument('--model', type=str, default='GCN')
parser.add_argument('--loss', type=str, default='train+1LC')
parser.add_argument('--ptb_rate', type=float, default=0.1)
parser.add_argument('--ratio', type=float, default=0.3, help='Budget B for changing graph sturcture')
parser.add_argument('--debug', type=float, default=0)
parser.add_argument('--noise_feature', type=float, default=0.0, help='ratio of nodes with abnormal features')
parser.add_argument('--noise_structure', type=float, default=0.0)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--loop_adj', type=int, default=1, help='#loops for optimizing structure')
parser.add_argument('--loop_feat', type=int, default=4, help='#loops for optimizing features')
parser.add_argument('--finetune', type=int, default=0)
parser.add_argument('--existing_space', type=int, default=1)
args = parser.parse_args()

torch.cuda.set_device(args.gpu_id)
print('===========')
print(args)

# random seed setting
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

dataset = get_dataset(args.dataset, args.normalize_features, if_dpr=False)
data = dataset[0]
if not hasattr(data, 'train_mask'):
    add_mask(data, dataset)
# resplit(data)

clean_edge_index = data.edge_index.clone()
loss_backup = args.loss

import os.path as osp
filename = f'saved/attacked_graphs/{args.dataset}_r{args.ptb_rate}_s{args.seed}.pt'
if args.debug and osp.exists(filename):
    edge_index, edge_weight = torch.load(filename, map_location='cuda'), None
else:
    agent = AttackAgent(data, args)
    if args.ptb_rate > 0:
        if args.dataset in ['cora', 'citeseer']:
            edge_index, edge_weight = agent.pgd_attack()
        else:
            from deeprobust.graph.global_attack import PRBCD
            attacker = PRBCD(data)
            edge_index, edge_weight = attacker.attack(ptb_rate=args.ptb_rate)
            # edge_index, edge_weight = agent.attack_graph() # method2: you can also use this function to generate attack
        if args.debug:
            torch.save(edge_index, filename)
    else:
        edge_index, edge_weight = clean_edge_index.cuda(), None

args.loss = loss_backup
data.edge_index = edge_index

if args.finetune:
    agent = FeatAgent(data, args)
    agent.finetune(edge_index=edge_index, edge_weight=edge_weight, train_iters=100, verbose=1)
    assert False

agent = GraphAgent(data, args)
args.ratio = args.ratio; args.lr_adj=0.1; args.epochs = 50
new_feat, new_edge_index, new_edge_weight = agent.learn_graph(edge_index, edge_weight)


