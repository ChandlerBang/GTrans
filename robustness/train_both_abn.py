import argparse
import numpy as np
from gtransform_both import GraphAgent
from gtransform_feat import FeatAgent
from utils import *
import torch
import random
import time
torch.autograd.set_detect_anomaly(True)
st = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--hidden', type=int, default=128)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--seed', type=int, default=0, help='Random seed.')
parser.add_argument('--dropout', type=float, default=0)
parser.add_argument('--lr_adj', type=float, default=0.1)
parser.add_argument('--lr_feat', type=float, default=0.0001)
parser.add_argument('--nlayers', type=int, default=2)
parser.add_argument('--model', type=str, default='GCN')
parser.add_argument('--loss', type=str, default='LC')
parser.add_argument('--ptb_rate', type=float, default=-1)
parser.add_argument('--ratio', type=float, default=0.05)
parser.add_argument('--debug', type=float, default=0)
parser.add_argument('--noise_feature', type=float, default=0.0)
parser.add_argument('--noise_structure', type=float, default=0.0)
parser.add_argument('--loop_adj', type=int, default=1)
parser.add_argument('--loop_feat', type=int, default=4)
parser.add_argument('--finetune', type=int, default=0)
parser.add_argument('--existing_space', type=int, default=1)
args = parser.parse_args()

torch.cuda.set_device(args.gpu_id)
print('===========')

reset_args(args)
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

agent = GraphAgent(data, args)
if args.ptb_rate > 0:
    edge_index, edge_weight = agent.pgd_attack()
else:
    edge_index, edge_weight = None, None

if args.finetune:
    agent.finetune(edge_index=edge_index, edge_weight=edge_weight, train_iters=100)
else:
    feat, edge_index, edge_weight = agent.learn_graph(edge_index_=edge_index, edge_weight=edge_weight)
    cross_arc = False
    if cross_arc:
        agent.data.x = feat
        agent.data.edge_index, agent.data.edge_weight = edge_index, edge_weight
        for m in ['GCN', 'APPNP', 'GAT', 'AirGNN']:
            print(f'===={m}====')
            args.model = m
            agent.pretrain_model()

