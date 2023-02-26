"""python train_abnormal_baselines.py --hidden=256 --noise_feature=0.2 --ptb_rate=0. --dataset cora --gpu_id=0 --model=GCN"""
import argparse
import numpy as np
from deeprobust.graph.defense_pyg import AirGNN, GCN, APPNP, GAT, SAGE, GPRGNN
from utils import *
import torch
import random
from gtransform_feat import add_feature_noise, add_structure_noise, add_feature_noise_test
import torch.nn.functional as F
import time
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
parser.add_argument('--nlayers', type=int, default=2)
parser.add_argument('--model', type=str, default='GCN')
parser.add_argument('--debug', type=float, default=0)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--noise_feature', type=float, default=0.0)
parser.add_argument('--tune', type=float, default=0)
parser.add_argument('--lambda_', type=float, default=0)
args = parser.parse_args()

torch.cuda.set_device(args.gpu_id)

def reset_args(args):
    args.with_bn = False
    if args.dataset == 'arxiv':
        args.nlayers = 3; args.hidden=256; args.with_bn=True
    elif args.dataset in ['cora', 'citeseer', 'pubmed']:
        args.nlayers = 2; args.hidden=64
    else:
        raise NotImplementedError

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
# resplit(data)

def pretrain_model():
    feat, labels = data.x, data.y
    nclass = max(labels).item()+1

    if args.model == "GCN":
        model = GCN(nfeat=feat.shape[1], nhid=args.hidden, dropout=args.dropout,
                nlayers=args.nlayers, with_bn=args.with_bn,
                weight_decay=args.weight_decay, nclass=nclass,
                device=device).to(device)
    elif args.model == "GAT":
        if args.dataset == 'arxiv':
            args.dropout = 0.2; args.hidden = 32; args.nlayers = 2
        else:
            args.dropout = 0.5; args.hidden = 8
        model = GAT(nfeat=feat.shape[1], nhid=args.hidden, heads=8, lr=0.005, nlayers=args.nlayers,
              nclass=nclass, with_bn=args.with_bn, weight_decay=args.weight_decay,
              dropout=args.dropout, device=device).to(device)
    elif args.model == "AirGNN" or args.model== "APPNP":
         if args.dataset == 'arxiv':
             args.dropout=0.8; args.lambda_amp=0.3; args.alpha=0.1; args.weight_decay=0
         else:
             args.dropout=0.8; args.lambda_amp=0.5; args.alpha=0.1

         if args.model == "APPNP": args.dropout = 0.2
         if args.tune:
             args.lambda_amp = args.lambda_

         model = AirGNN(nfeat=feat.shape[1], nhid=args.hidden, dropout=args.dropout, with_bn=args.with_bn,
                 K=10, weight_decay=args.weight_decay, args=args, nlayers=args.nlayers,
                 nclass=max(labels).item()+1, device=device).to(device)
    elif args.model == "SAGE":
        model = SAGE(feat.shape[1], 32, max(labels).item()+1, num_layers=5, dropout=0.0, lr=0.01, weight_decay=0, device=device).to(device)
    elif args.model == "GPR":
        model = GPRGNN(feat.shape[1], 32, max(labels).item()+1, dropout=0.0, lr=0.01, weight_decay=0, device=device).to(device)
    else:
        raise NotImplementedError

    print(model)
    train_iters = 1000
    import os.path as osp
    filename = f'saved/{args.dataset}_{args.model}_s{args.seed}.pt'
    if args.debug and osp.exists(filename):
        model.load_state_dict(torch.load(filename, map_location='cuda'))
    else:
        model.fit(data, train_iters=1000, patience=1000, verbose=True)
        if args.debug:
            torch.save(model.state_dict(), filename)

    model.eval()
    model.data = data.to(device)
    output = model.predict()
    labels = labels.to(device)
    print("Test set results:", get_perf(output, labels, data.test_mask, verbose=0)[1])
    return model

device = 'cuda'
model = pretrain_model()

if args.noise_feature > 0:
    feat_noise, noisy_nodes = add_feature_noise_test(data,
            args.noise_feature, args.seed)


output = model.predict()
labels = data.y.to(device)
print("After noise, test set results:", get_perf(output, labels, data.test_mask, verbose=0)[1])
print('Validation:', get_perf(output, labels, data.val_mask, verbose=0)[1])
print('Abnomral test nodes:', get_perf(output, labels, noisy_nodes, verbose=0)[1])
print('Normal test nodes:', get_perf(output, labels, data.test_mask & (~noisy_nodes), verbose=0)[1])


