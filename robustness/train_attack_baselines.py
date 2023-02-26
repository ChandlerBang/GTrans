'''
python train_attack_baselines.py --dataset arxiv --loss=train+1LC --ptb_rate=0.1  --model=GCN --gpu_id=3 --debug=1  --seed=0  --ratio=0.2 --defense=jaccard
'''
import argparse
import numpy as np
from gtransform_adj_small import EdgeAgent
# from gtransform_adj_new import EdgeAgent
from gtransform_adj import EdgeAgent as AttackAgent
from gtransform_feat import FeatAgent
from utils import *
import torch
import random
from deeprobust.graph.defense import GCNJaccard, GCNSVD, RGCN, SimPGCN
from torch_geometric.utils import to_scipy_sparse_matrix, from_scipy_sparse_matrix
import time
from deeprobust.graph.data import Pyg2Dpr
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
parser.add_argument('--nlayers', type=int, default=2)
parser.add_argument('--model', type=str, default='GCN')
parser.add_argument('--loss', type=str, default='val')
parser.add_argument('--ptb_rate', type=float, default=-1)
parser.add_argument('--ratio', type=float, default=0.1)
parser.add_argument('--debug', type=float, default=0)
parser.add_argument('--noise_feature', type=float, default=0.0)
parser.add_argument('--noise_structure', type=float, default=0.0)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--defense', type=str, default='jaccard')
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
holdout_val(data)

clean_data = Pyg2Dpr([data])
clean_edge_index = data.edge_index.clone()
loss_backup = args.loss

import os.path as osp
filename = f'saved/attacked_graphs/{args.dataset}_r{args.ratio}_s{args.seed}.pt'
if args.debug and osp.exists(filename):
    edge_index, edge_weight = torch.load(filename, map_location='cuda'), None
else:
    agent = AttackAgent(data, args)
    args.ptb_rate = args.ratio
    if args.ptb_rate > 0:
        if args.dataset in ['cora', 'citeseer']:
        # if False:
            edge_index, edge_weight = agent.pgd_attack()
        else:
            edge_index, edge_weight = agent.attack_graph()
    else:
        edge_index, edge_weight = None, None
    if args.debug:
        torch.save(edge_index, filename)

data.edge_index = edge_index
agent = EdgeAgent(data, args)
model = agent.model
model.eval() # should set to eval
edge_index, feat, labels = data.edge_index, data.x, data.y
adj = to_scipy_sparse_matrix(edge_index).tocsr()

if args.defense == 'jaccard':
    defense = GCNJaccard(nfeat=feat.shape[1], nclass=labels.max().item()+1,
                            nhid=16, device='cpu')

    if args.dataset in ['arxiv']:
        defense.threshold=0.8; defense.binary_feature=False
    elif args.dataset in ['cora', 'citeseer']:
        defense.threshold=0.01; defense.binary_feature=True
    adj_preprocessed = defense.drop_dissimilar_edges(feat.cpu().numpy(), adj)
    edge_index_ = from_scipy_sparse_matrix(adj_preprocessed)[0].cuda()
    output = model.predict(feat, edge_index_)
    print('Test:', agent.get_perf(output, labels, data.test_mask, verbose=0)[1])
    print('Validation:', agent.get_perf(output, labels, data.val_mask, verbose=0)[1])
    print('Train:', agent.get_perf(output, labels, data.train_mask, verbose=0)[1])

    agent.compare_diff(clean_edge_index, edge_index, edge_index_)

if args.defense == 'gat':
    args.model = 'GAT'
    agent.pretrain_model()

if args.defense == 'appnp':
    args.model = 'APPNP'
    agent.pretrain_model()

if args.defense == 'rgcn':
    from models.rgcn import RGCN
    clean_adj = clean_data.adj
    device = 'cuda'
    feat, labels = clean_data.features, clean_data.labels
    idx_train, idx_val, idx_test = clean_data.idx_train, clean_data.idx_val, clean_data.idx_test
    model = RGCN(nnodes=feat.shape[0], nfeat=feat.shape[1], nclass=labels.max()+1, nlayers=2,
                with_bn=True, dropout=0.0, beta1=0e-7, beta2=0, lr=0.01, nhid=128, device=device)

    model = model.to(device)

    import os.path as osp
    filename = f'saved/{args.dataset}_RGCN_s{args.seed}.pt'
    if args.debug and osp.exists(filename):
        model.load_state_dict(torch.load(filename, map_location='cuda'))
        model.labels = torch.LongTensor(labels).to(device)
        model.test(idx_test, feat, clean_adj)
    else:
        model.fit(feat, clean_adj, labels, idx_train, idx_val, train_iters=1000, verbose=True)
        if args.debug:
            torch.save(model.state_dict(), filename)

        model.test(idx_test)

    perturbed_data = Pyg2Dpr([data.cpu()])
    perturbed_adj = perturbed_data.adj
    model.test(idx_test, feat, perturbed_adj)

if args.defense == 'simpgcn':
    from models.simpgcn import SimPGCN
    clean_adj = clean_data.adj
    device = 'cuda'
    feat, labels = clean_data.features, clean_data.labels
    idx_train, idx_val, idx_test = clean_data.idx_train, clean_data.idx_val, clean_data.idx_test
    model = SimPGCN(nnodes=feat.shape[0], nfeat=feat.shape[1], nhid=64, nclass=labels.max()+1, weight_decay=0, gamma=0.1,
                    lr=0.005, nlayers=2, dropout=0.1, with_bn=True, lambda_=0, device=device) # set lambda_ to 0 to train large graphs
    model = model.to(device)

    import os.path as osp
    filename = f'saved/{args.dataset}_SimPGCN_s{args.seed}.pt'
    if args.debug and osp.exists(filename):
        print('loading model state_dict...')
        model.load_state_dict(torch.load(filename, map_location='cuda'))
        model.labels = torch.LongTensor(labels).to(device)
        model.test(idx_test, feat, clean_adj)
    else:
        model.fit(feat, clean_adj, labels, idx_train, idx_val, train_iters=1000, verbose=True)
        if args.debug:
            torch.save(model.state_dict(), filename)

        model.test(idx_test)

    perturbed_data = Pyg2Dpr([data.cpu()])
    perturbed_adj = perturbed_data.adj
    model.test(idx_test, feat, perturbed_adj)
