import argparse
import sys
import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected
from torch_scatter import scatter

from logger import Logger, SimpleLogger
from dataset import load_nc_dataset
from data_utils import normalize, gen_normalized_adjs, evaluate, evaluate_whole_graph, eval_acc, eval_rocauc, eval_f1, to_sparse_tensor, load_fixed_splits
from parse import parse_method_base, parse_method_ours, parser_add_main_args

# NOTE: for consistent data splits, see data_utils.rand_train_test_idx
def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
fix_seed(0)

def parse_args(parser, args=None, namespace=None):
    args, argv = parser.parse_known_args(args, namespace)
    return args

parser = argparse.ArgumentParser(description='General Training Pipeline')
parser_add_main_args(parser)
# args = parser.parse_args()
args = parse_args(parser)
print(args)


# ### Parse args ###
# parser = argparse.ArgumentParser(description='General Training Pipeline')
# parser_add_main_args(parser)
# args = parser.parse_args()
# print(args)

device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

args.dataset = 'amazon-photo'
def get_dataset(dataset, sub_dataset=None, gen_model=None):
    ### Load and preprocess data ###
    args.data_dir = "GraphOOD-EERM/data"
    if dataset == 'cora':
        dataset = load_nc_dataset(args.data_dir, 'cora', sub_dataset, gen_model)
    elif dataset == 'amazon-photo':
        dataset = load_nc_dataset(args.data_dir, 'amazon-photo', sub_dataset, gen_model)
    else:
        raise ValueError('Invalid dataname')

    if len(dataset.label.shape) == 1:
        dataset.label = dataset.label.unsqueeze(1)

    dataset.n = dataset.graph['num_nodes']
    dataset.c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
    dataset.d = dataset.graph['node_feat'].shape[1]

    dataset.graph['edge_index'], dataset.graph['node_feat'] = \
        dataset.graph['edge_index'], dataset.graph['node_feat']
    return dataset

if args.dataset == 'cora':
    tr_sub, val_sub, te_subs = [0], [1], list(range(2, 10))
    gen_model = args.gnn_gen
    dataset_tr = get_dataset(dataset='cora', sub_dataset=tr_sub[0], gen_model=gen_model)
    dataset_val = get_dataset(dataset='cora', sub_dataset=val_sub[0], gen_model=gen_model)
    datasets_te = [get_dataset(dataset='cora', sub_dataset=te_subs[i], gen_model=gen_model) for i in range(len(te_subs))]
elif args.dataset == 'amazon-photo':
    tr_sub, val_sub, te_subs = [0], [1], list(range(2, 10))
    gen_model = args.gnn_gen
    dataset_tr = get_dataset(dataset='amazon-photo', sub_dataset=tr_sub[0], gen_model=gen_model)
    dataset_val = get_dataset(dataset='amazon-photo', sub_dataset=val_sub[0], gen_model=gen_model)
    datasets_te = [get_dataset(dataset='amazon-photo', sub_dataset=te_subs[i], gen_model=gen_model) for i in range(len(te_subs))]
else:
    raise ValueError('Invalid dataname')

print(f"Train num nodes {dataset_tr.n} | num classes {dataset_tr.c} | num node feats {dataset_tr.d}")
print(f"Val num nodes {dataset_val.n} | num classes {dataset_val.c} | num node feats {dataset_val.d}")
for i in range(len(te_subs)):
    dataset_te = datasets_te[i]
    print(f"Test {i} num nodes {dataset_te.n} | num classes {dataset_te.c} | num node feats {dataset_te.d}")

