import argparse
import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected
from torch_scatter import scatter

from logger import Logger, SimpleLogger
from dataset import load_nc_dataset
from data_utils import normalize, gen_normalized_adjs, evaluate, evaluate_whole_graph, evaluate_whole_graph_multi, eval_acc, eval_rocauc, eval_f1, to_sparse_tensor, load_fixed_splits
from parse import parse_method_base, parse_method_ours, parse_method_ours_multi, parser_add_main_args

import warnings
warnings.filterwarnings("ignore")

# NOTE: for consistent data splits, see data_utils.rand_train_test_idx
def fix_seed(seed):
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
args.dataset = 'fb100'
print(args)

# ### Parse args ###
# parser = argparse.ArgumentParser(description='General Training Pipeline')
# parser_add_main_args(parser)
# args = parser.parse_args()
# print(args)

device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

def get_dataset(dataset, sub_dataset=None):
    ### Load and preprocess data ###
    args.data_dir = "GraphOOD-EERM/data"
    if dataset == 'twitch-e':
        dataset = load_nc_dataset(args.data_dir, 'twitch-e', sub_dataset)
    elif dataset == 'fb100':
        dataset = load_nc_dataset(args.data_dir, 'fb100', sub_dataset)
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

if args.dataset == 'twitch-e':
    twitch_sub_name = ['DE', 'ENGB', 'ES', 'FR', 'PTBR', 'RU', 'TW']
    tr_sub, val_sub, te_subs = ['DE'], ['ENGB'], ['ES', 'FR', 'PTBR', 'RU', 'TW']
    dataset_tr = get_dataset(dataset='twitch-e', sub_dataset=tr_sub[0])
    dataset_val = get_dataset(dataset='twitch-e', sub_dataset=val_sub[0])
    datasets_te = [get_dataset(dataset='twitch-e', sub_dataset=te_subs[i]) for i in range(len(te_subs))]
elif args.dataset == 'fb100':
    '''
    Configure different training sub-graphs
    '''
    tr_subs, val_subs, te_subs = ['Johns Hopkins55', 'Caltech36', 'Amherst41'], ['Cornell5', 'Yale4'],  ['Penn94', 'Brown11', 'Texas80']
    # tr_subs, val_subs, te_subs = ['Bingham82', 'Duke14', 'Princeton12'], ['Cornell5', 'Yale4'],  ['Penn94', 'Brown11', 'Texas80']
    # tr_subs, val_subs, te_subs = ['WashU32', 'Brandeis99', 'Carnegie49'], ['Cornell5', 'Yale4'], ['Penn94', 'Brown11', 'Texas80']
    datasets_tr = [get_dataset(dataset='fb100', sub_dataset=tr_subs[i]) for i in range(len(tr_subs))]
    datasets_val = [get_dataset(dataset='fb100', sub_dataset=val_subs[i]) for i in range(len(val_subs))]
    datasets_te = [get_dataset(dataset='fb100', sub_dataset=te_subs[i]) for i in range(len(te_subs))]
else:
    raise ValueError('Invalid dataname')

if args.dataset == 'fb100':
    dataset_tr = datasets_tr[0]
    dataset_val = datasets_val[0]
print(f"Train num nodes {dataset_tr.n} | num classes {dataset_tr.c} | num node feats {dataset_tr.d}")
print(f"Val num nodes {dataset_val.n} | num classes {dataset_val.c} | num node feats {dataset_val.d}")
for i in range(len(te_subs)):
    dataset_te = datasets_te[i]
    print(f"Test {i} num nodes {dataset_te.n} | num classes {dataset_te.c} | num node feats {dataset_te.d}")

