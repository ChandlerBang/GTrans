import torch
import os.path as osp
from torch_geometric.datasets import Planetoid, PPI, WikiCS, Coauthor, Amazon, CoraFull
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from deeprobust.graph.data import Dataset, PrePtbDataset
import scipy.sparse as sp
import numpy as np
from deeprobust.graph.data import Dataset
from deeprobust.graph.global_attack import NodeEmbeddingAttack
from deeprobust.graph import utils
from deeprobust.graph.utils import get_train_val_test_gcn, get_train_val_test
from torch_geometric.utils import train_test_split_edges
from torch_geometric.utils import add_remaining_self_loops, to_undirected
from ogb.nodeproppred import PygNodePropPredDataset
from sklearn.model_selection import train_test_split
from deeprobust.graph.data.pyg_dataset import Dpr2Pyg
from torch_geometric.utils import subgraph
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score
import subprocess


def get_dataset(name, normalize_features=False, transform=None):
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', name)
    if name in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(path, name)
    elif name in ['arxiv']:
        dataset = PygNodePropPredDataset(name='ogbn-'+name)
    else:
        raise NotImplementedError

    if transform is not None and normalize_features:
        dataset.transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        dataset.transform = T.NormalizeFeatures()
    elif transform is not None:
        dataset.transform = transform

    return to_inductive(dataset)

def mask_to_index(index, size):
    all_idx = np.arange(size)
    return all_idx[index]

def index_to_mask(index, size):
    mask = torch.zeros((size, ), dtype=torch.bool)
    mask[index] = 1
    return mask

def resplit(data):
    n = data.x.shape[0]
    idx = np.arange(n)
    idx_train, idx_val, idx_test = get_train_val_test(nnodes=n, val_size=0.2, test_size=0.2, stratify=data.y)

    data.train_mask = index_to_mask(idx_train, n)
    data.val_mask = index_to_mask(idx_val, n)
    data.test_mask = index_to_mask(idx_test, n)


def add_mask(data, dataset):
    # for arxiv
    split_idx = dataset.get_idx_split()
    train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
    n = data.x.shape[0]
    data.train_mask = index_to_mask(train_idx, n)
    data.val_mask = index_to_mask(valid_idx, n)
    data.test_mask = index_to_mask(test_idx, n)
    data.y = data.y.squeeze()
    data.edge_index = to_undirected(data.edge_index, data.num_nodes)

def holdout_val(data):
    """hold out a seperate validation from the original validation"""
    n = data.x.shape[0]
    idx = np.arange(n)
    idx_val = idx[data.val_mask]

    val1, val2 = train_test_split(idx_val, random_state=None,
                           train_size=0.8, test_size=0.2, stratify=data.y[idx_val])

    data.val1_mask = index_to_mask(val1, n)
    data.val2_mask = index_to_mask(val2, n)


def to_inductive(dataset):
    data = dataset[0]
    add_mask(data, dataset)

    def sub_to_inductive(data, mask):
        new_data = Graph()
        new_data.graph['edge_index'], _ = subgraph(mask, data.edge_index, None,
                              relabel_nodes=True, num_nodes=data.num_nodes)
        new_data.graph['num_nodes'] = mask.sum().item()
        new_data.graph['node_feat'] = data.x[mask]
        new_data.label = data.y[mask].unsqueeze(1)
        return new_data
    train_graph = sub_to_inductive(data, data.train_mask)
    val_graph = sub_to_inductive(data, data.val_mask)
    test_graph = sub_to_inductive(data, data.test_mask)
    val_graph.test_mask = torch.tensor(np.ones(val_graph.graph['num_nodes'])).bool()
    test_graph.test_mask = torch.tensor(np.ones(test_graph.graph['num_nodes'])).bool()
    return [train_graph, val_graph, [test_graph]]

class Graph:

    def __init__(self):
        self.test_mask = None
        self.label = None
        self.graph = {'edge_index': None, 'node_feat': None, 'num_nodes': None}

@torch.no_grad()
def eval_acc(y_true, y_pred):
    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()
    return (y_true == y_pred).sum() / y_true.shape[0]


@torch.no_grad()
def eval_rocauc(y_true, y_pred):
    """ adapted from ogb
    https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/evaluate.py"""
    rocauc_list = []
    y_true = y_true.detach().cpu().numpy()
    if y_true.shape[1] == 1:
        # use the predicted class for single-class classification
        y_pred = F.softmax(y_pred, dim=-1)[:,1].unsqueeze(1).cpu().numpy()
    else:
        y_pred = y_pred.detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            is_labeled = y_true[:, i] == y_true[:, i]
            score = roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i])

            rocauc_list.append(score)

    if len(rocauc_list) == 0:
        raise RuntimeError(
            'No positively labeled data available. Cannot compute ROC-AUC.')

    return sum(rocauc_list)/len(rocauc_list)

@torch.no_grad()
def eval_f1(y_true, y_pred):
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()
    f1 = f1_score(y_true, y_pred, average='macro')
    # macro_f1 = f1_score(y_true, y_pred, average='macro')
    return f1



def reset_args(args):
    args.weight_decay = 1e-3
    args.dropout = 0
    if args.dataset in ['cora', 'amazon-photo']:
        args.lr = 0.001
        args.nlayers = 2
        args.hidden = 32

    elif args.dataset == 'ogb-arxiv':
        if args.ood:
            args.lr = 0.01
            args.nlayers=5
            args.hidden = 32
            args.weight_decay = 0
        else:
            args.lr = 0.01
            args.dropout=0.5
            args.nlayers = 3
            args.hidden = 256
            args.weight_decay = 0
    elif args.dataset == 'fb100':
        args.lr = 0.01
        args.nlayers = 2
        args.hidden = 32
    elif args.dataset == 'twitch-e':
        args.lr = 0.01
        args.nlayers = 2
        args.hidden = 32
    elif args.dataset in ['elliptic']:
        args.lr = 0.01
        args.nlayers = 5
        args.hidden = 32
        args.weight_decay = 0
    else:
        raise NotImplementedError

    if args.tune == 0:
        import pandas as pd
        filename = 'models/params.csv'
        df = pd.read_csv(filename, delimiter=',')
        df2 = df[(df.dataset == args.dataset) & (df.model == args.model)]
        params = df2[['lr_feat', 'lr_adj', 'epoch', 'ratio']].values
        if len(params) == 1:
            args.lr_feat, args.lr_adj, args.epochs, args.ratio = params[0]
            args.epochs = int(args.epochs)

def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map
