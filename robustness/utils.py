import torch
import os.path as osp
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from deeprobust.graph.data import Dataset
import scipy.sparse as sp
import numpy as np
from deeprobust.graph.data import Dataset
from deeprobust.graph.global_attack import NodeEmbeddingAttack
from deeprobust.graph import utils
from deeprobust.graph.utils import get_train_val_test
from torch_geometric.utils import add_remaining_self_loops, to_undirected
from ogb.nodeproppred import PygNodePropPredDataset
from sklearn.model_selection import train_test_split
from deeprobust.graph.data.pyg_dataset import Dpr2Pyg
import torch.nn.functional as F


def get_dataset(name, normalize_features=True, transform=None, if_dpr=True):
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
    if not if_dpr:
        return dataset
    else:
        return Pyg2Dpr(dataset)

class Pyg2Dpr(Dataset):
    def __init__(self, pyg_data, multi_splits=False, **kwargs):
        try:
            splits = pyg_data.get_idx_split()
        except:
            pass

        dataset_name = pyg_data.name

        pyg_data = pyg_data[0]
        n = pyg_data.num_nodes

        if dataset_name == 'ogbn-arxiv': # symmetrization
            pyg_data.edge_index = to_undirected(pyg_data.edge_index, pyg_data.num_nodes)

        self.adj = sp.csr_matrix((np.ones(pyg_data.edge_index.shape[1]),
            (pyg_data.edge_index[0], pyg_data.edge_index[1])), shape=(n, n))
        self.features = pyg_data.x.numpy()
        self.labels = pyg_data.y.numpy()

        if len(self.labels.shape) == 2 and self.labels.shape[1] == 1:
            self.labels = self.labels.reshape(-1) # ogb-arxiv needs to reshape
        if not multi_splits:
            if hasattr(pyg_data, 'train_mask'):
                self.idx_train = mask_to_index(pyg_data.train_mask, n)
                self.idx_val = mask_to_index(pyg_data.val_mask, n)
                self.idx_test = mask_to_index(pyg_data.test_mask, n)
                self.name = 'Pyg2Dpr'
            else:
                try:
                    # for ogb
                    self.idx_train = splits['train']
                    self.idx_val = splits['valid']
                    self.idx_test = splits['test']
                    self.name = 'Pyg2Dpr'
                except:
                    # for other datasets
                    self.idx_train, self.idx_val, self.idx_test = get_train_val_test(
                            nnodes=n, val_size=0.1, test_size=0.8, stratify=self.labels)

    def load_splits(self, data):
        splits = {'train': [], 'val': [], 'test': []}
        n = data.num_nodes
        for i in range(0, data.train_mask.shape[1]):
            train_mask = data.train_mask[:, i]
            val_mask = data.val_mask[:, i]
            if len(data.test_mask.shape) == 1:
                test_mask = data.test_mask
            else:
                test_mask = data.test_mask[:, i]
            idx_train = mask_to_index(train_mask, n)
            idx_val = mask_to_index(val_mask, n)
            idx_test = mask_to_index(test_mask, n)

            splits['train'].append(idx_train)
            splits['val'].append(idx_val)
            splits['test'].append(idx_test)
        return splits

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

def get_perf(output, labels, mask, verbose=True):
    loss = F.nll_loss(output[mask], labels[mask])
    acc = utils.accuracy(output[mask], labels[mask])
    if verbose:
        print("loss= {:.4f}".format(loss.item()),
              "accuracy= {:.4f}".format(acc.item()))
    return loss.item(), acc.item()


def reset_args(args):
    if args.debug != 2:
        if args.dataset in ['pubmed']:
            args.lr_feat = 1; args.epochs=10; args.loss = 'train+1e-4LC'
        if args.dataset in ['cora']:
            args.lr_feat = 0.1; args.epochs=10; args.loss = 'train+1e-4LC'
        if args.dataset in ['arxiv']:
            args.lr_feat = 0.01; args.epochs=20; args.loss = 'train+0.01LC'

