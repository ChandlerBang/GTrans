import torch.nn as nn
import torch.nn.functional as F
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_geometric.nn import GATConv
from .base_model import BaseModel
from torch_sparse import coalesce, SparseTensor


class GAT(BaseModel):

    def __init__(self, nfeat, nhid, nclass, heads=8, output_heads=1, dropout=0., lr=0.01,
            nlayers=2, with_bn=False, weight_decay=5e-4, device=None, args=None):

        super(GAT, self).__init__()
        from utils import eval_acc, eval_f1, eval_rocauc
        if args.dataset == 'twitch-e':
            self.eval_func = eval_rocauc
        elif args.dataset == 'elliptic':
            self.eval_func = eval_f1
        elif args.dataset in ['cora', 'amazon-photo', 'ogb-arxiv', 'fb100']:
            self.eval_func = eval_acc
        else:
            raise NotImplementedError

        assert device is not None, "Please specify 'device'!"
        self.device = device

        if with_bn:
            self.bns = nn.ModuleList()
            self.bns.append(nn.BatchNorm1d(nhid*heads))

        self.convs = nn.ModuleList()
        self.convs.append(GATConv(
                nfeat,
                nhid,
                heads=heads,
                dropout=dropout))

        for _ in range(nlayers - 2):
            self.convs.append(GATConv(
                    nhid * heads,
                    nhid,
                    heads=heads,
                    dropout=dropout))
            if with_bn:
                self.bns.append(nn.BatchNorm1d(nhid*heads))

        self.convs.append(GATConv(
            nhid * heads,
            nclass,
            heads=output_heads,
            concat=False,
            dropout=dropout))

        self.dropout = dropout
        self.weight_decay = weight_decay
        self.lr = lr
        self.output = None
        self.best_model = None
        self.best_output = None
        self.activation = F.elu
        self.name = 'GAT'
        self.args = args
        self.with_bn = with_bn

    def forward(self, x, edge_index, edge_weight=None):
        if edge_weight is not None:
            adj = SparseTensor.from_edge_index(edge_index, edge_weight, sparse_sizes=2 * x.shape[:1]).t()

        for i, conv in enumerate(self.convs[:-1]):
            if edge_weight is not None:
                x = conv(x, adj)
            else:
                x = conv(x, edge_index, edge_weight)
            if self.with_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        if edge_weight is not None:
            x = self.convs[-1](x, adj)
        else:
            x = self.convs[-1](x, edge_index, edge_weight)
        # return F.log_softmax(x, dim=1)
        return x

    def initialize(self):
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def get_embed(self, x, edge_index, edge_weight=None):
        x, edge_index, edge_weight = self._ensure_contiguousness(x, edge_index, edge_weight)
        for ii, layer in enumerate(self.convs):
            if ii == len(self.convs) - 1:
                return x
            if edge_weight is not None:
                adj = SparseTensor.from_edge_index(edge_index, edge_weight, sparse_sizes=2 * x.shape[:1]).t()
                x = layer(x, adj)
            else:
                x = layer(x, edge_index)
            if ii != len(self.convs) - 1:
                if self.with_bn:
                    x = self.bns[ii](x)
                x = F.relu(x)
                # x = F.dropout(x, p=self.dropout, training=self.training)
        return x


if __name__ == "__main__":
    from deeprobust.graph.data import Dataset, Dpr2Pyg
    # from deeprobust.graph.defense import GAT
    data = Dataset(root='/tmp/', name='cora')
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    gat = GAT(nfeat=features.shape[1],
          nhid=8, heads=8,
          nclass=labels.max().item() + 1,
          dropout=0.5, device='cpu')
    gat = gat.to('cpu')
    pyg_data = Dpr2Pyg(data)
    gat.fit(pyg_data, verbose=True) # train with earlystopping
    gat.test()
    print(gat.predict())
