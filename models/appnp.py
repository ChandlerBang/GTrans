import torch.nn as nn
import torch.nn.functional as F
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_geometric.nn import APPNP as APPNPConv
from torch.nn import Linear
from .base_model import BaseModel
from torch_sparse import SparseTensor, matmul


class APPNP(BaseModel):

    def __init__(self, nfeat, nhid, nclass, nlayers=10, alpha=0.1, dropout=0.5, lr=0.01,
                with_bn=False, weight_decay=5e-4, with_bias=True, device=None, args=None):

        super(APPNP, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device
        from utils import eval_acc, eval_f1, eval_rocauc
        if args.dataset == 'twitch-e':
            self.eval_func = eval_rocauc
        elif args.dataset == 'elliptic':
            self.eval_func = eval_f1
        elif args.dataset in ['cora', 'amazon-photo', 'ogb-arxiv', 'fb100']:
            self.eval_func = eval_acc
        else:
            raise NotImplementedError


        self.lin1 = Linear(nfeat, nhid)
        if with_bn:
            self.bn1 = nn.BatchNorm1d(nhid)
            self.bn2 = nn.BatchNorm1d(nclass)

        self.lin2 = Linear(nhid, nclass)
        K = nlayers
        self.prop1 = APPNPConv(K, alpha)

        self.dropout = dropout
        self.weight_decay = weight_decay
        self.lr = lr
        self.output = None
        self.best_model = None
        self.best_output = None
        self.name = 'APPNP'
        self.with_bn = with_bn
        self.args = args

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin1(x)
        if self.with_bn:
            x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        if self.with_bn:
            x = self.bn2(x)
        if edge_weight is not None:
            adj = SparseTensor.from_edge_index(edge_index, edge_weight, sparse_sizes=2 * x.shape[:1])
            x = self.prop1(x, adj)
        else:
            x = self.prop1(x, edge_index, edge_weight)
        # return F.log_softmax(x, dim=1)
        return x

    def get_embed(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin1(x)
        if self.with_bn:
            x = self.bn1(x)
        x = F.relu(x)
        # x = F.dropout(x, p=self.dropout, training=self.training)
        # x = self.lin2(x)
        # if self.with_bn:
        #     x = self.bn2(x)
        if edge_weight is not None:
            adj = SparseTensor.from_edge_index(edge_index, edge_weight, sparse_sizes=2 * x.shape[:1])
            x = self.prop1(x, adj)
        else:
            x = self.prop1(x, edge_index, edge_weight)
        # return F.log_softmax(x, dim=1)
        return x

    def initialize(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        if self.with_bn:
            self.bn1.reset_parameters()
            self.bn2.reset_parameters()


if __name__ == "__main__":
    from deeprobust.graph.data import Dataset, Dpr2Pyg
    data = Dataset(root='/tmp/', name='cora', setting='gcn')
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    model = GCN(nfeat=features.shape[1],
          nhid=16,
          nclass=labels.max().item() + 1,
          dropout=0.5, device='cuda')
    model = model.to('cuda')
    pyg_data = Dpr2Pyg(data)[0]

    import ipdb
    ipdb.set_trace()

    model.fit(pyg_data, verbose=True) # train with earlystopping
    model.test()
    print(model.predict())
