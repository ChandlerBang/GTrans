import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
# from torch_geometric.nn import SAGEConv, GATConv, APPNP, MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import scipy.sparse
import numpy as np
from .base_model import BaseModel


class SAGE(BaseModel):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0.5, lr=0.01, weight_decay=0, device='cpu', with_bn=True, args=None):
        super(SAGE, self).__init__()
        from utils import eval_acc, eval_f1, eval_rocauc
        if args.dataset == 'twitch-e':
            self.eval_func = eval_rocauc
        elif args.dataset == 'elliptic':
            self.eval_func = eval_f1
        elif args.dataset in ['cora', 'amazon-photo', 'ogb-arxiv', 'fb100']:
            self.eval_func = eval_acc
        else:
            raise NotImplementedError

        self.args = args
        self.convs = nn.ModuleList()
        self.convs.append(
            SAGEConv(in_channels, hidden_channels))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.convs.append(
            SAGEConv(hidden_channels, out_channels))

        self.weight_decay = weight_decay
        self.lr = lr
        self.dropout = dropout
        self.activation = F.relu
        self.with_bn = with_bn
        self.device = device
        self.name = "SAGE2"

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



from typing import Union, Tuple
from torch_geometric.typing import OptPairTensor, Adj, Size

from torch import Tensor
from torch.nn import Linear
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing


class SAGEConv(MessagePassing):
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, normalize: bool = False,
                 bias: bool = True, **kwargs):  # yapf: disable
        kwargs.setdefault('aggr', 'mean')
        super(SAGEConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_l = Linear(in_channels[0], out_channels, bias=bias)
        self.lin_r = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None) -> Tensor:
        """"""
        if 0:
            if isinstance(x, Tensor):
                x: OptPairTensor = (x, x)
            # propagate_type: (x: OptPairTensor)
            out = self.propagate(edge_index, x=x, size=size)
            out = self.lin_l(out)
        else: # for  fb100 dataset
            if isinstance(x, Tensor):
                x: OptPairTensor = (x, x)
            out = self.lin_l(x[0])
            # propagate_type: (x: OptPairTensor)
            out = self.propagate(edge_index, x=(out, out), size=size)

        x_r = x[1]
        if x_r is not None:
            out += self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        # Deleted the following line to make propagation differentiable
        # adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
