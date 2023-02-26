import torch.nn as nn
import torch.nn.functional as F
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_geometric.nn import GCNConv
from .base_model import BaseModel
from torch_sparse import coalesce, SparseTensor


class GCNDAE(BaseModel):

    def __init__(self, nfeat, nhid, nclass, nlayers=2, dropout=0.5, lr=0.01, save_mem=True,
                with_bn=False, weight_decay=5e-4, with_bias=True, device=None, args=None):

        super(GCN, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.args = args
        from utils import eval_acc, eval_f1, eval_rocauc
        if args.dataset == 'twitch-e':
            self.eval_func = eval_rocauc
        elif args.dataset == 'elliptic':
            self.eval_func = eval_f1
        elif args.dataset in ['cora', 'amazon-photo', 'ogb-arxiv', 'fb100']:
            self.eval_func = eval_acc
        else:
            raise NotImplementedError

        self.layers = nn.ModuleList([])
        if with_bn:
            self.bns = nn.ModuleList()

        if nlayers == 1:
            self.layers.append(GCNConv(nfeat, nclass, bias=with_bias, normalize=not save_mem))
        else:
            self.layers.append(GCNConv(nfeat, nhid, bias=with_bias, normalize=not save_mem))
            if with_bn:
                self.bns.append(nn.BatchNorm1d(nhid))
            for i in range(nlayers-2):
                self.layers.append(GCNConv(nhid, nhid, bias=with_bias, normalize=not save_mem))
                if with_bn:
                    self.bns.append(nn.BatchNorm1d(nhid))
            self.layers.append(GCNConv(nhid, nclass, bias=with_bias, normalize=not save_mem))

        self.dropout = dropout
        self.weight_decay = weight_decay
        self.lr = lr
        self.output = None
        self.best_model = None
        self.best_output = None
        self.with_bn = with_bn
        self.name = 'GCN'

    def forward(self, x, edge_index, edge_weight=None):
        x, edge_index, edge_weight = self._ensure_contiguousness(x, edge_index, edge_weight)
        if edge_weight is not None:
            adj = SparseTensor.from_edge_index(edge_index, edge_weight, sparse_sizes=2 * x.shape[:1]).t()

        for ii, layer in enumerate(self.layers):
            if edge_weight is not None:
                x = layer(x, adj)
            else:
                # x = layer(x, edge_index, edge_weight=edge_weight)
                x = layer(x, edge_index)
            if ii != len(self.layers) - 1:
                if self.with_bn:
                    x = self.bns[ii](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x
        # return F.log_softmax(x, dim=1)

    def setup_dae(self, nfeat, nhid, nclass):
        self.dae_layers = nn.ModuleList([])
        self.dae_layers.append(GCNConv(nfeat, nhid))
        self.dae_layers.append(GCNConv(nhid, nclass))
        return

    def train_dae(self, x, edge_index):
        optimizer = optim.Adam(self.dae_layers.parameters(), lr=0.01, weight_decay=0)
        for epoch in range(1, args.epochs + 1):
            optimizer.zero_grad()
            loss = self.get_loss_masked_features(x, edge_index)
            loss.backward()
            optimizer.step()
            print("Epoch {:05d} | Train Loss {:.4f}".format(epoch, loss.item()))
        return

    def get_loss_masked_features(self, features, edge_index):
        ratio = 10; nr = 5
        noise = 'mask'
        mask = get_random_mask(features, ratio, nr).cuda()
        if noise == 'mask':
            masked_features = features * (1 - mask)
        elif noise == "normal":
            noise = torch.normal(0.0, 1.0, size=features.shape).cuda()
            masked_features = features + (noise * mask)

        for layer in self.dae_layers[:-1]:
            x = layer(x, edge_index)
            x = F.relu(x)
            # x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x, edge_index)
        logits = model(features, masked_features)
        indices = mask > 0
        loss = F.mse_loss(logits[indices], features[indices], reduction='mean')
        return loss


    def get_embed(self, x, edge_index, edge_weight=None):
        x, edge_index, edge_weight = self._ensure_contiguousness(x, edge_index, edge_weight)
        for ii, layer in enumerate(self.layers):
            if ii == len(self.layers) - 1:
                return x
            if edge_weight is not None:
                adj = SparseTensor.from_edge_index(edge_index, edge_weight,
                        sparse_sizes=2 * x.shape[:1]).t() # in case it is directed...

                # layer(x, edge_index, edge_weight)
                x = layer(x, adj)
            else:
                x = layer(x, edge_index)
            if ii != len(self.layers) - 1:
                if self.with_bn:
                    x = self.bns[ii](x)
                x = F.relu(x)
                # x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def initialize(self):
        for m in self.layers:
            m.reset_parameters()
        if self.with_bn:
            for bn in self.bns:
                bn.reset_parameters()

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
