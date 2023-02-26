"""
use test data to calculate the loss in SRGNN
"""
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from deeprobust.graph import utils
import torch.nn as nn
import torch
from torch_geometric.utils import dropout_adj


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()


    def fit_inductive(self, data, train_iters=1000, initialize=True, verbose=False, patience=100, **kwargs):
        if initialize:
            self.initialize()

        self.train_data = data[0]
        self.val_data = data[1]
        self.test_data = data[2]
        # By default, it is trained with early stopping on validation
        self.train_with_early_stopping(train_iters, patience, verbose)

    # def fit_with_val1_val2(self, pyg_data, train_iters=1000, initialize=True, verbose=False, **kwargs):
    def fit_with_val(self, pyg_data, train_iters=1000, initialize=True, patience=100, verbose=False, **kwargs):
        if initialize:
            self.initialize()

        self.data = pyg_data.to(self.device)
        self.data.train_mask = self.data.train_mask + self.data.val1_mask
        self.data.val_mask = self.data.val2_mask
        self.train_with_early_stopping(train_iters, patience, verbose)

    def train_with_early_stopping(self, train_iters, patience, verbose):
        """early stopping based on the validation loss
        """
        if verbose:
            print(f'=== training {self.name} model ===')
        # optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        train_data, val_data = self.train_data, self.val_data

        early_stopping = patience
        # best_loss_val = 100
        best_acc_val = float('-inf')

        if type(train_data) is not list:
            x, y = train_data.graph['node_feat'].to(self.device), train_data.label.to(self.device)#.squeeze()
            edge_index = train_data.graph['edge_index'].to(self.device)

            x_val, y_val = val_data.graph['node_feat'].to(self.device), val_data.label.to(self.device)#.squeeze()
            edge_index_val = val_data.graph['edge_index'].to(self.device)

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            if type(train_data) is not list:
                if hasattr(self, 'dropedge') and self.dropedge != 0:
                    edge_index, _ = dropout_adj(edge_index, p=self.dropedge)

                output = self.forward(x, edge_index)
                if self.args.dataset == 'elliptic':
                    loss_train = self.sup_loss(y[train_data.mask], output[train_data.mask])
                else:
                    loss_train = self.sup_loss(y, output)
            else:
                loss_train = 0
                for graph_id, dat in enumerate(train_data):
                    x, y = dat.graph['node_feat'].to(self.device), dat.label.to(self.device)#.squeeze()
                    edge_index = dat.graph['edge_index'].to(self.device)
                    if hasattr(self, 'dropedge') and self.dropedge != 0:
                        edge_index, _ = dropout_adj(edge_index, p=self.dropedge)
                    output = self.forward(x, edge_index)
                    if self.args.dataset == 'elliptic':
                        loss_train += self.sup_loss(y[dat.mask], output[dat.mask])
                    else:
                        loss_train += self.sup_loss(y, output)

                loss_train = loss_train / len(train_data)
            loss_train.backward()
            optimizer.step()

            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            self.eval()
            eval_func = self.eval_func
            if self.args.dataset in ['ogb-arxiv']:
                output = self.forward(x_val, edge_index_val)
                acc_val = eval_func(y_val[val_data.test_mask], output[val_data.test_mask])
            elif self.args.dataset in ['cora', 'amazon-photo', 'twitch-e']:
                output = self.forward(x_val, edge_index_val)
                acc_val = eval_func(y_val, output)
            elif self.args.dataset in ['fb100']:
                y_val, out_val = [], []
                for i, dataset in enumerate(val_data):
                    x_val = dataset.graph['node_feat'].to(self.device)
                    edge_index_val = dataset.graph['edge_index'].to(self.device)
                    out = self.forward(x_val, edge_index_val)
                    y_val.append(dataset.label.to(self.device))
                    out_val.append(out)
                acc_val = eval_func(torch.cat(y_val, dim=0), torch.cat(out_val, dim=0))
            elif self.args.dataset in ['elliptic']:
                # acc_val = eval_func(y_val, output)
                y_val, out_val = [], []
                for i, dataset in enumerate(val_data):
                    x_val = dataset.graph['node_feat'].to(self.device)
                    edge_index_val = dataset.graph['edge_index'].to(self.device)
                    out = self.forward(x_val, edge_index_val)
                    y_val.append(dataset.label[dataset.mask].to(self.device))
                    out_val.append(out[dataset.mask])
                acc_val = eval_func(torch.cat(y_val, dim=0), torch.cat(out_val, dim=0))
            else:
                raise NotImplementedError

            if best_acc_val < acc_val:
                best_acc_val = acc_val
                self.output = output
                weights = deepcopy(self.state_dict())
                patience = early_stopping
            else:
                patience -= 1
            if i > early_stopping and patience <= 0:
                break

        if verbose:
             # print('=== early stopping at {0}, loss_val = {1} ==='.format(i, best_loss_val) )
             print('=== early stopping at {0}, acc_val = {1} ==='.format(i, best_acc_val) )
        self.load_state_dict(weights)

    def sup_loss(self, y, pred):
        if self.args.dataset in ('twitch-e', 'fb100', 'elliptic'):
            if y.shape[1] == 1:
                true_label = F.one_hot(y, y.max() + 1).squeeze(1)
            else:
                true_label = y
            criterion = nn.BCEWithLogitsLoss()
            loss = criterion(pred, true_label.squeeze(1).to(torch.float))
        else:
            out = F.log_softmax(pred, dim=1)
            target = y.squeeze(1)
            criterion = nn.NLLLoss()
            loss = criterion(out, target)
        return loss

    def get_pred(self, logits):
        if self.args.dataset in ('twitch-e', 'fb100', 'elliptic'):
            pred = torch.sigmoid(logits)
        else:
            pred = F.softmax(logits, dim=1)
        return pred

    def test(self):
        """Evaluate model performance on test set.
        Parameters
        ----------
        idx_test :
            node testing indices
        """
        self.eval()
        test_mask = self.data.test_mask
        labels = self.data.y
        output = self.forward(self.data.x, self.data.edge_index)
        # output = self.output
        loss_test = F.nll_loss(output[test_mask], labels[test_mask])
        acc_test = utils.accuracy(output[test_mask], labels[test_mask])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test.item()

    @torch.no_grad()
    def predict(self, x=None, edge_index=None, edge_weight=None):
        """
        Returns
        -------
        torch.FloatTensor
            output (log probabilities)
        """
        self.eval()
        if x is None or edge_index is None:
            x, edge_index = self.test_data.graph['node_feat'], self.test_data.graph['edge_index']
            x, edge_index = x.to(self.device), edge_index.to(self.device)
        return self.forward(x, edge_index, edge_weight)

    def _ensure_contiguousness(self,
                               x,
                               edge_idx,
                               edge_weight):
        if not x.is_sparse:
            x = x.contiguous()
        if hasattr(edge_idx, 'contiguous'):
            edge_idx = edge_idx.contiguous()
        if edge_weight is not None:
            edge_weight = edge_weight.contiguous()
        return x, edge_idx, edge_weight



