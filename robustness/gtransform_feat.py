import numpy as np
from deeprobust.graph.defense_pyg import AirGNN, GCN, APPNP, GAT, SAGE, GPRGNN
import torch.nn.functional as F
import torch
import deeprobust.graph.utils as utils
from torch.nn.parameter import Parameter
from tqdm import tqdm
import scipy.sparse as sp
import pandas as pd
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from torch_geometric.utils import to_scipy_sparse_matrix, from_scipy_sparse_matrix, dropout_adj
from copy import deepcopy

class FeatAgent:

    def __init__(self, data, args, model=None):
        self.device = 'cuda'
        self.args = args
        self.data = data
        print(f'===Get {args.model}')
        if model is None:
            self.model = self.pretrain_model()
        else:
            self.model = model

        if args.noise_feature > 0:
            self.clean_feat = data.x.clone()
            self.feat_noise, self.noisy_nodes = add_feature_noise_test(data, args.noise_feature)
        if args.noise_structure > 0:
            add_structure_noise(data, args.noise_structure)

        output = self.model.predict()
        labels = data.y.to(self.device)
        print("Test set results:")
        self.get_perf(output, labels, data.test_mask)

        nnodes = data.x.shape[0]
        d = data.x.shape[1]

        self.delta_feat = Parameter(torch.FloatTensor(nnodes, d).to(self.device))
        self.delta_feat.data.fill_(0)
        self.optimizer_feat = torch.optim.Adam([self.delta_feat], lr=args.lr_feat)

    def initialize_as_ori_feat(self, feat):
        self.delta_feat.data.copy_(feat)

    def learn_graph(self, edge_index_=None, edge_weight=None):
        print(f'Learn graph based on {self.args.model}:')
        data = self.data
        args = self.args
        delta_feat = self.delta_feat
        model = self.model
        model.eval() # should set to eval

        edge_index, feat, labels = data.edge_index, data.x, data.y
        if edge_index_ is not None:
            edge_index = edge_index_

        self.edge_index = edge_index

        best_loss_val = 10000
        best_acc_val = 0
        for it in tqdm(range(args.epochs)):
            self.optimizer_feat.zero_grad()
            new_feat = feat+delta_feat
            loss = self.test_time_loss(model, new_feat, edge_index, edge_weight)
            loss.backward()
            if it % 100 == 0:
                print(f'Epoch {it}: {loss}')

            self.optimizer_feat.step()
            with torch.no_grad():
                new_feat = feat+delta_feat
                output = model.forward(feat+delta_feat, edge_index, edge_weight)
            val_mask = self.data.val_mask
            loss_val = F.nll_loss(output[val_mask], labels[val_mask])
            acc_val = utils.accuracy(output[val_mask], labels[val_mask])
            acc_train = utils.accuracy(output[data.train_mask], labels[data.train_mask])
            acc_test = utils.accuracy(output[data.test_mask], labels[data.test_mask])
            print(acc_train.item(), acc_test.item())

            if best_acc_val < acc_val:
                best_acc_val = acc_val
                best_output = output
                best_feat = new_feat
                best_delta_feat = delta_feat

        print('best_acc_val:', best_acc_val.item())
        output = best_output

        self.reverse_node_mask = (output.argmax(1) == labels)
        loss = self.test_time_loss(model, best_feat, edge_index, edge_weight)
        print(f'Final loss: {loss}')
        print('Test:')
        self.get_perf(output, labels, data.test_mask)
        print('Validatoin:')
        self.get_perf(output, labels, data.val_mask)
        self.best_feat = best_feat
        return best_feat, best_delta_feat

    def inference(self, edge_index=None, edge_weight=None, feat=None):
        data = self.data
        if edge_index is None:
            edge_index = data.edge_index
        if feat is None:
            feat = data.x
        labels = data.y
        self.model.eval()
        output = self.model.forward(feat, edge_index)
        print(f'Test on {self.args.model}:')
        self.get_perf(output, labels, data.test_mask)

    def finetune(self, edge_index=None, edge_weight=None, feat=None, train_iters=201, verbose=False):
        print('Finetuning for %s epochs' % train_iters)
        data = self.data
        labels = data.y
        for param in self.model.parameters():
            param.requires_grad = True
        model = self.model
        if edge_index is None:
            edge_index, edge_weight = data.edge_index, None
        if feat is None:
            feat = data.x
        if verbose:
            print(f'=== finetuning {model.name} model ===')
        optimizer = torch.optim.Adam(model.parameters(), lr=model.lr, weight_decay=model.weight_decay)
        train_mask, val_mask = data.train_mask, data.val_mask
        best_loss_val = 100
        best_acc_val = 0
        for i in range(train_iters):
            model.train()
            optimizer.zero_grad()
            output = model.forward(feat, edge_index, edge_weight)
            # loss_train = F.nll_loss(output[train_mask], labels[train_mask])
            loss_train = self.test_time_loss(model, feat, edge_index, edge_weight)
            loss_train.backward()
            optimizer.step()

            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            model.eval()
            with torch.no_grad():
                output = model.forward(feat, edge_index)
            loss_val = F.nll_loss(output[val_mask], labels[val_mask])
            acc_val = utils.accuracy(output[val_mask], labels[val_mask])

            if best_acc_val < acc_val:
                best_acc_val = acc_val
                best_output = output
                weights = deepcopy(model.state_dict())

        print('best_acc_val:', best_acc_val.item())
        model.load_state_dict(weights)
        output = best_output
        if self.args.noise_feature > 0:
            print("After noise, test set results:", self.get_perf(output, labels, data.test_mask, verbose=0)[1])
            print('Abnomral test nodes:', self.get_perf(output, labels, self.noisy_nodes, verbose=0)[1])
            print('Normal test nodes:', self.get_perf(output, labels, data.test_mask & (~self.noisy_nodes), verbose=0)[1])
        else:
            print('Test:', self.get_perf(output, labels, data.test_mask, verbose=0)[1])
            print('Validatoin:', self.get_perf(output, labels, data.val_mask,verbose=0)[1])

    def pgd_attack(self):
        from deeprobust.graph.global_attack import PGDAttack
        from deeprobust.graph.defense import GCN
        from torch_geometric.utils import to_scipy_sparse_matrix, from_scipy_sparse_matrix
        from utils import mask_to_index

        device = self.device
        args = self.args
        data = self.data
        adj = to_scipy_sparse_matrix(data.edge_index).tocsr()
        adj_unnorm = adj
        adj = utils.normalize_adj(adj)
        adj = utils.sparse_mx_to_torch_sparse_tensor(adj).to(device)
        feat = data.x
        labels = data.y
        n = feat.shape[0]
        idx_train = mask_to_index(data.train_mask.cpu(), n)
        idx_val = mask_to_index(data.val_mask.cpu(), n)
        idx_test = mask_to_index(data.test_mask.cpu(), n)
        # Setup Attack Model
        target_gcn = GCN(nfeat=feat.shape[1],
              nhid=16,
              nclass=labels.max().item() + 1,
              dropout=0.5, device=device, lr=0.01)
        target_gcn = target_gcn.to(device)
        target_gcn.fit(feat, adj, labels, idx_train, idx_val, patience=200)

        print('=== setup attack model ===')
        attacker = PGDAttack(model=target_gcn, nnodes=adj.shape[0], loss_type='CE', device=device)
        attacker = attacker.to(device)

        # Here for the labels we need to replace it with predicted ones
        fake_labels = target_gcn.predict(feat, adj)
        fake_labels = torch.argmax(fake_labels, 1).to(device)
        idx_fake = np.concatenate([idx_train, idx_test])
        idx_others = list(set(np.arange(len(labels))) - set(idx_train))
        fake_labels = torch.cat([labels[idx_train], fake_labels[idx_others]])

        adj = utils.sparse_mx_to_torch_sparse_tensor(adj_unnorm).to(device)
        perturbations = int(args.ptb_rate * (adj._nnz()//2))
        adj = adj.to_dense()
        attacker.attack(feat.cpu(), adj.cpu(), fake_labels.cpu(), idx_fake, perturbations, epochs=100)

        print('=== testing GCN on Evasion attack ===')
        modified_adj = attacker.modified_adj
        self.model.eval()
        edge_index = modified_adj.nonzero().T
        output = self.model.predict(feat, edge_index)
        print("Test set results:")
        self.get_perf(output, labels, data.test_mask)
        return edge_index, None

    def augment(self, strategy='dropedge', p=0.5, edge_index=None, edge_weight=None):
        model = self.model
        if hasattr(self, 'delta_feat'):
            delta_feat = self.delta_feat
            feat = self.data.x + delta_feat
        else:
            feat = self.data.x
        if strategy == 'shuffle':
            idx = np.random.permutation(feat.shape[0])
            shuf_fts = feat[idx, :]
            output = model.get_embed(shuf_fts, edge_index, edge_weight)
        if strategy == "dropedge":
            edge_index, edge_weight = dropout_adj(edge_index, edge_weight, p=p)
            output = model.get_embed(feat, edge_index, edge_weight)
        if strategy == "dropfeat":
            feat = F.dropout(self.data.x, p=p) + self.delta_feat
            output = model.get_embed(feat, edge_index, edge_weight)
        if strategy == "dropnode":
            feat = self.data.x + self.delta_feat
            mask = torch.cuda.FloatTensor(len(feat)).uniform_() > p
            feat = feat * mask
            output = model.get_embed(feat, edge_index, edge_weight)
        if strategy == "featnoise":
            mean, std = 0, p
            noise = torch.randn(feat.size()) * std + mean
            feat = feat + noise.to(feat.device)
            output = model.get_embed(feat, edge_index)
        return output

    def test_time_loss(self, model, feat, edge_index, edge_weight=None):
        args = self.args

        loss = 0
        if 'LC' in args.loss: # label constitency
            output1 = self.augment(strategy='dropedge', p=0.5, edge_index=edge_index, edge_weight=edge_weight)
            output2 = self.augment(strategy='dropedge', p=0.0, edge_index=edge_index, edge_weight=edge_weight)
            output3 = self.augment(strategy='shuffle', edge_index=edge_index, edge_weight=edge_weight)
            lc_string = [v for v in args.loss.split('+') if 'LC' in v][0]
            beta = float(lc_string[:-2])
            loss += beta*(inner(output1, output2) - inner(output2, output3))

        if 'recon' in args.loss: # data reconstruction
            model = self.model
            output2 = self.augment(strategy='dropedge', p=0.2, edge_index=edge_index, edge_weight=edge_weight)
            loss += inner(output2[edge_index[0]], output2[edge_index[1]])

        if "train" in args.loss:
            train_mask = self.data.train_mask
            labels = self.data.y
            output = model.forward(feat, edge_index, edge_weight)
            loss += F.nll_loss(output[train_mask], labels[train_mask])

        if args.loss == "attack":
            loss = F.nll_loss(output, labels)

        if  "entropy" in args.loss:
            test_mask = self.data.test_mask
            batch_size = 10000
            sampled = np.random.permutation(np.arange(len(test_mask))[: batch_size])
            loss += entropy(output[sampled]).mean(0)

        return loss

    def pretrain_model(self):
        data = self.data
        args = self.args
        device = self.device
        feat, labels = data.x, data.y
        args.with_bn = False
        if args.dataset == 'arxiv':
            args.nlayers = 3; args.hidden=256; args.with_bn=True
        elif args.dataset in ['cora', 'citeseer', 'pubmed']:
            # args.nlayers = 2; args.hidden=256
            args.nlayers = 2; args.hidden=64
        elif args.dataset in ['cs', 'photo', 'physics', 'computers']:
            args.nlayers = 2; args.hidden=16; args.weight_decay=0; train_iters=400;
            alpha = 0.1
        else:
            raise NotImplementedError

        nclass = max(labels).item()+1

        if args.model == "GCN":
            model = GCN(nfeat=feat.shape[1], nhid=args.hidden, dropout=args.dropout,
                    nlayers=args.nlayers, with_bn=args.with_bn,
                    weight_decay=args.weight_decay, nclass=nclass,
                    device=device).to(device)
        elif args.model == "GAT":
            if args.dataset == 'arxiv':
                args.dropout = 0.2; args.hidden = 32; args.nlayers = 2
            else:
                args.dropout = 0.5; args.hidden = 8
            model = GAT(nfeat=feat.shape[1], nhid=args.hidden, heads=8, lr=0.005, nlayers=args.nlayers,
                  nclass=nclass, with_bn=args.with_bn, weight_decay=args.weight_decay,
                  dropout=args.dropout, device=device).to(device)
        elif args.model == "AirGNN" or args.model== "APPNP":
             if args.dataset == 'arxiv':
                 args.dropout=0.5; args.lambda_amp=0.3; args.alpha=0.1; args.weight_decay=0
             else:
                 args.dropout=0.8; args.lambda_amp=0.5; args.alpha=0.1

             if args.model == "APPNP": args.dropout = 0.2

             model = AirGNN(nfeat=feat.shape[1], nhid=args.hidden, dropout=args.dropout, with_bn=args.with_bn,
                     K=10, weight_decay=args.weight_decay, args=args, nlayers=args.nlayers,
                     nclass=max(labels).item()+1, device=device).to(device)
        elif args.model == "SAGE":
            model = SAGE(feat.shape[1], 32, max(labels).item()+1, num_layers=5,
                    dropout=0.0, lr=0.01, weight_decay=0, device=device).to(device)
        elif args.model == "GPR":
            model = GPRGNN(feat.shape[1], 32, max(labels).item()+1, dropout=0.0,
                    lr=0.01, weight_decay=0, device=device).to(device)
        else:
            raise NotImplementedError

        print(model)

        train_iters = 1000
        import os.path as osp
        filename = f'saved/{args.dataset}_{args.model}_s{args.seed}.pt'
        if args.debug and osp.exists(filename):
            model.load_state_dict(torch.load(filename, map_location='cuda'))
        else:
            model.fit(data, train_iters=1000, patience=1000, verbose=True)
            if args.debug:
                torch.save(model.state_dict(), filename)

        model.eval()
        model.data = data.to(self.device)

        import ipdb
        ipdb.set_trace()

        output = model.predict()
        labels = labels.to(device)
        print(f"{model.name} Test set results:", self.get_perf(output, labels, data.test_mask, verbose=0)[1])
        self.clean_node_mask = (output.argmax(1) == labels)
        return model

    def get_perf(self, output, labels, mask, verbose=True):
        loss = F.nll_loss(output[mask], labels[mask])
        acc = utils.accuracy(output[mask], labels[mask])
        if verbose:
            print("loss= {:.4f}".format(loss.item()),
                  "accuracy= {:.4f}".format(acc.item()))
        return loss.item(), acc.item()

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from **logits**."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

@torch.jit.script
def entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from **log_softmax**."""
    return -(torch.exp(x) * x).sum(1)

def add_feature_noise(data, noise_ratio, seed):
    np.random.seed(seed)
    n, d = data.x.shape
    # noise = torch.normal(mean=torch.zeros(int(noise_ratio*n), d), std=1)
    noise = torch.FloatTensor(np.random.normal(0, 1, size=(int(noise_ratio*n), d))).to(data.x.device)
    indices = np.arange(n)
    indices = np.random.permutation(indices)[: int(noise_ratio*n)]
    delta_feat = torch.zeros_like(data.x)
    delta_feat[indices] = noise - data.x[indices]
    data.x[indices] = noise
    mask = np.zeros(n)
    mask[indices] = 1
    mask = torch.tensor(mask).bool().to(data.x.device)
    return delta_feat, mask

def add_feature_noise_test(data, noise_ratio, seed):
    np.random.seed(seed)
    n, d = data.x.shape
    indices = np.arange(n)
    test_nodes = indices[data.test_mask.cpu()]
    selected = np.random.permutation(test_nodes)[: int(noise_ratio*len(test_nodes))]
    noise = torch.FloatTensor(np.random.normal(0, 1, size=(int(noise_ratio*len(test_nodes)), d)))
    noise = noise.to(data.x.device)

    delta_feat = torch.zeros_like(data.x)
    delta_feat[selected] = noise - data.x[selected]
    data.x[selected] = noise
    # mask = np.zeros(len(test_nodes))
    mask = np.zeros(n)
    mask[selected] = 1
    mask = torch.tensor(mask).bool().to(data.x.device)
    return delta_feat, mask

def add_structure_noise(data, noise_ratio, seed):
    np.random.seed(seed)
    from pyg_utils import coalesce
    num_edges_to_add = int(noise_ratio * data.edge_index.shape[1])
    edge_index_to_add = torch.randint(0, data.x.shape[0],
        (2, num_edges_to_add), dtype=torch.long)
    edge_index_to_add = edge_index_to_add.to(data.edge_index.device)
    # edge_index_to_add = to_undirected(edge_index_to_add)
    edge_index = torch.cat([data.edge_index, edge_index_to_add], dim=1)
    data.edge_index = coalesce(edge_index)

def add_structure_noise_remove(data, noise_ratio, seed):
    np.random.seed(seed)
    from deeprobust.graph.global_attack import Random
    from torch_geometric.utils import to_scipy_sparse_matrix, from_scipy_sparse_matrix, dropout_adj
    adj = to_scipy_sparse_matrix(data.edge_index)
    model = Random()
    model.attack(adj, n_perturbations=int(noise_ratio*adj.nnz)//2, type='remove')
    modified_adj = model.modified_adj
    data.edge_index = from_scipy_sparse_matrix(modified_adj)[0].to(data.edge_index.device)

def mask2indices(mask):
    all = torch.arange(len(mask))
    indices = all[mask]
    return indices.to(mask.device)

def inner(t1, t2):
    t1 = t1 / (t1.norm(dim=1).view(-1,1)+1e-15)
    t2 = t2 / (t2.norm(dim=1).view(-1,1)+1e-15)
    return (1-(t1 * t2).sum(1)).mean()

def l21norm(t):
    l2 = ((t**2).sum(1)+1e-8).sqrt()
    return l2.mean()

