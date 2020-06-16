import gc
import sys
import math
import time
import pickle
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl import DGLGraph
from dgl.nn.pytorch.conv import SAGEConv
from sklearn.metrics import f1_score

class GraphSAGE(object):
    def __init__(self, adj, adj_eval, features, labels, tvt_nids, cuda=-1, hidden_size=128, n_layers=1, epochs=200, seed=-1, lr=1e-2, weight_decay=5e-4, dropout=0.5, print_progress=True, dropedge=0):
        self.t = time.time()
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.print_progress = print_progress
        self.dropedge = dropedge
        # config device
        if not torch.cuda.is_available():
            cuda = -1
        self.device = torch.device(f'cuda:{cuda%8}' if cuda>=0 else 'cpu')
        # fix random seeds if needed
        if seed > 0:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        self.load_data(adj, adj_eval, features, labels, tvt_nids)

        self.model = GraphSAGE_model(self.features.size(1),
                                    hidden_size,
                                    self.n_class,
                                    n_layers,
                                    F.relu,
                                    dropout,
                                    aggregator_type='gcn')
        # move everything to device
        self.model.to(self.device)

    def load_data(self, adj, adj_eval, features, labels, tvt_nids):
        if isinstance(features, torch.FloatTensor):
            self.features = features
        else:
            self.features = torch.FloatTensor(features)
        if self.features.size(1) in (1433, 3703):
            self.features = F.normalize(self.features, p=1, dim=1)
        if len(labels.shape) == 2:
            labels = torch.FloatTensor(labels)
        else:
            labels = torch.LongTensor(labels)
        self.labels = labels
        if len(self.labels.size()) == 1:
            self.n_class = len(torch.unique(self.labels))
        else:
            self.n_class = labels.size(1)
        self.train_nid = tvt_nids[0]
        self.val_nid = tvt_nids[1]
        self.test_nid = tvt_nids[2]
        # adj for training
        assert sp.issparse(adj)
        if not isinstance(adj, sp.coo_matrix):
            adj = sp.coo_matrix(adj)
        adj.setdiag(1)
        self.adj = adj
        adj = sp.csr_matrix(adj)
        self.G = DGLGraph(self.adj)
        # adj for inference
        assert sp.issparse(adj_eval)
        if not isinstance(adj_eval, sp.coo_matrix):
            adj_eval = sp.coo_matrix(adj_eval)
        adj_eval.setdiag(1)
        adj_eval = sp.csr_matrix(adj_eval)
        self.adj_eval = adj_eval
        self.G_eval = DGLGraph(self.adj_eval)

    def dropEdge(self):
        upper = sp.triu(self.adj, 1)
        n_edge = upper.nnz
        n_edge_left = int((1 - self.dropedge) * n_edge)
        index_edge_left = np.random.choice(n_edge, n_edge_left, replace=False)
        data = upper.data[index_edge_left]
        row = upper.row[index_edge_left]
        col = upper.col[index_edge_left]
        adj = sp.coo_matrix((data, (row, col)), shape=self.adj.shape)
        adj = adj + adj.T
        adj.setdiag(1)
        self.G = DGLGraph(adj)
        # normalization (D^{-1/2})
        degs = self.G.in_degrees().float()
        norm = torch.pow(degs, -0.5)
        norm[torch.isinf(norm)] = 0
        norm = norm.to(self.device)
        self.G.ndata['norm'] = norm.unsqueeze(1)

    def fit(self):
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.lr,
                                     weight_decay=self.weight_decay)
        # data
        features = self.features.to(self.device)
        labels = self.labels.to(self.device)
        # loss function for node classification
        if len(self.labels.size()) == 2:
            nc_criterion = nn.BCEWithLogitsLoss()
        else:
            nc_criterion = nn.CrossEntropyLoss()

        best_vali_acc = 0.0
        best_logits = None
        for epoch in range(self.epochs):
            if self.dropedge > 0:
                self.dropEdge()
            self.model.train()
            logits = self.model(self.G, features)
            # losses
            # l = F.nll_loss(logits[self.train_nid], labels[self.train_nid])
            l = nc_criterion(logits[self.train_nid], labels[self.train_nid])
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            # validate (without dropout)
            self.model.eval()
            with torch.no_grad():
                logits_eval = self.model(self.G_eval, features).detach().cpu()
            vali_acc, _ = self.eval_node_cls(logits_eval[self.val_nid], labels[self.val_nid].cpu())
            if self.print_progress:
                print('Epoch [{:2}/{}]: loss: {:.4f}, vali acc: {:.4f}'.format(epoch+1, self.epochs, l.item(), vali_acc))
            if vali_acc > best_vali_acc:
                best_vali_acc = vali_acc
                best_logits = logits_eval
                test_acc, conf_mat = self.eval_node_cls(logits_eval[self.test_nid], labels[self.test_nid].cpu())
                if self.print_progress:
                    print(f'                 test acc: {test_acc:.4f}')
        if self.print_progress:
            print(f'Final test results: acc: {test_acc:.4f}')
        del self.model, features, labels, self.G
        torch.cuda.empty_cache()
        gc.collect()
        t = time.time() - self.t
        return test_acc, best_vali_acc, best_logits

    def eval_node_cls(self, logits, labels):
        # preds = torch.argmax(logits, dim=1)
        # correct = torch.sum(preds == labels)
        # acc = correct.item() / len(labels)
        if len(labels.size()) == 2:
            preds = torch.round(torch.sigmoid(logits))
        else:
            preds = torch.argmax(logits, dim=1)
        micro_f1 = f1_score(labels, preds, average='micro')
        # calc confusion matrix
        # conf_mat = np.zeros((self.n_class, self.n_class))
        # for i in range(len(preds)):
        #     conf_mat[labels[i], preds[i]] += 1
        return micro_f1, 1

class GraphSAGE_model(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type):
        super(GraphSAGE_model, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type, feat_drop=0., activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type, feat_drop=dropout, activation=activation))
        # output layer
        self.layers.append(SAGEConv(n_hidden, n_classes, aggregator_type, feat_drop=dropout, activation=None))

    def forward(self, g, features):
        h = features
        for layer in self.layers:
            h = layer(g, h)
        return h
