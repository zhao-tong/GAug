import os
import copy
import torch
import pickle
import argparse
import numpy as np
import optuna
import scipy.sparse as sp
from models.GCN_dgl import GCN
from models.GAT_dgl import GAT
from models.GSAGE_dgl import GraphSAGE
from models.JKNet_dgl import JKNet

class AdaEdge(object):
    def __init__(self, adj_matrix, features, labels, tvt_nids, gnn, add_first, n_add, n_remove, conf_add, conf_remove, max_t=5):
        # hyperparameters
        self.add_first = add_first
        self.n_add = n_add
        self.n_remove = n_remove
        self.conf_add = conf_add
        self.conf_remove = conf_remove
        self.max_t = max_t
        self.gnn = gnn
        # data
        self.adj = adj_matrix
        self.features = features
        self.labels = labels
        self.tvt_nids = tvt_nids

    def fit(self):
        if self.gnn == 'gcn':
            GNN = GCN
        elif self.gnn == 'gat':
            GNN = GAT
        elif self.gnn == 'gsage':
            GNN = GraphSAGE
        elif self.gnn == 'jknet':
            GNN = JKNet

        model = GNN(self.adj, self.adj, self.features, self.labels, self.tvt_nids, print_progress=False, cuda=0)
        test_acc_init, val_acc_init, logits = model.fit()
        adj_new = self.adjustGraph(self.adj, logits)
        best_val_acc = val_acc_init
        test_acc = test_acc_init
        for i in range(1, self.max_t):
            model = GNN(adj_new, adj_new, self.features, self.labels, self.tvt_nids, print_progress=False, cuda=0)
            test_acc_i, val_acc_i, logits = model.fit()
            if val_acc_i <= best_val_acc:
                return test_acc
            else:
                test_acc = test_acc_i
                best_val_acc = val_acc_i
            adj_new = self.adjustGraph(adj_new, logits)
        return test_acc

    def adjustGraph(self, adj, logits):
        adj = copy.deepcopy(adj)
        adj = sp.lil_matrix(adj)
        logits = torch.softmax(logits, 1)
        tmp = torch.max(logits, 1)
        conf = tmp[0].numpy()
        pred = tmp[1].numpy()
        if self.add_first:
            adj_new = self.addEdge(adj, pred, conf)
            adj_new = self.removeEdge(adj_new, pred, conf)
        else:
            adj_new = self.removeEdge(adj, pred, conf)
            adj_new = self.addEdge(adj_new, pred, conf)
        return sp.csr_matrix(adj_new)

    def addEdge(self, adj, pred, conf):
        add_cnt = 0
        for i in range(adj.shape[0]):
            for j in range(i+1, adj.shape[0]):
                if adj[i, j] == 0 and pred[i] == pred[j] and conf[i] >= self.conf_add and conf[j] >= self.conf_add:
                    adj[i, j] = 1
                    adj[j, i] = 1
                    add_cnt += 1
                    if add_cnt >= self.n_add:
                        return adj
        return adj

    def removeEdge(self, adj, pred, conf):
        rm_cnt = 0
        for i in range(adj.shape[0]):
            for j in range(i+1, adj.shape[0]):
                if adj[i, j] > 0 and pred[i] != pred[j] and conf[i] >= self.conf_add and conf[j] >= self.conf_add:
                    adj[i, j] = 0
                    adj[j, i] = 0
                    rm_cnt += 1
                    if rm_cnt >= self.n_remove:
                        return adj
        return adj




