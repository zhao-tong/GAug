import os
import pickle
import argparse
import numpy as np
import scipy.sparse as sp
import torch
from collections import Counter
from models.GCN_dgl import GCN
from models.GAT_dgl import GAT
from models.GSAGE_dgl import GraphSAGE
from models.JKNet_dgl import JKNet

parser = argparse.ArgumentParser(description='single')
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()

gpu = args.gpu
if gpu == '-1':
    cuda = -1
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    cuda = 0

def test(dataset, gnn, add_train=0):
    tvt_nids = pickle.load(open(f'data/graphs/{dataset}_tvt_nids.pkl', 'rb'))
    adj_orig = pickle.load(open(f'data/graphs/{dataset}_adj.pkl', 'rb'))
    features = pickle.load(open(f'data/graphs/{dataset}_features.pkl', 'rb'))
    labels = pickle.load(open(f'data/graphs/{dataset}_labels.pkl', 'rb'))
    if sp.issparse(features):
        features = torch.FloatTensor(features.toarray())
    if add_train > 0:
        if add_train < 20:
            new_trainids = []
            cnt = Counter()
            for i in tvt_nids[0]:
                if cnt[labels.numpy()[i]] < add_train:
                    new_trainids.append(i)
                    cnt[labels.numpy()[i]] += 1
            tvt_nids[0] = np.array(new_trainids)
        else:
            tvt_nids[0] = np.concatenate((tvt_nids[0], np.arange(640, 640+add_train)))
    if gnn == 'gcn':
        GNN = GCN
    elif gnn == 'gat':
        GNN = GAT
    elif gnn == 'gsage':
        GNN = GraphSAGE
    elif gnn == 'jknet':
        GNN = JKNet
    accs = []
    for _ in range(30):
        model = GNN(adj_orig, adj_orig, features, labels, tvt_nids, print_progress=False, cuda=0, dropedge=0)
        acc, _, _ = model.fit()
        accs.append(acc)
    acc = np.mean(accs)
    std = np.std(accs)
    print(f'{dataset} {gnn}, Micro F1: {acc:.6f} std: {std:.6f}')
    return acc

if __name__ == "__main__":
    datasets = ['cora', 'citeseer', 'flickr', 'blogcatalog', 'ppi', 'airport']
    gnns = ['gcn', 'gsage', 'gat', 'jknet']
    for d in datasets:
        for model in gnns:
            test(d, model)

