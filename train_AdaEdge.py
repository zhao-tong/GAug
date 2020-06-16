import os
import json
import pickle
import argparse
import numpy as np
import scipy.sparse as sp
import torch
from models.adaedge import AdaEdge

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='single')
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--gnn', type=str, default='gcn')
    parser.add_argument('--gpu', type=str, default='0')
    args = parser.parse_args()

    if args.gpu == '-1':
        gpu = -1
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        gpu = 0

    tvt_nids = pickle.load(open(f'data/graphs/{args.dataset}_tvt_nids.pkl', 'rb'))
    adj_orig = pickle.load(open(f'data/graphs/{args.dataset}_adj.pkl', 'rb'))
    features = pickle.load(open(f'data/graphs/{args.dataset}_features.pkl', 'rb'))
    labels = pickle.load(open(f'data/graphs/{args.dataset}_labels.pkl', 'rb'))
    if sp.issparse(features):
        features = torch.FloatTensor(features.toarray())

    params_all = json.load(open('best_parameters.json', 'r'))
    params = params_all['AdaEdge'][args.dataset][args.gnn]

    accs = []
    for _ in range(30):
        model = AdaEdge(adj_orig, features, labels, tvt_nids, args.gnn, params['add_first'], params['n_add'], params['n_remove'], params['conf_add'], params['conf_remove'])
        acc = model.fit()
        accs.append(acc)
    print(f'Micro F1: {np.mean(accs):.6f}, std: {np.std(accs):.6f}')
