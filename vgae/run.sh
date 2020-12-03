# vgae
python main.py --cuda --no_mask --val_frac 0.9 --dataset cora --gen_graphs 200 --epochs 200
# gae
python main.py --cuda --no_mask --val_frac 0.9 --dataset cora --gen_graphs 1 --epochs 200 --gae

# vgae
python main.py --cuda --no_mask --val_frac 0.9 --dataset citeseer --gen_graphs 200 --epochs 200

# vgae
python main.py --cuda --no_mask --val_frac 0.4 --dataset pubmed --gen_graphs 50 --epochs 200

python main.py --cuda 0 --no_mask --val_frac 0.05 --dataset blogcatalog --gen_graphs 5 --epochs 200
python main.py --cuda 1 --no_mask --val_frac 0.05 --dataset flickr --gen_graphs 5 --epochs 200
python main.py --cuda 2 --no_mask --val_frac 0.05 --dataset ppi --gen_graphs 5 --epochs 200
python main.py --no_mask --val_frac 1 --dataset zkc --epochs 200

python main.py --cuda 0 --no_mask --val_frac 0.05 --dataset flickr --gen_graphs 1 --epochs 200 --gae
python main.py --cuda 0 --no_mask --val_frac 0.99 --dataset airport --gen_graphs 1 --epochs 200 --gae
