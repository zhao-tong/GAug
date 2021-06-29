Data Augmentation for Graph Neural Networks
====
This repository contains the source code for the AAAI'2021 paper:

[Data Augmentation for Graph Neural Networks](https://arxiv.org/pdf/2006.06830.pdf)

by [Tong Zhao](https://tzhao.io/) (tzhao2@nd.edu), [Yozen Liu](https://research.snap.com/team/yozen-liu),  [Leonardo Neves](https://research.snap.com/team/leonardo-neves), [Oliver Woodford](https://ojwoodford.github.io/), [Meng Jiang](http://www.meng-jiang.com/), and [Neil Shah](http://nshah.net/).

## Requirements

This code package was developed and tested with Python 3.7.6. Make sure all dependencies specified in the ```requirements.txt``` file are satisfied before running the model. This can be achieved by
```
pip install -r requirements.txt
```

## Usage
The scripts for hyperparameter search with Optuna are ```optuna_[method].py```.

All the parameters are included in ```best_parameters.json```. Results can be reproduced with the scripts ```train_[method].py```, which will automatically load the parameters. For example, to reproduce the result of GAugO with GCN on Cora, you can simply run:
```
python train_GAugO.py --dataset cora --gnn gcn --gpu 0
```

## Data
The format of data files are described in detail in the file ```data/README```.
Due to file size limit, for GAugM, only the edge_probabilities of Cora is provided.
Please find the all edge_probabilities files at https://tinyurl.com/gaug-data. The VGAE implementation I used for generating these edge_probabilities are also provided under the folder ```vgae/```.

## Cite
If you find this repository useful in your research, please cite our paper:

```bibtex
@inproceedings{zhao2021data,
  title={Data Augmentation for Graph Neural Networks},
  author={Zhao, Tong and Liu, Yozen and Neves, Leonardo and Woodford, Oliver and Jiang, Meng and Shah, Neil},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  number={12},
  pages={11015--11023},
  year={2021}
}
```

