Data Augmentation for Graph Neural Networks
====

This is the implementation of the proposed GAugM and GAugO and baselines.
\[[paper](https://arxiv.org/pdf/2006.06830.pdf)\]

## Requirements

* Python 3.7.6
* Please refer to ```requirements.txt``` for all the packages used.

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
  booktitle={The Thirty-Fifth AAAI Conference on Artificial Intelligence},
  pages={},
  year={2021}
}
```

