Data Augmentation for Graph Neural Networks
====

This is the implementation of the proposed GAugM and GAugO and baselines.

## Requirements

* Python 3.7.6
* Please refer to ```requirements.txt``` for all the packages used.

## Usage
The scripts for Optuna parameter search are ```optuna_[method].py```.

All the parameters are included in ```best_parameters.json```. Results can be reproduced with the scripts ```train_[method].py```, which will automatically load the parameters. For example, to reproduce the result of GAugO with GCN on Cora, you can simply run:
```
python train_GAugO.py --dataset cora --gnn gcn --gpu 0
```

## Data
The format of data files are described in detail in the file ```data/README```.
Due to file size limit, only the edge_probabilities of Cora is provided.
Please find the all edge_probabilities files at https://tinyurl.com/gaug-data.

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

