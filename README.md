# HGCN (Hierarchical Graph Capsule Network)
Code of the AAAI 2021 paper [HGCN](https://arxiv.org/abs/2012.08734) 

## Prerequisites and Dependencies
Option 1: Nvidia Docker image
```
docker pull utasmile/pytorch:1.4.0-cuda10.0-cudnn7.5-devel
```
Option 2: Install the dependencies by yourself
* Pytorch >= 1.4.0
* [Pytorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

## Data
All the datasets used in the paper can be downloaded from [Benchmark Data Sets for Graph Kernels](https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets)

## Model Training
All the commands can be found in ```script.txt```
For example:
```
python main.py --epochs 100 --batch_size 256 --capsule-dimensions 128 --capsule-num 10 --dropout 0.1 --theta 0.1 --lr 0.003 --dataset IMDB-BINARY
```

## Citation
If you use HGCN in your research, please cite the following paper:
```
@article{yang2020hierarchical,
  title={Hierarchical Graph Capsule Network},
  author={Yang, Jinyu and Zhao, Peilin and Rong, Yu and Yan, Chaochao and Li, Chunyuan and Ma, Hehuan and Huang, Junzhou},
  journal={AAAI Conference on Artificial Intelligence (AAAI)},
  year={2021}
}
```
The code is largely borrowed from:
```
@article{chen2019powerful,
  title={Are powerful graph neural nets necessary? a dissection on graph classification},
  author={Chen, Ting and Bian, Song and Sun, Yizhou},
  journal={arXiv preprint arXiv:1905.04579},
  year={2019}
}
```
