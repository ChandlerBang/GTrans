# GTrans


[ICLR'23] Implementation of ["Empowering Graph Neural Networks with Test-Time Graph Transformation"](https://openreview.net/pdf?id=Lnxl5pr018)

**Key words**: out-of-distribution generalization, distribution shift, adversarial robustness, graph neural networks

Abstract
----
As powerful tools for representation learning on graphs, graph neural networks (GNNs) have facilitated various applications from drug discovery to recommender systems. Nevertheless, the effectiveness of GNNs is immensely challenged by issues related to data quality, such as distribution shift, abnormal features and adversarial attacks. Recent efforts have been made on tackling these issues from a modeling perspective which requires additional cost of changing model architectures or re-training model parameters. In this work, we provide a data-centric view to tackle these issues and propose a graph transformation framework named GTRANS which adapts and refines graph data at test time to achieve better performance. 


<div align=center><img src="https://github.com/ChandlerBang/GTrans/blob/main/GTrans.png" width="800"/></div>

## Requirements
We used Python 3.7.10. For the Python packages, please see [requirements.txt]().
```
deeprobust==0.2.8
dgl==0.9.1
dgl_cu102==0.6.1
GCL==0.6.11
googledrivedownloader==0.4
ipdb==0.13.7
matplotlib==3.5.2
networkx==2.5
numpy==1.20.1
ogb==1.3.5
pandas==1.2.3
scikit_learn==1.1.3
scipy==1.6.2
torch==1.13.0
torch_geometric==2.0.1
torch_scatter==2.0.8
torch_sparse==0.6.12
tqdm==4.60.0
visualization==1.0.0
```

## Download Datasets
We used the datasets provided by [Wu et al.](https://github.com/qitianwu/GraphOOD-EERM). We slightly modified their code to support data loading and put the code in the `GraphOOD-EERM` folder. 

You can make a directory `./GraphOOD-EERM/data` and download all the datasets through the Google drive:
```
https://drive.google.com/drive/folders/15YgnsfSV_vHYTXe7I4e_hhGMcx0gKrO8?usp=sharing
```
Make sure the data files are in the `./GraphOOD-EERM/data` folder:
```
project
│   README.md
│   train_both_all.py
│   script.sh
|   ...
|
└───GraphOOD-EERM
│   └───data
│       │   Amazon
│       │   elliptic
│       │   ...
│   
└───robustness
```
## Note
We note that the GCN used in the experiments of EERM does not normalize the adjacency matrix according to its open-source code. Here we normalize the adjacency matrix to make it consistent with the original GCN.

## Run our code
Simply run the following command to get started.
```
python train_both_all.py --gpu_id=0 --dataset=cora --model=GCN  --seed=0 --tune=0
python train_both_all.py --gpu_id=0 --dataset=ogb-arxiv --model=GCN  --seed=0 --tune=0
python train_both_all.py --gpu_id=0 --dataset=elliptic --model=GCN  --seed=0 --tune=0
python train_both_all.py --gpu_id=0 --dataset=cora --model=SAGE  --seed=0 --tune=0
python train_both_all.py --gpu_id=0 --dataset=ogb-arxiv --model=SAGE  --seed=0 --tune=0
python train_both_all.py --gpu_id=0 --dataset=cora --model=GAT  --seed=0 --tune=0
```
where `tune=0` indicates that we are using fixed hyper-parameters provided by us.

You can also run the following script.
```
mkdir saved
bash script.sh 
```
You can also try different losses for test-time graph transformation:
```
python train_both_all.py --gpu_id=0 --dataset=cora --model=GCN  --tune=0 --seed=0 --debug=1 --loss=LC
python train_both_all.py --gpu_id=0 --dataset=cora --model=GCN  --tune=0 --seed=0 --debug=1 --loss=recon
python train_both_all.py --gpu_id=0 --dataset=cora --model=GCN  --tune=0 --seed=0 --debug=1 --loss=entropy
```
Note that `LC` is the contrastive loss used in our work and by default the graph transformation is using this loss.

## Hyper-parameter tuning suggestion
Test-time graph transformation requires careful tuning skills. A general suggestion would be to choose small learning rate and small training epochs. If you are using GTrans for other datasets, please first tune the hyperparameters based on the validation set, i.e., `--test_val=1 --tune=1`.




## Robustness
Go to the robustness folder
```
cd robustness
```


Run abnormal setting:
```
python train_both_abn.py  --model=GCN --debug=1 --gpu_id=0 --dataset=cora --noise_feature=0.2
```
Before we run the adversarial attack setting, please first download the attacked graphs from the [[Google Drive link]](https://drive.google.com/file/d/1CVyO8v6NQcuFOkyOHtyXxvnHM5eNGIwU/view?usp=share_link). Place the downloaded zip file under the `robustness/saved` folder. 
```
cd saved; unzip attacked_graphs.zip; cd ..
```
You may also skip the above steps as our code also generate the attacks automatically.

Run adversarial attack:
```
python train_both_attack.py --dataset arxiv --model=GCN --seed=0 --ptb_rate=0.1 --debug=1 --gpu_id=0
```

## Cite
If you find this repo to be useful, please cite our paper. Thank you!
```
@inproceedings{
jin2023empowering,
title={Empowering Graph Representation Learning with Test-Time Graph Transformation},
author={Wei Jin and Tong Zhao and Jiayuan Ding and Yozen Liu and Jiliang Tang and Neil Shah},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=Lnxl5pr018}
}
```
