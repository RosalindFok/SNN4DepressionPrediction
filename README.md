# 基于脉冲神经网络的抑郁症风险预测与可解释性分析
## 项目用途
中国科学院大学 2023秋季学期 系统与计算神经科学<br>
[![BrainCog](https://img.shields.io/badge/SNN-BrainCog-brightgreen.svg)](https://www.brain-cog.network/)
[![Brainnetome Atlas](https://img.shields.io/badge/Atlas-Brainnetome-blue.svg)](https://atlas.brainnetome.org/)

## 贡献者(排名不分先后)
霍育福 肖展琪 解森炜 南佳霖 张嘉峻

## 数据与环境配置
### 数据集
可以调节您自己的路径: `load_path.py`<br>
所用数据集为: [depression_ds002748](https://openneuro.org/datasets/ds002748/versions/1.0.5)
```shell
.
├─SNN4DepressionPrediction
│  └─BN_Atlas
├─connection_matrix
└─depression_ds002748
    ├─sub-01
    │  ├─anat
    │  └─func
    ├─ ... ...
```

### 运行环境
|Device|Info|
|:-:|:-:|
|CPU|AMD Ryzen 7 5800H with Radeon Graphics 16.0 GB|
|GPU|GeForce 3060 Laptop GPU; Driver Version: 512.36;CUDA Version: 11.6|
|OS|Windows 11 |

### 依赖环境
```shell
# Python 3.11.4
pip install braincog
pip install nibabel
pip install nilearn
pip install matplotlib
pip install yaml
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install tqdm
```

## 程序运行
基于脑网络组图谱生成每名受试者的功能连接矩阵(并行计算 可根据实验机器情况调整): `python functional_connection.py`<br>
更改超参数: `config.yaml`<br>
更改聚合类型并运行脉冲神经网络: `python run.py`<br>

## 运行结果
|Aggregation Type|AUC|LogLoss|
|:-:|:-:|:-:|
|Lobe|0.8809523809523809|0.5515018714141696|
|Gyrus|0.8711484593837535|0.5555866610637409|

以脑叶为聚合层面, 训练过程中的损失函数变化值如下图所示:
![Lobe Train Loss](./Figs/LobeTrainLoss.svg)
