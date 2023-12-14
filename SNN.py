# -*- coding: UTF-8 -*-
"""
脉冲神经网络(Spiking Neural Network)
"""
import os, torch, time, json, random, copy, argparse, yaml
import numpy as np
from load_path  import *
from typing import Tuple
from torch.utils.data import Dataset, DataLoader
from torch import FloatTensor, Tensor, optim, nn
#在分类、聚类算法中，需要使用距离来度量相似性的时候、或者使用PCA技术进行降维的时候，StandardScaler表现更好（避免不同量纲对方差、协方差计算的影响）；
#在不涉及距离度量、协方差、数据不符合正态分布、异常值较少的时候，可使用MinMaxScaler。（eg：图像处理中，将RGB图像转换为灰度图像后将其值限定在 [0, 255] 的范围）；
#在带有的离群值较多的数据时，推荐使用RobustScaler。
from sklearn.preprocessing import MinMaxScaler,RobustScaler, StandardScaler
from sklearn.metrics import roc_auc_score, auc, roc_curve, log_loss
from torch.nn import (
    Module,
    Linear,
    Tanh,
    ReLU,
    Sigmoid,
    Sequential
)
np.random.seed(0)

# 参数解析
parser = argparse.ArgumentParser(description='parameter')
parser.add_argument('--counterfactual_sector', type=int)
parser.add_argument('--aggregation_type', type=str)
args = parser.parse_args()
counterfactual_sector = args.counterfactual_sector
aggregation_type = args.aggregation_type

# 加载脑图谱分区信息 
if not os.path.exists(BNA_SUBREGION_PATH):
    print(f'Pleas check {BNA_SUBREGION_PATH}, make sure it is there.')
    exit(1)
with open(BNA_SUBREGION_PATH, 'r') as file:
    # {lobe : {gyrus : "name labelID_start labelID_end", ...}, ...}
    subregion_info = json.load(file)
lobe_index =  {}  # {name : "startIdx endIdx"}. Idx = labelID - 1
gyrus_index = {}  # {name : "startIdx endIdx"}. Idx = labelID - 1
lobe_full_name, gyrus_full_name = [],[]
for lobe in subregion_info:
    lobe_full_name.append(lobe)
    l_startIdx_list, l_endIdx_list = [], []
    for gyrus in subregion_info[lobe]:
        gyrus_full_name.append(subregion_info[lobe][gyrus].split(',')[0])
        startIdx = int(subregion_info[lobe][gyrus].split(',')[-2]) - 1
        endIdx   = int(subregion_info[lobe][gyrus].split(',')[-1]) - 1
        gyrus_index[gyrus] = [startIdx, endIdx]
        l_startIdx_list.append(startIdx)
        l_endIdx_list.append(endIdx)      
    l_startIdx, l_endIdx = min(l_startIdx_list), max(l_endIdx_list)
    lobe_index[lobe] = [l_startIdx, l_endIdx]


# Embedding for each subregion 
def aggregation_embeddings(
        embeddings : list[np.array], 
        start : int, 
        end : int,
)->np.array:
    """
    聚合每个脑叶或者脑回中的亚区的嵌入向量
    
    Args:
        embeddings : 若干亚区的嵌入向量构成的列表
        start : 属于该脑叶/脑回的亚区起始坐标
        end : 属于该脑叶/脑回的亚区终止坐标

    Returns:
        result : 对该脑叶/脑回进行聚合得到的聚合嵌入向量

    """
    array = embeddings[start : end+1]
    Euclidean_distance = np.zeros([end-start+1, end-start+1], dtype=float)
    for i in range(len(array)):
        for j in range(i+1, len(array)):
            Euclidean_distance[i][j] = Euclidean_distance[j][i] = np.linalg.norm(array[i]-array[j])
    assert np.diag(Euclidean_distance).all() == 0
    weights = [sum(row) for row in Euclidean_distance]
    sum_weights = sum(weights)

    new_weights = [x if sum_weights == 0 else x/sum_weights for x in weights] 
    assert len(new_weights) == len(array)
    result = np.zeros(array[0].shape, dtype=float)
    for weight, embedd in zip(new_weights, array):
        result += weight*embedd
    return result
# 72 participants
all_data_pair = {}  # key(sub-id) : value(matirx 246×490, label∈{0,1})
for file in select_path_list(CONNECTION_MATRIX, '.npy'):
    # 1-抑郁症 0-健康人群
    name = file.split(os.sep)[2].split('_')[0]
    label = int(file[-len('.npy')-1])
    # 246×246 matrix for each participants
    connected_matrix = np.load(file)  # 246×246 取值(-1,1]

    embedding_from_edges = {}  # key(subregion id) : value(embedding 245 from its edges)
    for i in range(len(connected_matrix)):
        this_subregion_embedding_from_edges = [] 
        for j in range(len(connected_matrix[i])):
            if not i == j: 
                this_subregion_embedding_from_edges.append(connected_matrix[i][j]) 
        
        embedding_from_edges[i] = this_subregion_embedding_from_edges
    
    # 需要考虑图层面的任务 可解释性则研究边层面、节点层面
    embeddings = [] # 246×490
    for subregion_id in embedding_from_edges.keys():
        embeddings.append(np.array(embedding_from_edges[subregion_id]))

    results = []
    # 按照7个lobe(脑叶)进行节点聚合
    if aggregation_type == aggregation_lobe:
        for lobe in lobe_index:
            startIdx, endIdx = lobe_index[lobe][0], lobe_index[lobe][1]
            aggregation_result = aggregation_embeddings(embeddings, startIdx, endIdx)
            results.append(aggregation_result)  # 7×245
    # 按照24个gyrus(脑回)进行节点聚合
    elif aggregation_type == aggregation_gyrus:
        for gyrus in gyrus_index:
            startIdx, endIdx = gyrus_index[gyrus][0], gyrus_index[gyrus][1]
            aggregation_result = aggregation_embeddings(embeddings, startIdx, endIdx)
            results.append(aggregation_result)  # 24×245
    # 不进行节点聚合
    elif aggregation_type == aggregation_not:
        results = embeddings # 246*245
    else:
        print(f'Please check you aggregation type = {aggregation_type}')
        exit(1)
    all_data_pair[name] = (results, label)  # sub-01到sub-51 label=1; sub-52~sub72 label=0

# 数据增强 
def randomly_choose_half_point(m : int, n : int)->list:
    """
    从一个m * n的二维矩阵中随机挑选一半的位置
    """
    position = []
    for x in range(m):
        for y in range(n):
            position.append((x,y))
    return random.sample(position, len(position)//2)
# 72 -> 465. 51 * 5 = 255; 21 * 10 = 210
noise = 1e-1
original_keys = copy.deepcopy(list(all_data_pair.keys()))
for name in original_keys:
    (results, label) = all_data_pair[name]
    # 患者
    if label == 1:
        # 扩充5倍样本量, 新增4份样本
        for i in range(1,5):
            new_name = name + str(i)
            position = randomly_choose_half_point(len(results), len(results[0]))
            new_results = copy.deepcopy(results)
            for pos in position:
                add_or_reduce = random.randint(0,1)  # 该位置处的值 加上或者减去 噪声值
                new_results[pos[0]][pos[1]] += noise if add_or_reduce == 0 else -noise
            all_data_pair[new_name] = (new_results, label)
    # 正常人群
    elif label == 0:
        # 扩充10倍样本量, 新增9份样本
        for i in range(1,10):
            new_name = name + str(i)
            position = randomly_choose_half_point(len(results), len(results[0]))
            new_results = copy.deepcopy(results)
            for pos in position:
                add_or_reduce = random.randint(0,1)  # 该位置处的值 加上或者减去 噪声值
                new_results[pos[0]][pos[1]] += noise if add_or_reduce == 0 else -noise
            all_data_pair[new_name] = (new_results, label)

# 超参数 
with open(YAML_PATH, 'r') as file:
    yaml_data = yaml.load(file, Loader=yaml.FullLoader)
batch_size = len(all_data_pair) if yaml_data['batch_size'] == None else yaml_data['batch_size']
learning_rate = yaml_data['learning_rate']
epochs = yaml_data['epochs']
save_weights_pth = yaml_data['save_weights_pth']
save_result_txt = yaml_data['save_result_txt']

# 算力设备 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Device = {device}')


class GetData(Dataset):
    """
    依据数据集制作自己的datasets 

    Attributes:
        participant : 受试者
        embedding : 每个脑区的embedding
        targes : 01值. 1-抑郁症 0-心理健康
    """
    def __init__(self, participant : list, embeddings : list, targets : list) -> None:
        self.participant = participant
        self.embeddings = FloatTensor(embeddings)
        self.targets = FloatTensor(targets)
    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        return self.participant[index], self.embeddings[index], self.targets[index]
    def __len__(self) -> int:
        assert len(self.participant) == len(self.embeddings) == len(self.targets)
        return len(self.embeddings)

def get_train_value_dataloader() -> None:
    """ 
    划分训练集 验证集 测试集 
    """

    def make_dataloader(
            participants : list,
            x : np.array, 
            y : list
    ) -> DataLoader:
        dataset = GetData(participants, x, y)
        return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    
    # 组织数据
    participants, embeddings, labels = [], [], []
    for name in all_data_pair:
        participants.append(name)

        # 脑叶
        if aggregation_type == aggregation_lobe:
            if counterfactual_sector >= 0:
                sector_name = lobe_full_name[counterfactual_sector]
                embeddings.append(MinMaxScaler().fit_transform(all_data_pair[name][0])[counterfactual_sector].tolist())
            else:
                sector_name = 'All'
                embeddings.append(MinMaxScaler().fit_transform(all_data_pair[name][0]).flatten().tolist())
        # 脑回
        elif aggregation_type == aggregation_gyrus:
            if counterfactual_sector >= 0:
                sector_name = gyrus_full_name[counterfactual_sector]
                embeddings.append(MinMaxScaler().fit_transform(all_data_pair[name][0])[counterfactual_sector].tolist())
            else:
                sector_name = 'All'
                embeddings.append(MinMaxScaler().fit_transform(all_data_pair[name][0]).flatten().tolist())
        # 亚区
        elif aggregation_type == aggregation_not:
            if counterfactual_sector >= 0:
                sector_name = str(counterfactual_sector)
                embeddings.append(MinMaxScaler().fit_transform(all_data_pair[name][0])[counterfactual_sector].tolist())
            else:
                sector_name = 'All'
                embeddings.append(MinMaxScaler().fit_transform(all_data_pair[name][0]).flatten().tolist())
        else:
            print(f'Please check you aggregation type = {aggregation_type}')
            exit(1)

        # embeddings.append(MinMaxScaler().fit_transform(all_data_pair[name][0]).flatten().tolist())
        labels.append(all_data_pair[name][1])
    
    # 人工核验点
    assert len(participants) == len(embeddings) == len(labels) == 51*5 + 21*10

    # 全样本 465 = 255 + 210. 255 = [0~50]+[72~275]; 210 = [51~71]+[276~464]
    # 训练集:测试集 = 372:93. 372=204+168; 93=51+42
    train_loader = make_dataloader(participants[72:-21],
                                   embeddings[  72:-21],
                                   labels[      72:-21])
    test_loader  = make_dataloader(participants[:72]+participants[-21:],
                                   embeddings[  :72]+embeddings[  -21:],
                                   labels[      :72]+labels[      -21:])
    return sector_name, train_loader, test_loader

def calculate_AUC(pred_list : list, true_list : list) -> None:
    """
    两种方式的API计算AUC值

    Args:
        pred_list : 预测概率
        true_list : 真实标签

    """
    pred_np = np.array(pred_list)
    true_np = np.array(true_list)
    
    fpr, tpr, thresholds = roc_curve(true_np, pred_np, pos_label=1)
    roc_auc = auc(fpr, tpr)

    assert roc_auc == roc_auc_score(true_np, pred_np)
    return roc_auc, pred_list

###
# TODO 脉冲神经网络
###

def main():
    pass

if __name__ == '__main__':
    main()
