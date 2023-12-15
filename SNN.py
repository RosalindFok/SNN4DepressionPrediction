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
from braincog.base.node.node import *
from braincog.base.connection.layer import *
from braincog.base.encoder.encoder import *
from braincog.model_zoo.base_module import BaseModule, BaseLinearModule
from braincog.utils import rand_ortho, mse
from torch import autograd
from tqdm import tqdm
np.random.seed(0)

# 参数解析
parser = argparse.ArgumentParser(description='parameter')
parser.add_argument('--counterfactual_sector', type=int)
parser.add_argument('--aggregation_type', type=str)
parser.add_argument('-step', type=int, default=10)
parser.add_argument('-lr_target', type=float, default=0.001)
parser.add_argument('-encode_type', type=str, default='direct')
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
    return roc_auc

class BaseGLSNN(BaseModule):
    """
    The fully connected model of the GLSNN
    :param input_size: the shape of the input
    :param hidden_sizes: list, the number of neurons of each layer in the hidden layers
    :param ouput_size: the number of the output layers
    """

    def __init__(self, input_size, hidden_sizes, output_size, opt=None):
        super().__init__(step=opt.step, encode_type=opt.encode_type)
        network_sizes = [input_size] + hidden_sizes + [output_size]
        feedforward = []
        for ind in range(len(network_sizes) - 1):
            feedforward.append(BaseLinearModule(in_features=network_sizes[ind], out_features=network_sizes[ind + 1], node=LIFNode))
        self.ff = nn.ModuleList(feedforward)
        feedback = []
        for ind in range(1, len(network_sizes) - 2):
            feedback.append(nn.Linear(network_sizes[-1], network_sizes[ind]))
        self.fb = nn.ModuleList(feedback)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                out_, in_ = m.weight.shape
                m.weight.data = torch.Tensor(rand_ortho((out_, in_), np.sqrt(6. / (out_ + in_))))
                m.bias.data.zero_()
        self.step = opt.step
        self.lr_target = opt.lr_target
        self.input_size = input_size
        # 交叉熵应该比MSE好
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        """
        process the information in the forward manner
        :param x: the input
        """
        self.reset()
        x = x.view(x.shape[0], self.input_size)
        sumspikes = [0] * (len(self.ff) + 1)
        sumspikes[0] = x
        for ind, mod in enumerate(self.ff):
            for t in range(self.step):
                spike = mod(sumspikes[ind])
                sumspikes[ind + 1] += spike
            sumspikes[ind + 1] = sumspikes[ind + 1] / self.step
        
        # 最后一层用Sigmoid拟合成概率
        sumspikes[-1] = torch.sigmoid(sumspikes[-1])

        return sumspikes

    def feedback(self, ff_value, y_label):
        """
        process information in the feedback manner and get target
        :param ff_value: the feedforward value of each layer
        :param y_label: the label of the corresponding input
        """
        fb_value = []
        # cost = mse(ff_value[-1], y_label)
        cost = self.loss(ff_value[-1], y_label)

        P = ff_value[-1]
        h_ = ff_value[-2] - self.lr_target * torch.autograd.grad(cost, ff_value[-2], retain_graph=True)[0]

        fb_value.append(h_)
        for i in range(len(self.fb) - 1, -1, -1):
            h = ff_value[i + 1]
            h_ = h - self.fb[i](P - y_label)
            fb_value.append(h_)
        return fb_value, cost

    def set_gradient(self, x, y):
        """
        get the corresponding update of each layer
        """
        ff_value = self.forward(x)

        fb_value, cost = self.feedback(ff_value, y)

        ff_value = ff_value[1:]
        len_ff = len(self.ff)
        for idx, layer in enumerate(self.ff):
            if idx == len_ff - 1:
                layer.fc.weight.grad, layer.fc.bias.grad = autograd.grad(cost, layer.fc.parameters())
            else:
                in1 = ff_value[idx]
                in2 = fb_value[len(fb_value) - 1 - idx]
                # loss_local = mse(in1, in2.detach())
                loss_local = self.loss(in1, in2.detach())
                layer.fc.weight.grad, layer.fc.bias.grad = autograd.grad(loss_local, layer.fc.parameters())
        return ff_value, cost

    def forward_parameters(self):
        res = []
        for layer in self.ff:
            res += layer.parameters()
        return res

    def feedback_parameters(self):
        res = []
        for layer in self.fb:
            res += layer.parameters()
        return res

length_node_embedding = 245
num_agg = 7 if aggregation_type == aggregation_lobe else 24
input_size = length_node_embedding if counterfactual_sector >= 0 else num_agg * length_node_embedding
# snn = BaseGLSNN(input_size=input_size, hidden_sizes=[800] * 3, output_size=1, opt=args)  # 原始的MNIST的
snn = BaseGLSNN(input_size=input_size, hidden_sizes=[800]*5, output_size=1, opt=args)
snn.to(device)
optimizer = torch.optim.Adam(snn.forward_parameters(), lr=learning_rate)

# Dataloader
sector_name, train_loader, test_loader = get_train_value_dataloader()

# 训练
def train(ep : int) -> list:
    snn.train()
    for user, xt, yt in train_loader:  # 372
        optimizer.zero_grad()  # 清除梯度
        xt = xt.to(device)
        yt = yt.view(-1, 1).to(device)
        outputs, loss = snn.set_gradient(xt, yt)
        optimizer.step()  # 优化更新
    return loss

# 测试
def test():
    snn.eval()
    pred_list = []
    true_list = []
    with torch.no_grad():
        for user, xv, yv in test_loader: # 93
            xv = xv.to(device)
            y_pred = snn(xv)[-1].cpu().flatten().tolist()
            pred_list += y_pred
            true_list += yv   

    roc_auc = calculate_AUC(pred_list, true_list)
    y_true, y_pred = np.array(true_list), np.array(pred_list)
    logLoss = log_loss(y_true, y_pred)

    for x,y in zip(y_pred, y_true):
        print(x,y)
    print(f'AUC = {roc_auc}, logLoss = {logLoss}')

def main():
    start_time = time.time()
    all_train_loss = []
    for ep in tqdm(range(epochs)):
        train_loss = train(ep)
        all_train_loss.append(train_loss.cpu().tolist())
    
    print(all_train_loss)    
    
    test()
    exit(1)


if __name__ == '__main__':
    main()
