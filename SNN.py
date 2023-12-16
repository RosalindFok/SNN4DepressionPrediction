# -*- coding: UTF-8 -*-
"""
脉冲神经网络(Spiking Neural Network)
"""
import torch, time, argparse, yaml
import numpy as np
from load_path  import *
from dataloader import get_train_value_dataloader
from torch import nn
from sklearn.metrics import roc_auc_score, auc, roc_curve, log_loss
from braincog.base.node.node import *
from braincog.base.connection.layer import *
from braincog.base.encoder.encoder import *
from braincog.model_zoo.base_module import BaseModule, BaseLinearModule
from braincog.utils import rand_ortho, mse
from torch import autograd
from tqdm import tqdm
np.random.seed(0)
torch.manual_seed(2122)
torch.cuda.manual_seed(2122)

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

# 超参数 
with open(YAML_PATH, 'r') as file:
    yaml_data = yaml.load(file, Loader=yaml.FullLoader)
batch_size = yaml_data['batch_size']
learning_rate = yaml_data['learning_rate']
epochs = yaml_data['epochs']
save_weights_pth = yaml_data['save_weights_pth']
save_result_txt = yaml_data['save_result_txt']

# 算力设备 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Device = {device}')


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
                #out_, in_ = m.weight.shape
                #m.weight.data = torch.Tensor(rand_ortho((out_, in_), np.sqrt(6. / (out_ + in_))))
                #m.bias.data.zero_()
                m.weight.data.zero_()
                m.bias.data.zero_()

        self.step = opt.step
        self.lr_target = opt.lr_target
        self.input_size = input_size
        self.loss = torch.nn.BCELoss() # torch.nn.CrossEntropyLoss() # mse

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
        cost = self.loss(ff_value[-1].flatten(), y_label.flatten())

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
snn = BaseGLSNN(input_size=input_size, hidden_sizes=[800] * 3, output_size=1, opt=args)  # 原始的MNIST的
snn.to(device)
optimizer = torch.optim.Adam(snn.forward_parameters(), lr=learning_rate)

# Dataloader
sector_name, train_loader, test_loader = get_train_value_dataloader(aggregation_type, counterfactual_sector, batch_size)

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
