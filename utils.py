import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
def pearson_correlation(tensor1, tensor2):
    if tensor1.shape != tensor2.shape:
        raise ValueError("Input tensors must have the same shape")

    mean1 = tensor1.mean()
    mean2 = tensor2.mean()
    centered1 = tensor1 - mean1
    centered2 = tensor2 - mean2
    numerator = (centered1 * centered2).sum()
    denominator = torch.sqrt((centered1 ** 2).sum() * (centered2 ** 2).sum())
    correlation = numerator / denominator if denominator != 0 else torch.tensor(0.0)
    return correlation
class Seq2SeqDataset(Dataset):
    def __init__(self, input_data, target_data):
        self.input_data = input_data
        self.target_data = target_data

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        return self.input_data[idx], self.target_data[idx]
class Seq2SeqDataset_1(Dataset):
    def __init__(self, input_data):
        self.input_data = input_data


    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        return self.input_data[idx]

def z_score_normalization(data):
    mean_val = np.mean(data)
    std_dev = np.std(data)
    normalized_data = [(x - mean_val) / std_dev for x in data]
    return np.array(normalized_data)
class UnlabelDataset(Dataset):
    def __init__(self, input_data):
        self.input_data = input_data


    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        return self.input_data[idx]



def create_time_windows(data, window_size, stride):
    """
    对时间序列进行时窗处理

    Parameters:
    - data: 输入的时间序列
    - window_size: 时窗大小
    - stride: 取样间隔

    Returns:
    - windows: 时窗处理后的数据，返回一个二维数组，每行代表一个时窗
    """
    windows = []
    for i in range(0, int((len(data) - window_size)//stride + 1),):

        window = data[i*stride:i*stride + window_size]
        windows.append(window)
    return np.array(windows)

def add_zeros_to_sequence(sequence, x, y):
    """
    在序列的前 x 个数字和后 y 个数字添加零值，不更改原始序列长度。

    参数:
    sequence (list): 输入序列。
    x (int): 前 x 个数字为零。
    y (int): 后 y 个数字为零。

    返回:
    list: 修改后的序列。
    """
    if x < 0 or y < 0:
        raise ValueError("x和y应该是非负整数")

    # 复制原始序列
    modified_sequence = sequence.copy()

    # 在序列前面添加零值
    modified_sequence[:x] = [0] * x

    # 在序列后面添加零值
    modified_sequence[-y:] = [0] * y

    return modified_sequence


class DTWLoss(nn.Module):
    def __init__(self):
        super(DTWLoss, self).__init__()

    def forward(self, x, y):
        device = x.device  # 获取设备信息
        x = x.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        alignment = dtw(x, y, keep_internals=False, distance_only=True)
        dtw_distance = alignment.distance
        return torch.tensor(dtw_distance, dtype=torch.float32, requires_grad=True).to(device)
