import itertools
import os
import torch
import torch.nn as nn
import torch.utils.data as data_utils
from matplotlib import pyplot as plt
from torch.utils import data
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from Network import c1l1_mini, lstm_1
from Network.lstm_1 import BidirectionalLSTM
from utils import Seq2SeqDataset, UnlabelDataset, z_score_normalization
import random
import numpy as np
seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
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
def custom_collate(batch):
    return torch.stack(batch)


mean = [1932.1715,3612.8735,8298.253]
std = [920.264,1103.41710,3129.3345]
data_range=[2965.5173,3310.0,12338.255]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
unlabel=np.load("D:/Code/python/多属性反演/Synthetic seismic records/Seam/30Hz_1000sample.npy")
unlabel=z_score_normalization(unlabel)
# unlabel=np.delete(unlabel,np.arange(0,1750,100),axis=1)#(1500, 1733)
unlabel_transposed = unlabel.T  # Shape will be (1733, 1500)

unlabel_tensor = torch.tensor(unlabel_transposed).float()

batch_size = 8
unlabel_dataset = UnlabelDataset(unlabel_tensor)  # Create TensorDataset
unlabel_dataloader = iter(DataLoader(unlabel_dataset, batch_size=batch_size, shuffle=True))


input_train_folder = 'D:/Code/python/多属性反演/Dataset/Seam_norm_t/seismic'
target_train_folder = 'D:/Code/python/多属性反演/Dataset/Seam_norm_t/attribute'
input_val_folder = 'D:/Code/python/多属性反演/Dataset/val/seismic'
target_val_folder = 'D:/Code/python/多属性反演/Dataset/val/attribute'

input_train_data = []
target_train_data = []
input_val_data = []
target_val_data = []
for file_name in os.listdir(input_train_folder):
    input_file = os.path.join(input_train_folder, file_name)
    input_train_data.append(np.load(input_file))
for file_name in os.listdir(target_train_folder):
    target_file = os.path.join(target_train_folder, file_name)
    target_train_data.append(np.load(target_file))
for file_name in os.listdir(input_val_folder):
    input_val_file = os.path.join(input_val_folder, file_name)
    input_val_data.append(np.load(input_val_file))
for file_name in os.listdir(target_val_folder):
    target_val_file = os.path.join(target_val_folder, file_name)
    target_val_data.append(np.load(target_val_file))

train_dataset = Seq2SeqDataset(input_train_data, target_train_data)
val_dataset = Seq2SeqDataset(input_val_data, target_val_data)

# sampler = torch.Generator().manual_seed(seed)
# dataset_train, dataset_val = data_utils.random_split(my_dataset, [train_size, val_size],generator=sampler)
dataloader_train = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True,)
dataloader_val= torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=True)


in_model =c1l1_mini.BidirectionalLSTM(input_size=1, hidden_size=16,  output_size=1,num_layers=1)
for_model =lstm_1.BidirectionalLSTM(input_size=1,hidden_size=16,num_layers=1,output_size=1)


in_model.cuda()
for_model.cuda()

optimizer = torch.optim.Adam(itertools.chain(in_model.parameters(), for_model.parameters()),lr=1e-2)#log2
scheduler=torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.98, last_epoch=-1)

criterion= nn.MSELoss(reduction='mean')

writer = SummaryWriter()
min_loss=10
train_losses = []
val_losses = []

for epoch in range(200):
    train_loss = 0.0

    for input, target in dataloader_train:


        input= input.float().to(device).cuda()
        target = target.float().to(device).cuda()
        optimizer.zero_grad()

        output_pre=in_model(input)
        input_res=for_model(output_pre)
        loss_in = criterion(output_pre[:, :, 0], target[:, :, 1])##三处target改
        ##target[:,:,0]means vs,target[:,:,1]means vp,target[:,:,2]means imp,here is single for vp
        loss_cycle_in=criterion(input_res[:, :, 0], input[:, :, 0])

        input_pre=for_model(target[:, :, 1].unsqueeze(-1))##三处target改
        target_res=in_model(input_pre)
        loss_for=criterion(input_pre[:, :, 0],input[:, :, 0])
        loss_cycle_for=criterion(target_res[:, :, 0],target[:, :, 1])##三处target改


        try:
            x_u = next(unlabel_dataloader).to(device)
        except:
            unlabel_dataloader = iter(DataLoader(unlabel_dataset, batch_size=8, shuffle=True))
            x_u = next(unlabel_dataloader).to(device)

        pre=in_model(x_u.unsqueeze(-1))
        rec_x_u=for_model(pre)
        loss_un=criterion(x_u.unsqueeze(-1),rec_x_u)

        loss=loss_for*0.1+loss_in*1+loss_cycle_for*0.1+loss_cycle_in*0.1+0.1*loss_un

        train_loss +=loss.item()
        loss.backward()
        optimizer.step()

    writer.add_scalars('Loss', {'train_loss': train_loss/len(dataloader_train),
                                   }, epoch)
    with torch.no_grad():  # 禁用梯度计算，以节省显存
        val_loss = 0.0
        correlation_imp = 0
        for input_data, target_data in dataloader_val:
            input_data = input_data.float().to(device).cuda()
            target_data = target_data.float().to(device).cuda()
            imp = in_model(input_data)
            correlation_imp = correlation_imp + pearson_correlation(imp[:, :, 0], target_data[:, :, 1]) ##target要改
            loss_imp = criterion(imp[:, :, 0], target_data[:, :, 1])
            val_loss += loss_imp.item()

        writer.add_scalars('Loss', {'val_loss': val_loss / len(dataloader_val)}, epoch)
        writer.add_scalars('imp_pear', {'val': correlation_imp / len(dataloader_val)}, epoch)
        if (val_loss/len(dataloader_val) < min_loss):
            min_loss =(val_loss / len(dataloader_val))
            torch.save(in_model.state_dict(), f'D:/Code/python/多属性反演/c1l1_mini_pt/SEAM/单属性+半监督/vp-best')
            print(epoch,(val_loss / len(dataloader_val)))

        torch.save(in_model.state_dict(), f'D:/Code/python/多属性反演/c1l1_mini_pt/SEAM/单属性+半监督/vp{epoch}')
    # # scheduler.step()

writer.close()
