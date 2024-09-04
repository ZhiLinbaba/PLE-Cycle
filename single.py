import os
import torch
import torch.nn as nn
import torch.utils.data as data_utils
from torch.utils.tensorboard import SummaryWriter
from dtw import dtw
from Network import c1l1_mini
from Network.lstm_1 import BidirectionalLSTM
from utils import Seq2SeqDataset, pearson_correlation
import random
import numpy as np

seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

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
dataloader_train = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True,)
dataloader_val= torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=True)

model =c1l1_mini.BidirectionalLSTM(input_size=1, hidden_size=16,  output_size=1,num_layers=1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.cuda()
optimizer = torch.optim.Adam(model.parameters(),lr=1e-2)#log2
scheduler=torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.98, last_epoch=-1)
criterion= nn.MSELoss(reduction='mean')
writer = SummaryWriter()
min_loss=10
train_losses = []
val_losses = []

for epoch in range(200):
    train_loss = 0.0
    correlation_imp = 0

    for input_data, target_data,in dataloader_train:

        input_data = input_data.float().to(device).cuda() #torch.Size([32, 1000, 1])
        target_data = target_data.float().to(device).cuda()
        optimizer.zero_grad()
        imp= model(input_data)
        correlation_imp = correlation_imp + pearson_correlation(imp[:, :, 0], target_data[:, :,1])##
        ##target[:,:,0]means vs,target[:,:,1]means vp,target[:,:,2]means imp,here is single for vp
        loss_imp = criterion(imp[:, :, 0], target_data[:, :,1]) ##改
        loss = loss_imp
        train_loss +=loss.item()
        loss.backward()
        optimizer.step()

    writer.add_scalars('Loss', {'train_loss': train_loss/len(dataloader_train)}, epoch)
    writer.add_scalars('imp_pear', {'train_seam': correlation_imp / len(dataloader_train)}, epoch)
    with torch.no_grad():  # 禁用梯度计算，以节省显存
        val_loss = 0.0
        correlation_imp = 0
        for input_data, target_data in dataloader_val:

            input_data = input_data.float().to(device).cuda()
            target_data = target_data.float().to(device).cuda()
            imp = model(input_data)
            correlation_imp = correlation_imp + pearson_correlation(imp[:, :, 0], target_data[:, :, 0])
            loss_imp = criterion(imp[:, :, 0], target_data[:, :,1])  ##改
            val_loss += loss_imp.item()

        writer.add_scalars('Loss', {'val_loss': val_loss / len(dataloader_val)}, epoch)
        writer.add_scalars('imp_pear', {'val': correlation_imp / len(dataloader_val)}, epoch)
        if (val_loss/len(dataloader_val) < min_loss):
            min_loss =(val_loss / len(dataloader_val))
            torch.save(model.state_dict(), f'D:/Code/python/多属性反演/c1l1_mini_pt/SEAM/单属性/vp-best')#***********************************
            print(epoch,(val_loss / len(dataloader_val)))
        # #
        torch.save(model.state_dict(), f'D:/Code/python/多属性反演/c1l1_mini_pt/SEAM/单属性/vp{epoch}')#**********************************
    # # scheduler.step()
writer.close()
