import itertools
import os
import torch
import torch.nn as nn
import torch.utils.data as data_utils
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from Network import lstm_1, PLE_c1l1mini
from utils import Seq2SeqDataset, UnlabelDataset, z_score_normalization
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

mean = [1932.1715,3612.8735,8298.253] ## mean of vs,vp,imp
std = [920.264,1103.41710,3129.3345]  ## std of vs vp imp
data_range=[2965.5173,3310.0,12338.255]## datarange of vs vp imp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
unlabel=np.load("D:/Code/python/多属性反演/Synthetic seismic records/Seam/30Hz_1000sample.npy") ##unlabel data load
unlabel=z_score_normalization(unlabel) #unlabel data normalized

unlabel_transposed = unlabel.T

# Convert to PyTorch tensor
unlabel_tensor = torch.tensor(unlabel_transposed).float()

# Create a DataLoader with custom collate function
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

dataloader_train = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True,)
dataloader_val= torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=True)
in_model =PLE_c1l1mini.PLE_cl(input_size=1,hidden_size=16,num_layers=1,output_size=1)
for_model =lstm_1.BidirectionalLSTM(input_size=1,hidden_size=16,num_layers=1,output_size=1)

in_model.cuda()
for_model.cuda()
optimizer = torch.optim.Adam(itertools.chain(in_model.parameters(), for_model.parameters()),lr=1e-2)#log2

criterion= nn.MSELoss(reduction='mean')
writer = SummaryWriter()
min_loss=10
train_losses = []
val_losses = []

# Run training loop

for epoch in range(130):
    train_loss = 0.0
    correlation_imp = 0

    for input, target in dataloader_train:
        ##input (batch,length,dim),target(batch,length,dim),dim=0 is vs,dim=1 is vp, dim=2 is imp
        input= input.float().to(device).cuda()
        target = target.float().to(device).cuda()
        optimizer.zero_grad()
        #train inversion network

        output_pre_1,output_pre_2,output_pre_3=in_model(input)
        input_res=for_model(output_pre_3)

        loss_in = (criterion(output_pre_1[:, :, 0], target[:, :, 0])+\
                  criterion(output_pre_2[:, :, 0], target[:, :, 1])+\
                  criterion(output_pre_3[:, :, 0], target[:, :, 2]))/3

        loss_cycle_in=criterion(input_res[:, :, 0], input[:, :, 0])

#       #train forward network
        input_pre=for_model(target[:, :, 2].unsqueeze(-1))
        target_res_1,target_res_2,target_res_3=in_model(input_pre)

        loss_for=criterion(input_pre[:, :, 0],input[:, :, 0])
        loss_cycle_for=(criterion(target_res_1[:, :, 0],target[:, :, 0])\
                       +criterion(target_res_2[:, :, 0],target[:, :, 1])\
                       +criterion(target_res_3[:, :, 0], target[:, :, 2]))/3
        #unlabel data train
        try:
            x_u = next(unlabel_dataloader).to(device)
        except:
            unlabel_dataloader = iter(DataLoader(unlabel_dataset, batch_size=8, shuffle=True))
            x_u = next(unlabel_dataloader).to(device)

        pre_1,pre_2,pre_3=in_model(x_u.unsqueeze(-1))
        rec_x_u=for_model(pre_3)
        loss_un=criterion(x_u.unsqueeze(-1),rec_x_u)
        loss=loss_for*0.1+loss_in*1+loss_cycle_for*0.1+loss_cycle_in*0.1+0.1*loss_un  #loss of PLE-Cycle

        train_loss +=loss.item()
        loss.backward()
        optimizer.step()

    writer.add_scalars('Loss', {'train_loss': train_loss/len(dataloader_train),
                                   }, epoch)
    writer.add_scalars('imp_pear', {'train_seam': correlation_imp / len(dataloader_train),}, epoch)

    with torch.no_grad():
        val_loss = 0.0
        val_vs_loss = 0
        val_vp_loss = 0
        val_imp_loss = 0
        correlation_imp = 0
        for input_data, target_data in dataloader_val:
            input_data = input_data.float().to(device).cuda()
            target_data = target_data.float().to(device).cuda()
            vs, vp, imp = in_model(input_data)
            # imp = model(input_data)
            loss_vs = criterion(vs[:, :, 0], target_data[:, :, 0])
            loss_vp = criterion(vp[:, :, 0], target_data[:, :, 1])
            loss_imp = criterion(imp[:, :, 0], target_data[:, :, 2])

            loss = (loss_vs + loss_vp + loss_imp )
            val_loss += loss.item()
            val_vs_loss += loss_vs.item()
            val_vp_loss += loss_vp.item()
            val_imp_loss += loss_imp.item()

        writer.add_scalars('Loss', {'val_loss': val_loss / len(dataloader_val),
                                    }, epoch)
        writer.add_scalars('vs_Loss', {'val': val_vs_loss / len(dataloader_val),
                                        }, epoch)
        writer.add_scalars('vp_Loss', {'val': val_vp_loss / len(dataloader_val),
                                       }, epoch)
        writer.add_scalars('imp_Loss', {'val': val_imp_loss / len(dataloader_val),
                                        }, epoch)
        ##Model save   auto save the model of lowest loss of validation set as PLE-best
        if (val_loss/len(dataloader_val) < min_loss):
            min_loss =(val_loss / len(dataloader_val))
            torch.save(in_model.state_dict(), f'D:/Code/python/多属性反演/c1l1_mini_pt/SEAM/多属性+半监督/PLE-test')
            print(epoch,(val_loss / len(dataloader_val)))
        # if ((epoch%10) == 0):
        torch.save(in_model.state_dict(), f'D:/Code/python/多属性反演/c1l1_mini_pt/SEAM/多属性+半监督/PLE{epoch}')

writer.close()
