import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data_utils
from torch.utils.tensorboard import SummaryWriter
from Network import PLE_c1l1mini
from utils import Seq2SeqDataset


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


input_train_folder = 'D:/Code/python/多属性反演/Dataset/Seam_norm_t/seismic'  ##path to change
target_train_folder = 'D:/Code/python/多属性反演/Dataset/Seam_norm_t/attribute'##path to change
input_val_folder = 'D:/Code/python/多属性反演/Dataset/val/seismic'##path to change
target_val_folder = 'D:/Code/python/多属性反演/Dataset/val/attribute'##path to change

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
#
model = PLE_c1l1mini.PLE_cl(input_size=1, hidden_size=16,  output_size=1,num_layers=1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.cuda()
optimizer = torch.optim.Adam(model.parameters(),lr=1e-2)#log2
criterion = nn.MSELoss(reduction='mean')
writer = SummaryWriter()

min_loss=10
for epoch in range(130):
    train_loss = 0.0
    train_vs_loss=0
    train_vp_loss=0
    train_imp_loss=0


    for input_data, target_data,in dataloader_train:

        input_data = input_data.float().to(device).cuda()
        target_data = target_data.float().to(device).cuda()

        vs, vp ,imp= model(input_data)
        # imp = model(input_data)

        loss_vs =criterion(vs[:,:,0], target_data[:,:,0]) ##target[:,:,0]means vs,target[:,:,1]means vp,target[:,:,2]means imp
        loss_vp = criterion(vp[:,:,0], target_data[:,:,1])
        loss_imp = criterion(imp[:,:,0], target_data[:,:,2])
        loss = (loss_vs + loss_vp  + loss_imp )/3

        train_loss +=loss.item()
        train_vs_loss +=loss_vs.item()
        train_vp_loss +=loss_vp.item()
        train_imp_loss +=loss_imp.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    writer.add_scalars('Loss', {'train_loss': train_loss/len(dataloader_train),
                                   }, epoch)
    writer.add_scalars('vs_Loss', {'train_seam': train_vs_loss / len(dataloader_train),
                                }, epoch)
    writer.add_scalars('vp_Loss', {'train_seam': train_vp_loss / len(dataloader_train),
                                }, epoch)
    writer.add_scalars('imp_Loss', {'train_seam': train_imp_loss / len(dataloader_train),
                                }, epoch)

    with torch.no_grad():  # 禁用梯度计算，以节省显存
        val_loss = 0.0
        val_vs_loss = 0
        val_vp_loss = 0
        val_imp_loss = 0

        for input_data, target_data in dataloader_val:

            input_data = input_data.float().to(device).cuda()
            target_data = target_data.float().to(device).cuda()
            vs, vp ,imp = model(input_data)
            # imp = model(input_data)
            loss_vs = criterion(vs[:, :, 0], target_data[:, :,0])
            loss_vp = criterion(vp[:, :, 0], target_data[:, :, 1])
            loss_imp = criterion(imp[:, :, 0], target_data[:, :, 2])

            loss = (loss_vs + loss_vp  + loss_imp )/3
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
        ##
        if (val_loss / len(dataloader_val) < min_loss):
            min_loss =val_loss / len(dataloader_val)
            torch.save(model.state_dict(), f'D:/Code/python/多属性反演/c1l1_mini_pt/SEAM/多属性/PLE-best')##path to save model_pt
        #     print(epoch,(val_loss / len(dataloader_val)))
        # torch.save(model.state_dict(), f'D:/Code/python/多属性反演/c1l1_mini_pt/SEAM/多属性/PLE{epoch}')
    # scheduler.step()
writer.close()
