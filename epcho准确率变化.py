import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.metrics import mean_squared_error as compare_mse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from torch.utils import data
from Network import lstm_1, cnn_lstm, c1l1, c1l1_mini
from Network.lstm_1 import BidirectionalLSTM
from utils import z_score_normalization, Seq2SeqDataset_1, pearson_correlation

mean = [1932.1715,3612.8735,8298.253]
std = [920.264,1103.41710,3129.3345]
data_range=[2965.5173,3310.0,12338.255]
#实际

Model = "Seam"
model_imp = c1l1_mini.BidirectionalLSTM(input_size=1, hidden_size=16,  output_size=1,num_layers=1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_imp.to(device)
imp=np.load(f'D:/Code/python/多属性反演/Model_data/Npy/{Model}/vp.npy')#改
imp=imp[:-1,:]
seis=np.load(f'D:/Code/python/多属性反演/Synthetic seismic records/{Model}/30Hz_1000sample.npy')
seis=z_score_normalization(seis.T)
seis=seis[:,:,np.newaxis]
seis=torch.tensor(seis).to(device).float()
dataset=Seq2SeqDataset_1(seis)
train_loader = data.DataLoader(dataset, batch_size=512, shuffle=False)
q = 0

# start_index = 0
# end_index = 499
# #
# for model_index in range(start_index, end_index + 1, 1):
#     # Load the saved model parameters
#     model_path = f'D:/Code/python/多属性反演/lstm_pt/SEAM/单属性+半监督/imp{model_index}'
#     model_imp.load_state_dict(torch.load(model_path, map_location=device))
#     model_imp.eval()
#
#     pre_imp = torch.empty(0).to(device)
#
#     # Iterate over the DataLoader
#     for batch in train_loader:
#         batch = batch.to(device)
#         with torch.no_grad():
#             output = model_imp(batch)
#         pre_imp = torch.cat((pre_imp, output), dim=0)
#
#     pre_imp = (pre_imp) * std[2] + mean[2]
#
#     p = compare_psnr(imp, pre_imp.squeeze(-1).cpu().numpy().T, data_range=data_range[2])
#     s = compare_ssim(imp, pre_imp.squeeze(-1).cpu().numpy().T, data_range=data_range[2])
#     m = np.sqrt(compare_mse(imp, pre_imp.squeeze(-1).cpu().numpy().T))
#     print(f"{model_index}    {p:.3f}     {s:.3f}    {m:.4f}")
model_path = f'D:/Code/python/多属性反演/c1l1_mini_pt/SEAM/单属性/vp-best'#改
model_imp.load_state_dict(torch.load(model_path, map_location=device))
model_imp.eval()

pre_imp = torch.empty(0).to(device)

for batch in train_loader:
    batch = batch.to(device)
    with torch.no_grad():
        output = model_imp(batch)
    pre_imp = torch.cat((pre_imp, output), dim=0)
pre_imp = (pre_imp) * std[1] + mean[1]#改


p = compare_psnr(imp, pre_imp.squeeze(-1).cpu().numpy().T, data_range=data_range[1])#改
s = compare_ssim(imp, pre_imp.squeeze(-1).cpu().numpy().T, data_range=data_range[1])#改
m = np.sqrt(compare_mse(imp, pre_imp.squeeze(-1).cpu().numpy().T))

pre_imp= pre_imp.squeeze(-1).cpu().numpy().T
print(f" {p:.3f}     {s:.3f}    {m:.4f}")
fig1, ax1, = plt.subplots(1, 1)
im1 = ax1.imshow(imp , vmax=12500, vmin=1500  ,cmap='jet')
fig5, ax5, = plt.subplots(1, 1)
im5 = ax5.imshow(pre_imp, vmax=12500, vmin=1500   ,cmap='jet')

plt.xlabel('Trace No.')
plt.ylabel('Times')
plt.colorbar(im5, ax=ax5)
# 显示图像
fig6, ax6, = plt.subplots(1, 1)
im6 = ax6.imshow(abs(imp - pre_imp), vmax=6500, vmin=0 ,cmap='Greys')

plt.xlabel('Trace No.')
plt.ylabel('Times')
plt.colorbar(im6, ax=ax6)
# 显示图像
plt.show()

