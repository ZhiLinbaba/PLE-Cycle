import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.metrics import mean_squared_error as compare_mse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from torch.utils import data

from Network import lstm_1, c1l1_mini, PLE_c1l1mini

from utils import z_score_normalization, Seq2SeqDataset_1

mean = [1932.1715,3612.8735,8298.253]
std = [920.264,1103.41710,3129.3345]
data_range=[2965.5173,3310.0,12338.255]

Model = "Seam"


model_imp = PLE_c1l1mini.PLE_cl(input_size=1, hidden_size=16,  output_size=1,num_layers=1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_imp.to(device)
imp=np.load(f'D:/Code/python/多属性反演/Model_data/Npy/{Model}/imp.npy')
vp=np.load(f'D:/Code/python/多属性反演/Model_data/Npy/{Model}/vp.npy')
vs=np.load(f'D:/Code/python/多属性反演/Model_data/Npy/{Model}/vs.npy')
imp=imp[:-1,:]
vp=vp[:-1,:]
vs=vs[:-1,:]
seis=np.load(f'D:/Code/python/多属性反演/Synthetic seismic records/{Model}/30Hz_1000sample.npy')
seis=z_score_normalization(seis.T)
seis=seis[:,:,np.newaxis]
seis=torch.tensor(seis).to(device).float()
dataset=Seq2SeqDataset_1(seis)
train_loader = data.DataLoader(dataset, batch_size=512, shuffle=False)
q = 0

model_path = f'D:/Code/python/多属性反演/c1l1_mini_pt/SEAM/多属性/PLE-best'
model_imp.load_state_dict(torch.load(model_path, map_location=device))
model_imp.eval()
# Initialize an empty tensor to store the predictions
pre_imp = torch.empty(0).to(device)
pre_vp = torch.empty(0).to(device)
pre_vs = torch.empty(0).to(device)
# Iterate over the DataLoader
for batch in train_loader:
    batch = batch.to(device)  # Move the batch to the same device as the model
    with torch.no_grad():  # Disable gradient calculation for inference
        output, output_1, output_2 = model_imp(batch)
    pre_imp = torch.cat((pre_imp, output_2), dim=0)  # Append the output to pre_imp
    pre_vp = torch.cat((pre_vp, output_1), dim=0)
    pre_vs = torch.cat((pre_vs, output), dim=0)

pre_imp = (pre_imp) * std[2] + mean[2]
pre_vp = (pre_vp) * std[1] + mean[1]
pre_vs = (pre_vs) * std[0] + mean[0]

p_vs = compare_psnr(vs, pre_vs.squeeze(-1).cpu().numpy().T, data_range=data_range[0])
s_vs = compare_ssim(vs, pre_vs.squeeze(-1).cpu().numpy().T, data_range=data_range[0])
m_vs = np.sqrt(compare_mse(vs, pre_vs.squeeze(-1).cpu().numpy().T))

p_vp = compare_psnr(vp, pre_vp.squeeze(-1).cpu().numpy().T, data_range=data_range[1])
s_vp = compare_ssim(vp, pre_vp.squeeze(-1).cpu().numpy().T, data_range=data_range[1])
m_vp = np.sqrt(compare_mse(vp, pre_vp.squeeze(-1).cpu().numpy().T))

p_imp = compare_psnr(imp, pre_imp.squeeze(-1).cpu().numpy().T, data_range=data_range[2])
s_imp = compare_ssim(imp, pre_imp.squeeze(-1).cpu().numpy().T, data_range=data_range[2])
m_imp = np.sqrt(compare_mse(imp, pre_imp.squeeze(-1).cpu().numpy().T))

pre_vs= pre_imp.squeeze(-1).cpu().numpy().T
pre_vp= pre_imp.squeeze(-1).cpu().numpy().T
pre_imp= pre_imp.squeeze(-1).cpu().numpy().T
print(
    f"    {p_vs:.3f}     {s_vs:.3f}    {m_vs:.4f}     {p_vp:.3f}     {s_vp:.3f}    {m_vp:.4f}     {p_imp:.3f}     {s_imp:.3f}    {m_imp:.4f}")
fig1, ax1, = plt.subplots(1, 1)
im1 = ax1.imshow(imp , vmax=12500, vmin=1500  ,cmap='jet')
fig5, ax5, = plt.subplots(1, 1)
im5 = ax5.imshow(pre_imp, vmax=12500, vmin=1500   ,cmap='jet')

plt.xlabel('Trace No.')
plt.ylabel('Times')
plt.colorbar(im5, ax=ax5)
plt.colorbar(im1, ax=ax1)
# 显示图像
fig6, ax6, = plt.subplots(1, 1)
im6 = ax6.imshow(abs(imp - pre_imp), vmax=6500, vmin=0 ,cmap='Greys')

plt.xlabel('Trace No.')
plt.ylabel('Times')
plt.colorbar(im6, ax=ax6)
# 显示图像
plt.show()

