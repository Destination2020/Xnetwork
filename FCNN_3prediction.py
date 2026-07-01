import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import Draw_line
import pandas as pd
from datetime import datetime

XRR_size = 621
Fuzzy_SLD_size = 6
Mat_size = 13
batch_size = 1
waveLength = 0.154056
num_layer = 2
Angle_init = np.linspace(0.4000, 3.5000, 621, endpoint=True, dtype = np.float64)
Angle =  np.float64(Angle_init/180.0*np.pi)

min_real_den = [5e-4, 5e-4, 5e-4]
max_real_den = [1e-2, 1e-2, 1e-2]
max_imag_den = [1e-3, 1e-3, 1e-4]
min_imag_den = [5e-5, 5e-5, 1e-5]

def Normalize_Fuzzy_SLD(data_Fuzzy):
    min_rangeFuzzy = torch.tensor(min_real_den + min_imag_den, dtype = torch.float)
    max_rangeFuzzy = torch.tensor(max_real_den + max_imag_den, dtype = torch.float)
    data_Fuzzy = (data_Fuzzy - min_rangeFuzzy)/(max_rangeFuzzy - min_rangeFuzzy)
    return data_Fuzzy

def main():
    model = XRRNet()
    model.load_state_dict(torch.load(r"D:/Work_files/EUV-OCT/XRR/XRR_Software/DeepLearning/XRR_FCNN/AEExperiment/XRR_net_stage1_final_20260402_225426.pth", map_location='cuda:0'))
    model.eval()

    
    test_file = r"d:/Work_files/EUV-OCT/XRR/XRR_Software/DeepLearning/data/test_data/test_exp_stage1/0000611.txt"

    train_data = np.loadtxt(test_file)
    Xrr_data = train_data[:XRR_size]
    fuzzy_sld_data = train_data[XRR_size:XRR_size+Fuzzy_SLD_size]
    ground_truth = train_data[-Mat_size:]
    '''
    exp_file = r"d:/Work_files/EUV-OCT/XRR/XRR_Software/DeepLearning/XRR_FCNN/AEExperiment/doublelayerT2.csv"

    import csv
    reflectivityExperiment = []
    with open(exp_file, mode='r', newline='') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if len(row) >= 2:
                try:
                    reflectivityExperiment.append(float(row[1]))
                except ValueError:
                    continue

    Xrr_data = np.array(reflectivityExperiment)
    if len(Xrr_data) > XRR_size:
        Xrr_data = Xrr_data[:XRR_size]
    elif len(Xrr_data) < XRR_size:
        print(f"Warning: Data size {len(Xrr_data)} is smaller than model input size {XRR_size}")
        Xrr_data = np.pad(Xrr_data, (0, XRR_size - len(Xrr_data)), 'constant')

    fuzzy_real_L1 = float(input("请输入 Layer1 Real SLD 模糊值 (如 5e-4 ~ 1e-2): ") or "5e-3")
    fuzzy_real_L2 = float(input("请输入 Layer2 Real SLD 模糊值: ") or "5e-3")
    fuzzy_real_Sub = float(input("请输入 Substrate Real SLD 模糊值: ") or "2e-3")
    fuzzy_imag_L1 = float(input("请输入 Layer1 Imag SLD 模糊值 (如 5e-5 ~ 1e-3): ") or "5e-4")
    fuzzy_imag_L2 = float(input("请输入 Layer2 Imag SLD 模糊值: ") or "5e-4")
    fuzzy_imag_Sub = float(input("请输入 Substrate Imag SLD 模糊值 (如 1e-5 ~ 1e-4): ") or "5e-5")

    fuzzy_sld_data = np.array([fuzzy_real_L1, fuzzy_real_L2, fuzzy_real_Sub,
                               fuzzy_imag_L1, fuzzy_imag_L2, fuzzy_imag_Sub])

    ground_truth = np.zeros(Mat_size)
    '''
    R1 = Xrr_data

    xrr_pre = Log_XRR(torch.tensor(Xrr_data, dtype=torch.float))
    xrr_pre_norm = xrr_pre.view(1, XRR_size)
    fuzzy_sld_input = Normalize_Fuzzy_SLD(torch.tensor(fuzzy_sld_data, dtype=torch.float)).view(1, Fuzzy_SLD_size)

    Cover_data = []
    with torch.no_grad():
        predicted_params = model(xrr_pre_norm, fuzzy_sld_input)

    Cover_data.append(predicted_params[0][0] * (15 - 5) + 5) # min_thi=[5,5], max_thi=[15,15]
    Cover_data.append(predicted_params[0][1] * (15 - 5) + 5)
    Cover_data.append(predicted_params[0][2] * 0.5) # min_rou=0, max_rou=0.5
    Cover_data.append(predicted_params[0][3] * 1.0) # min_rou=0, max_rou=1.0
    Cover_data.append(predicted_params[0][4] * 1.0) # min_rou=0, max_rou=1.0
    Cover_data.append(predicted_params[0][5] * (1e-2 - 1e-3) + 1e-3) # min_real=1e-3, max_real=1e-2
    Cover_data.append(predicted_params[0][6] * (1e-2 - 1e-3) + 1e-3)
    Cover_data.append(predicted_params[0][7] * (1e-2 - 1e-3) + 1e-3)
    Cover_data.append(predicted_params[0][8] * (1e-3 - 1e-4) + 1e-4) # min_imag=1e-4, max_imag=1e-3
    Cover_data.append(predicted_params[0][9] * (1e-3 - 1e-4) + 1e-4)
    Cover_data.append(predicted_params[0][10] * (2e-5 - 5e-5) + 5e-5) # min_imag=5e-5, max_imag=2e-5 ? Wait, check Net file.
    # Net file: min_imag_den = [1e-4, 1e-4, 5e-5], max_imag_den = [1e-3, 1e-3, 2e-5]
    # So index 10 (last imag) is: val * (2e-5 - 5e-5) + 5e-5 ? 
    # Usually max > min. Let's check Net file again.
    # min_imag_den = [1e-4, 1e-4, 5e-5]
    # max_imag_den = [1e-3, 1e-3, 2e-5] -> 2e-5 < 5e-5? This might be inverted in Net file or intention.
    # Assuming user intended range [2e-5, 5e-5] or [5e-5, 2e-5].
    # Let's use the values from Net file literally: val * (max - min) + min
    
    Cover_data.append(predicted_params[0][11] * (0.025 - 0.015) + 0.015)
    Cover_data.append(predicted_params[0][12] * (0.01 - (-0.01)) + (-0.01))

    Cover_data = np.array(Cover_data)
    print(f"真实结果：{ground_truth}")
    print(f"预测结果：{Cover_data}")
    Error_data = np.abs((Cover_data - ground_truth) / (ground_truth + 1e-10))
    print(f"相对误差：{Error_data}")
    print(f"平均相对误差：{np.mean(Error_data):.4f}")
    
    # Calculate recovered curve
    # Cover_data includes sigma_deg at the end (index 11)
    # Materials function needs to be updated to handle sigma_deg or use it externally
    # Current Materials function: uses global waveLength, Angle
    # And calls Angle_matrix
    
    R2, fit_params = Reconstruct_R2(Cover_data, R1)
    print(f"R2拟合补偿参数：offset={fit_params[0]:.6f} deg, scale={fit_params[1]:.6f}, background={fit_params[2]:.3e}")
    
    Draw_line.Draw_line((Angle_init, R1, R2), xlabel=['ang(°))'],yscale = 'log',
                                ylabel=['R'], sub_title=[f'Wavelength={waveLength}nm'],  
                                legend = [{'loc': 'upper right', 'fontsize': 10, 'title': 'Comparison'}],
                                label=[['Experiment', 'Predicted']])
    Store = input("输入1保存文件: ")
    if Store=='1':
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        params_list = ['Thickness (L1)', 'Thickness (L2)', 'Roughness (Sub)', 'Roughness (L1)', 'Roughness (L2)', 
                       'Real (L1)', 'Real (L2)', 'Real (Sub)', 
                       'Imag (L1)', 'Imag (L2)', 'Imag (Sub)', 'Sigma (deg)', 'Angle offset (deg)']
        
        Excel_data = {
            '参数': params_list,
            '真实结果': ground_truth,
            '预测结果': Cover_data,
            '相对误差': Error_data
        }
        df = pd.DataFrame(Excel_data)
        df.to_excel(f'{timestamp}.xlsx', index = False)
        print(f"文件已经保存为{timestamp}.xlsx")

def Log_XRR(data_XRR):
    
    outputs = torch.log10(data_XRR + 1e-11)                 #log transform

    return outputs

def Normalize_XRR(data_XRR, mean_XRR, std_XRR):             
 
    outputs = (data_XRR - mean_XRR)/std_XRR                 #Z-score standardization

    return outputs

Hidden_size = 256
Fuzzy_encoder_output_size = 128

class XRRNet(nn.Module):
    def __init__(self, xrr_input_size=1, fuzzy_sld_input_size=Fuzzy_SLD_size):
        super(XRRNet, self).__init__()
        self.hidden_size = Hidden_size
        self.num_layers = 3
        self.biLSTM = nn.LSTM(xrr_input_size, self.hidden_size, bidirectional=True, num_layers=self.num_layers, dropout=0.2)
        self.fuzzy_encoder = nn.Sequential(
            nn.Linear(fuzzy_sld_input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, Fuzzy_encoder_output_size)
        )
        self.result = nn.Sequential(
            nn.Linear(2*self.hidden_size + Fuzzy_encoder_output_size, 256),
            nn.Sigmoid(),
            nn.Linear(256, 128),
            nn.Sigmoid(),
            nn.Linear(128, Mat_size)
        )
        self.Attention = nn.ModuleList([MultiHeadAttention() for _ in range(6)])

    def forward(self, xrr_input, fuzzy_sld_input):
        batch_size = xrr_input.size(0)
        XRR_size = xrr_input.size(1)
        xrr_seq = xrr_input.view(batch_size, XRR_size, 1).transpose(0, 1)
        output, (hidden_state, cell_state) = self.biLSTM(xrr_seq)
        hidden_state = hidden_state[-2:]
        xrr_features = hidden_state.transpose(0, 1).contiguous().view(batch_size, 1, 2*Hidden_size)
        for layer in self.Attention:
            xrr_features = layer(xrr_features, output, output)
        xrr_features = xrr_features.squeeze(1)
        fuzzy_features = self.fuzzy_encoder(fuzzy_sld_input)
        combined = torch.cat([xrr_features, fuzzy_features], dim=1)
        result = self.result(combined)
        return result

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads = 3):
        super(MultiHeadAttention, self).__init__()
        self.d_Q = 64
        self.d_K = 64
        self.d_V = 64
        self.heads = n_heads
        self.fc_Q = nn.Linear(2*Hidden_size, self.heads*self.d_Q)       # Q: [batch_size, bidirection(2), hidden_size]
        self.fc_K = nn.Linear(2*Hidden_size, self.heads*self.d_K)       #K V :[batch_size, XRR_size, directions*hidden_size]
        self.fc_V = nn.Linear(2*Hidden_size, self.heads*self.d_V)
        self.recover = nn.Linear(self.heads*self.d_V, 2*Hidden_size)
        self.LayerNorm1 = nn.LayerNorm(2*Hidden_size)
        self.LayerNorm2 = nn.LayerNorm(2*Hidden_size)
        self.PoswiseFeedforward = nn.Sequential(
            nn.Linear(2*Hidden_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2*Hidden_size)
        )

    def ScaleDotProductAttention(self, Q, K, V):
        scores = torch.matmul(Q, K.transpose(-1, -2))/np.sqrt(self.d_K)
        attn = F.softmax(scores, dim = -1)
        output = torch.matmul(attn, V)
        return output
        
        
    def forward(self, Q_input, K_input, V_input):
        residual = Q_input
        Q = self.fc_Q(residual).view(batch_size,  -1, self.heads, self.d_Q).transpose(1, 2)
        K = self.fc_K(K_input.transpose(0, 1)).view(batch_size,  -1, self.heads, self.d_Q).transpose(1, 2)
        V = self.fc_V(V_input.transpose(0, 1)).view(batch_size,  -1, self.heads, self.d_Q).transpose(1, 2)
        output = self.recover(self.ScaleDotProductAttention(Q, K, V).transpose(1, 2).contiguous().view(batch_size, -1, self.heads*self.d_V))
        output = self.LayerNorm1(output + residual)
        output = self.LayerNorm2(output + self.PoswiseFeedforward(output))
        return output


def Materials(input):
    real_den = np.array([0.0] + list(input[5:8]))
    imag_den = np.array([0.0] + list(input[8:11]))
    n = N_angle(real_den, imag_den, waveLength)
    n = n.flatten()
    thickness = np.array([0.0] + list(input[0:2]) + [0.0])
    roughness = input[2:5]
    r, R, Q = Angle_matrix(thickness, roughness, n, Angle, waveLength)
    return r, R, Q

def Reconstruct_R2(input_data, target_curve):
    from scipy.ndimage import gaussian_filter1d
    _, R_raw, _ = Materials(input_data)
    step_size = Angle_init[1] - Angle_init[0]
    conv_sigma = input_data[11] / step_size
    R_conv = gaussian_filter1d(R_raw, sigma=conv_sigma, mode='nearest')
    eps = 1e-10
    y = np.maximum(np.asarray(target_curve, dtype=np.float64), eps)
    offset_center = float(input_data[12])
    offset_min = max(-0.01, offset_center - 0.003)
    offset_max = min(0.01, offset_center + 0.003)
    offsets = np.linspace(offset_min, offset_max, 61)
    tail_mask = Angle_init > 2.8
    if np.any(tail_mask):
        tail_level = float(np.median(y[tail_mask]))
    else:
        tail_level = float(np.median(y))
    bg_min = max(1e-10, 0.25 * tail_level)
    bg_max = max(bg_min * 5.0, float(np.percentile(y, 35)))
    bg_grid = np.linspace(bg_min, bg_max, 80)
    best_err = np.inf
    best_curve = np.maximum(R_conv, eps)
    best_params = (offset_center, 1.0, bg_min)
    for off in offsets:
        shifted = np.interp(Angle_init, Angle_init + off, R_conv, left=R_conv[0], right=R_conv[-1])
        shifted = np.maximum(shifted, eps)
        denom = float(np.dot(shifted, shifted)) + 1e-20
        for bg in bg_grid:
            scale = float(max(np.dot(shifted, y - bg) / denom, 1e-6))
            pred = np.maximum(scale * shifted + bg, eps)
            err = np.mean((np.log10(pred) - np.log10(y))**2)
            if err < best_err:
                best_err = err
                best_curve = pred
                best_params = (float(off), scale, float(bg))
    return best_curve, best_params
     
def N_angle(Re_sld,Im_sld,waveLength):
        k0 = np.float64(np.power(waveLength,2)/(2*np.pi))
        n = 1 - k0*Re_sld + 1j*k0*Im_sld
        return np.array(n)     


def Angle_matrix(thickness, roughness, n, incidence_angle, waveLength):
    lay_num = len(thickness)
    k0 = 2*np.pi/waveLength
    len_ang = len(incidence_angle)
    #calculate k of ecah layer of each incidence angle
    k = np.zeros((lay_num,len_ang), dtype=complex)              
    for j in range(lay_num):
            k[j] = -np.sqrt((k0*n[j])**2-(k0*n[0]*np.cos(incidence_angle))**2)

    m = np.zeros((lay_num-1,len_ang), dtype=complex)
    p = np.zeros((lay_num-1,len_ang), dtype=complex)
    ek = np.zeros((lay_num-1,len_ang), dtype=complex)
    for j in range(lay_num-1):
            '''
            p[j] = (Layer_n.Layer_para[j+1].n**2*k[j]+Layer_n.Layer_para[j].n**2*k[j+1])/(2*Layer_n.Layer_para[j+1].n**2*k[j])
            m[j] = (Layer_n.Layer_para[j+1].n**2*k[j]-Layer_n.Layer_para[j].n**2*k[j+1])/(2*Layer_n.Layer_para[j+1].n**2*k[j])
            '''
            k_add = k[j]+k[j+1]
            k_sub = k[j]-k[j+1]
            p[j] = k_add/(2*k[j])*np.exp(-1.0*(k_sub*roughness[j])**2/2)
            m[j] = k_sub/(2*k[j])*np.exp(-1.0*(k_add*roughness[j])**2/2)
            ek[j] = 1j*k[j]*thickness[j]
    h1 = np.exp(-ek)
    h2 = 1/h1
    M1T = np.array([[p, m], [m, p]])
    M1 = np.transpose(M1T, [2, 3, 0, 1])
    zeros = np.zeros((lay_num-1, len_ang), dtype = complex)
    H1T = np.array([[h1, zeros], [zeros, h2]])
    H1 = np.transpose(H1T, [2, 3, 0, 1])
    tra_Matrix = np.matmul(H1, M1)
    transferMatrix = np.tile(np.array([[1,0], [0,1]], dtype = complex), (len_ang, 1, 1))
    for j in range(lay_num-1):
        transferMatrix = np.matmul(transferMatrix, tra_Matrix[j])

    r = transferMatrix[:, 0, 1]/transferMatrix[:, 1, 1]
    R = np.abs(r)**2
    Q = 0.2*k0*np.sin(incidence_angle)*np.abs(n[0])  #Dividing by 10 is for the unit as A^-1
    return r,R,Q



if __name__ == '__main__':
    main()
