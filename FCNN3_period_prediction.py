import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from ptflops import get_model_complexity_info
import Draw_line
import pandas as pd
from datetime import datetime

periods = 5
XRR_size = 512
Mat_size = 7
batch_size = 1
waveLength = 0.154
num_layer = 2
#Angle_init = np.geomspace(0.1, 4, 400, dtype = np.float64)
#Angle_init = np.linspace(0.1, 4, 500, dtype = np.float64)
Angle_init = np.array(list(np.linspace(0.5, 4, 512,endpoint=True, dtype = np.float64)))
Angle =  np.float64(Angle_init/180.0*np.pi)

def main():
    model = XRRNet()
    #model.load_state_dict(torch.load("D:/Work_files/EUV-OCT/XRR/XRR_Software/DeepLearning/XRR_FCNN/FCNN_LSTM3/XRR_net_train -path", map_location='cuda:0'))
    #model.load_state_dict(torch.load("D:/Work_files/EUV-OCT/XRR/XRR_Software/DeepLearning/XRR_FCNN/FCNN_LSTM3/Bigsamples/XRR_net_2_test -path", map_location='cuda:0'))
    model.load_state_dict(torch.load("D:/Work_files/EUV-OCT/XRR/XRR_Software/DeepLearning/XRR_FCNN/FCNN_LSTM3/model_save/LSTM_period/XRR_net_train -path", map_location='cuda:0'))
    model.eval()
    
    #data = np.loadtxt("D:/Work_files/EUV-OCT/XRR/XRR_Software/DeepLearning/XRR_FCNN/FCNN_LSTM3/20250728_113534.txt") #00000
    data = np.loadtxt("D:/Work_files/EUV-OCT/XRR/XRR_Software/DeepLearning/data/test_data/test3_period/0000012.txt")
    Xrr_data = data[:XRR_size]   # 前 500 行：XRR 反射率曲线
    Xrr_mat = data[-Mat_size:]
    print(f"真实结果：{Xrr_mat}")

    '''
    Xrr_mean = np.loadtxt(f'./data/train_data/mean.txt')
    xrr_pre = torch.tensor(Xrr_data, dtype=torch.float)
    xrr_pre_norm = Normalize_XRR(Log_XRR(xrr_pre), Xrr_mean[0], Xrr_mean[1])
    xrr_pre_norm = xrr_pre_norm.view(1, 200)
    '''
    
    xrr_pre = Log_XRR(torch.tensor(Xrr_data, dtype=torch.float))
    xrr_pre_norm = xrr_pre.view(1, XRR_size)

    Cover_data = []
    with torch.no_grad():
        predicted_params = model(xrr_pre_norm)

    Cover_data.append(predicted_params[0][0] * 25 + 5)
    Cover_data.append(predicted_params[0][1] * 25 + 5)
    #Cover_data.append(predicted_params[0][2] * 1.0) //roughness of the first surface
    Cover_data.append(predicted_params[0][2] * 1.0)
    Cover_data.append(predicted_params[0][3] * (2e-2 - 1e-3) + 1e-3)
    Cover_data.append(predicted_params[0][4] * (2e-2 - 1e-3) + 1e-3)
    Cover_data.append(predicted_params[0][5] * (2e-3 - 1e-5) + 1e-5)
    Cover_data.append(predicted_params[0][6] * (2e-3 - 1e-5) + 1e-5)

    Cover_data = np.array(Cover_data)
    Error_data = np.round((np.abs((Cover_data - Xrr_mat)/Xrr_mat)), 2)
    print(f"恢复结果：{Cover_data}")
    print(f"相对误差：{Error_data}")

    r1, R1, Q1 = Materials(Xrr_mat)
    r2, R2, Q2 = Materials(Cover_data)

    Draw_line.Draw_line((Q1, R1, R2), xlabel=['Q(nm^-1)'],yscale = 'log',
                                ylabel=['R'], sub_title=[f'Wavelength={waveLength}nm'],  
                                legend = [{'loc': 'upper right', 'fontsize': 10, 'title': 'Calculation method'}],
                                label=[['Real', 'Recover']]) 
    Store = input("输入1保存文件: ")
    if Store=='1':
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        Excel_data = {
            '参数': ['Thickness',' ','Roughness', 'Real(SLD)', ' ', 'Imag(SLD)', ' '],
            '真实结果': Xrr_mat,
            '恢复结果': Cover_data,
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
class XRRNet(nn.Module):
    def __init__(self, input_size = 1):
        super(XRRNet, self).__init__()
        self.hidden_size = Hidden_size
        self.num_layers = 3
        self.biLSTM = nn.LSTM(input_size, self.hidden_size, bidirectional = True, num_layers = self.num_layers, dropout = 0.2)
        self.result = nn.Sequential(
            nn.Linear(2*self.hidden_size, 100),
            nn.Sigmoid(),
            nn.Linear(100, Mat_size)
        )
        self.Attention = nn.ModuleList([MultiHeadAttention() for _ in range(6)])

    def forward(self, x):
        x = x.view(batch_size, XRR_size, 1).transpose(0, 1)
        output, (hidden_state, cell_state) = self.biLSTM(x) #output:[XRR_size, batch_size, directions*hidden_size] hidden_state:[bidirection(2)*num_layers, batch_size, hidden_size]
        hidden_state = hidden_state[-2:]
        result = hidden_state.transpose(0, 1).contiguous().view(batch_size, 1, 2*Hidden_size)
        for layer in self.Attention:
            result = layer(result, output, output)
        result = self.result(result.squeeze(1))
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
    real_den = np.array([0.0] + list(input[3:5]) * periods + [1.8865e-03])
    imag_den = np.array([0.0] + list(input[5:]) * periods + [2.4360e-05])
    n = N_angle(real_den, imag_den, waveLength)
    n = n.flatten()
    thickness = np.array([0.0] + list(input[0:2]) * periods + [0.0])
    roughness = np.array([0.0] + [input[2]] * (2 * periods -1) + [0.0])
    r, R, Q = Angle_matrix(thickness, roughness, n, Angle, waveLength)
    return r, R, Q
     
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
            p[j] = k_add/(2*k[j])*np.exp(-1.0*((k[j+1]-k[j])*roughness[j]**2)/2)
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