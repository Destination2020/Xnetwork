import os 
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import yaml
import torch.nn as nn
#from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np 
import Draw_line
import torch.nn.functional as F
import math
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
import sys

#with open("D:/Work_files/EUV-OCT/XRR/XRR_Software/DeepLearning/XRR_FCNN/FCNN3/config.yaml", "r") as file:
#   config = yaml.safe_load(file)
XRR_size = 621
Fuzzy_SLD_size = 6
Mat_size = 13
batch_size = 64
num_epochs = 50
min_thi = [5, 5]
max_thi = [15, 15]
min_rou = [0.0, 0.0, 0.0]
max_rou = [0.5 , 1.0, 1.0]
min_real_den = [5e-4, 5e-4, 5e-4]
max_real_den = [1e-2, 1e-2, 1e-2]
max_imag_den = [1e-3, 1e-3, 1e-4]
min_imag_den = [5e-5, 5e-5, 1e-5]
min_sigma_deg = [0.015]
max_sigma_deg = [0.025]
min_angle_offset = [-0.01]
max_angle_offset = [0.01]

#device = torch.device("cuda")




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=[1, 2], default=1)
    parser.add_argument("--train_dir", type=str, default="")
    parser.add_argument("--test_dir", type=str, default="")
    parser.add_argument("--pretrained_path", type=str, default="")
    parser.add_argument("--num_epochs", type=int, default=num_epochs)
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"python_executable: {sys.executable}")
    print(f"torch_version: {torch.__version__}")
    print(f"torch_cuda_version: {torch.version.cuda}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda_device_count: {torch.cuda.device_count()}")
        print(f"cuda_device_name: {torch.cuda.get_device_name(0)}")
    stage = args.stage
    default_test_dir = f"D:/Work_files/EUV-OCT/XRR/XRR_Software/DeepLearning/data/test_data/test_exp_stage{stage}"
    default_train_dir = f"D:/Work_files/EUV-OCT/XRR/XRR_Software/DeepLearning/data/train_data/train_exp_stage{stage}"
    test_dir = args.test_dir if args.test_dir else default_test_dir
    train_dir = args.train_dir if args.train_dir else default_train_dir
    train_dataset = XRRDataset(train_dir)
    test_dataset = XRRDataset(test_dir)
    train_epochs = args.num_epochs
    learning_rate = 0.0007 if stage == 1 else 0.0001
    model_dir = r"D:\Work_files\EUV-OCT\XRR\XRR_Software\DeepLearning\XRR_FCNN\AEExperiment"
    
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, drop_last = True)
    test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False, drop_last = True)

    XRR_net = XRRNet()
    if stage == 2 and args.pretrained_path and os.path.exists(args.pretrained_path):
        XRR_net.load_state_dict(torch.load(args.pretrained_path, map_location='cpu'))
    if torch.cuda.is_available():
        XRR_net.cuda()
    #loss_fn = nn.MSELoss()
    loss_fn = nn.L1Loss()
    
    #loss_fn = nn.SmoothL1Loss(beta=1)

    if torch.cuda.is_available():
        loss_fn = loss_fn.cuda()
    optimizer = torch.optim.AdamW(XRR_net.parameters(), lr = learning_rate, weight_decay=1e-4)
    #scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, min_lr=1e-6)
    scheduler = CosineAnnealingLR(optimizer, T_max = train_epochs)

    #total_sum = torch.tensor(0.0, dtype=torch.float64)
    total_num = torch.tensor(0.0, dtype=torch.float64)
    total_std = torch.tensor(0.0, dtype=torch.float64)
    '''
    XRR_mean = 0
    for batch_XRR, batch_Mat in train_dataloader:
        #noise = torch.normal(0, 1e-11, size = batch_XRR.size())   #add noise 
        #batch_XRR = batch_XRR + noise  
        batch_XRR = Log_XRR(batch_XRR)
        total_num += batch_XRR.numel()
        delta = batch_XRR - XRR_mean
        XRR_mean += torch.sum(delta)/total_num
        #print(f"total_num:{total_num}")
    for batch_XRR, batch_Mat in train_dataloader:
        batch_std =  (Log_XRR(batch_XRR)-XRR_mean)**2
        total_std += torch.sum(batch_std)
    XRR_std = torch.sqrt(total_std/total_num)
    XRR_mean = XRR_mean.to(torch.float32)
    XRR_std = XRR_std.to(torch.float32)
    X_mean = [XRR_mean, XRR_std]
    np.savetxt(f'./data/train_data/mean.txt', X_mean)
    '''
    Total_Loss, Test_Loss = [], []
    # Create a timestamp string for this training session
    session_timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    for epoch in range(train_epochs):
        XRR_net.train()
        total_loss = 0
        valid_batches = 0
        for batch_XRR, batch_Fuzzy_SLD, batch_Mat in train_dataloader:
            batch_Mat = Normalize_Mat(batch_Mat)
            if not torch.isfinite(batch_XRR).all() or not torch.isfinite(batch_Mat).all():
                continue
            if torch.cuda.is_available():
                batch_XRR = batch_XRR.cuda()
                batch_Fuzzy_SLD = batch_Fuzzy_SLD.cuda()
                batch_Mat = batch_Mat.cuda()
            output = XRR_net(batch_XRR, batch_Fuzzy_SLD)
            if not torch.isfinite(output).all():
                continue
            loss = loss_fn(output, batch_Mat)
            if not torch.isfinite(loss):
                continue
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(XRR_net.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            valid_batches += 1
        scheduler.step()
        train_epoch_loss = total_loss/max(valid_batches, 1)
        Total_Loss.append(train_epoch_loss)
        if epoch >= 1:
            if Total_Loss[epoch] < Total_Loss[epoch-1] :
                torch.save(XRR_net.state_dict(), os.path.join(model_dir, f"XRR_net_stage{stage}_train_{session_timestamp}.pth"))
        print(f"第{epoch+1}轮\tloss: {train_epoch_loss}")

        XRR_net.eval()
        with torch.no_grad():
            test_loss = 0
            valid_test_batches = 0
            for batch_XRR, batch_Fuzzy_SLD, batch_Mat in test_dataloader:
                batch_Mat = Normalize_Mat(batch_Mat)
                if not torch.isfinite(batch_XRR).all() or not torch.isfinite(batch_Mat).all():
                    continue
                if torch.cuda.is_available():
                    batch_XRR = batch_XRR.cuda()
                    batch_Fuzzy_SLD = batch_Fuzzy_SLD.cuda()
                    batch_Mat = batch_Mat.cuda()
                output = XRR_net(batch_XRR, batch_Fuzzy_SLD)
                if not torch.isfinite(output).all():
                    continue
                loss = loss_fn(output, batch_Mat)
                if not torch.isfinite(loss):
                    continue
                test_loss += loss.item()
                valid_test_batches += 1
        test_epoch_loss = test_loss/max(valid_test_batches, 1)
        Test_Loss.append(test_epoch_loss)
        if epoch >= 1:
            if Test_Loss[epoch] <= min(Test_Loss):
                torch.save(XRR_net.state_dict(), os.path.join(model_dir, f"XRR_net_stage{stage}_test_{session_timestamp}.pth"))
        print(f"整体测试集上的Loss:{test_epoch_loss}")
    Draw_line.Draw_line((range(train_epochs), Total_Loss, Test_Loss), xlabel=['epoch'],
                                ylabel=['Loss'], sub_title=[f'MLP-10000'],  
                                legend = [{'loc': 'upper right', 'fontsize': 10, 'title': 'Calculation method'}],
                                label=[['Train', 'Test']])
    torch.save(XRR_net.state_dict(), os.path.join(model_dir, f"XRR_net_stage{stage}_final_{session_timestamp}.pth"))



def Log_XRR(data_XRR):
    data_XRR = torch.clamp(data_XRR, min=1e-10)
    outputs = torch.log10(data_XRR + 1e-11)
    outputs = torch.nan_to_num(outputs, nan=-10.0, posinf=10.0, neginf=-10.0)
    return outputs

def Normalize_XRR(data_XRR, mean_XRR, std_XRR):             
 
    outputs = (data_XRR - mean_XRR)/std_XRR                 #Z-score standardization

    return outputs

def Normalize_Mat(data_Mat):
    min_rangeMat = torch.tensor(min_thi + min_rou + min_real_den + min_imag_den + min_sigma_deg + min_angle_offset, dtype = torch.float)
    max_rangeMat = torch.tensor(max_thi + max_rou + max_real_den + max_imag_den + max_sigma_deg + max_angle_offset, dtype = torch.float)
    data_Mat = (data_Mat - min_rangeMat)/(max_rangeMat - min_rangeMat)
    return data_Mat

def Normalize_Fuzzy_SLD(data_Fuzzy):
    min_rangeFuzzy = torch.tensor(min_real_den + min_imag_den, dtype = torch.float)
    max_rangeFuzzy = torch.tensor(max_real_den + max_imag_den, dtype = torch.float)
    data_Fuzzy = (data_Fuzzy - min_rangeFuzzy)/(max_rangeFuzzy - min_rangeFuzzy)
    return data_Fuzzy





class XRRDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.files = sorted(os.listdir(data_dir))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.files[idx])
        data = np.loadtxt(file_path)
        XRR_data = data[:XRR_size]
        Fuzzy_SLD_data = data[XRR_size:XRR_size+Fuzzy_SLD_size]
        Mat_data = data[-Mat_size:]
        XRR_data = np.nan_to_num(XRR_data, nan=1e-10, posinf=1e-10, neginf=1e-10)
        XRR_data = np.maximum(XRR_data, 1e-10)
        Mat_data = np.nan_to_num(Mat_data, nan=0.0, posinf=0.0, neginf=0.0)
        XRR_data = Log_XRR(torch.tensor(XRR_data, dtype = torch.float))
        Fuzzy_SLD_data = Normalize_Fuzzy_SLD(torch.tensor(Fuzzy_SLD_data, dtype = torch.float))
        Mat_data = torch.tensor(Mat_data, dtype = torch.float)
        return XRR_data, Fuzzy_SLD_data, Mat_data

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

       


if __name__ == '__main__':
    main()
