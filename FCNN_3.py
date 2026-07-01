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

#with open("D:/Work_files/EUV-OCT/XRR/XRR_Software/DeepLearning/XRR_FCNN/FCNN3/config.yaml", "r") as file:
#   config = yaml.safe_load(file)
XRR_size = 512
Mat_size = 11
batch_size = 64
num_epochs = 100
min_thi = [20, 20]
max_thi = [80, 80]
min_rou = [0.0, 0.0, 0.0]
max_rou = [0.5 , 1.0, 1.0]
min_real_den = [1e-3, 1e-3, 1e-3]
max_real_den = [2e-2, 2e-2, 2e-2]
max_imag_den = [2e-3, 2e-3, 2e-3]
min_imag_den = [2e-5, 2e-5, 2e-5]

#device = torch.device("cuda")




def main():
    test_dir = "D:/Work_files/EUV-OCT/XRR/XRR_Software/DeepLearning/data/test_data/test3_512"
    train_dir = "D:/Work_files/EUV-OCT/XRR/XRR_Software/DeepLearning/data/train_data/train3_512"
    train_dataset = XRRDataset(train_dir)
    test_dataset = XRRDataset(test_dir)
    learning_rate = 0.001
    
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, drop_last = True)
    test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False, drop_last = True)

    XRR_net = XRRNet()
    #XRR_net.to(device)
    if torch.cuda.is_available():
        XRR_net.cuda()
    #loss_fn = nn.MSELoss()
    loss_fn = nn.L1Loss()
    
    #loss_fn = nn.SmoothL1Loss(beta=1)

    if torch.cuda.is_available():
        loss_fn = loss_fn.cuda()
    optimizer = torch.optim.AdamW(XRR_net.parameters(), lr = learning_rate, weight_decay=1e-4)
    #scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, min_lr=1e-6)
    scheduler = CosineAnnealingLR(optimizer, T_max = num_epochs)

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
    for epoch in range(num_epochs):
        XRR_net.train()
        total_loss = 0
        for batch_XRR, batch_Mat in train_dataloader:
            batch_Mat = Normalize_Mat(batch_Mat)
            if torch.cuda.is_available():
                batch_XRR = batch_XRR.cuda()
                batch_Mat = batch_Mat.cuda()
            output = XRR_net(batch_XRR)
            loss = loss_fn(output, batch_Mat)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            #print(loss.item())
        Total_Loss.append(total_loss/len(train_dataloader))
        if epoch >= 1:
            if Total_Loss[epoch] < Total_Loss[epoch-1] :
                torch.save(XRR_net.state_dict(), "D:/Work_files/EUV-OCT/XRR/XRR_Software/DeepLearning/XRR_FCNN/FCNN_LSTM3/model_save/XRR_net_train -path")
        print(f"第{epoch+1}轮\tloss: {total_loss/len(train_dataloader)}")

        XRR_net.eval()
        with torch.no_grad():
            test_loss = 0
            for batch_XRR, batch_Mat in test_dataloader:
                batch_Mat = Normalize_Mat(batch_Mat)
                #print(f"测试集的Mat:{batch_Mat}")
                if torch.cuda.is_available():
                    batch_XRR = batch_XRR.cuda()
                    batch_Mat = batch_Mat.cuda()
                output = XRR_net(batch_XRR)
                #print(f"Test的输出:{output}")
                loss = loss_fn(output, batch_Mat)
                test_loss += loss.item()
        Test_Loss.append(test_loss/len(test_dataloader))
        if epoch >= 1:
            if Test_Loss[epoch] <= min(Test_Loss):
                torch.save(XRR_net.state_dict(), "D:/Work_files/EUV-OCT/XRR/XRR_Software/DeepLearning/XRR_FCNN/FCNN_LSTM3/model_save/XRR_net_test -path")
        print(f"整体测试集上的Loss:{test_loss/len(test_dataloader)}")
    Draw_line.Draw_line((range(num_epochs), Total_Loss, Test_Loss), xlabel=['epoch'],
                                ylabel=['Loss'], sub_title=[f'MLP-10000'],  
                                legend = [{'loc': 'upper right', 'fontsize': 10, 'title': 'Calculation method'}],
                                label=[['Train', 'Test']])
    torch.save(XRR_net.state_dict(), "D:/Work_files/EUV-OCT/XRR/XRR_Software/DeepLearning/XRR_FCNN/FCNN_LSTM3/model_save/XRR_net_final -path")



def Log_XRR(data_XRR):
    
    outputs = torch.log10(data_XRR + 1e-11) + torch.tensor(np.random.normal(0.0, 0.01, size = XRR_size), dtype = torch.float32)              #log transform

    return outputs

def Normalize_XRR(data_XRR, mean_XRR, std_XRR):             
 
    outputs = (data_XRR - mean_XRR)/std_XRR                 #Z-score standardization

    return outputs

def Normalize_Mat(data_Mat):
    min_rangeMat = torch.tensor((min_thi + min_rou + min_real_den + min_imag_den), dtype = torch.float)
    max_rangeMat = torch.tensor(max_thi + max_rou + max_real_den + max_imag_den, dtype = torch.float)
    data_Mat = (data_Mat - min_rangeMat)/(max_rangeMat - min_rangeMat)
    return data_Mat





class XRRDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.files = sorted(os.listdir(data_dir))
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.files[idx])    #input R*Q**4
        data = np.loadtxt(file_path)
        XRR_data = data[:XRR_size]   # 前 500 行：XRR 反射率曲线
        Mat_data = data[-Mat_size:]  # 后 11 行：材料参数数据
        '''
        df = np.loadtxt.read_csv(file_path, header = None)
        XRR_data = df.iloc[:, :XRR_size].values
        Mat_data = df.iloc[:, -Mat_size:].values
        '''
        XRR_data = Log_XRR(torch.tensor(XRR_data, dtype = torch.float))
        Mat_data = torch.tensor(Mat_data, dtype = torch.float)
        return XRR_data, Mat_data

'''Resnet
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(in_channels, out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(out_channels, out_channels)
        
        # 确保输入和输出维度一致
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.ReLU()
            )
    
    def forward(self, x):
        # 残差连接：输入 x 与通过两层线性变换后的结果相加
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out += self.shortcut(x)  # 跳过连接
        out = self.relu(out)
        return out

class XRRNet(nn.Module):
    def __init__(self):
        super(XRRNet, self).__init__()
        self.layer1 = ResidualBlock(XRR_size, 1000)
        self.layer2 = ResidualBlock(1000, 2000)
        self.layer3 = ResidualBlock(2000, 1000)
        self.layer4 = ResidualBlock(1000, 200)
        self.output_layer = nn.Linear(200, Mat_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.output_layer(x)=
        return x
'''

'''
class XRRNet(nn.Module):
    def __init__(self, seq_len=XRR_size, feature_size=1, embed_size=64, Mat_size=Mat_size, num_heads=8, num_layers=6, dropout=0.1):
        super(XRRNet, self).__init__()

        # 输入的 XRR 曲线点通过一个线性层映射到嵌入空间
        self.embedding = nn.Linear(feature_size, embed_size)  # 将每个点的反射率映射到 512 维嵌入空间

        # 位置编码
        self.positional_encoding = nn.Parameter(torch.zeros(1, seq_len, embed_size))  # 位置编码

        # Transformer 编码器层
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出层，将 Transformer 输出映射到材料参数的维度
        self.output_layer = nn.Linear(embed_size, Mat_size)  # Mat_size 为你输出的材料参数的大小

    def forward(self, x):
        # 输入形状：(batch_size, seq_len, feature_size)
        # 假设 x 的形状为 (batch_size, seq_len, 1)，即每个样本有 200 个 XRR 点，每个点有一个特征（反射率）

        # 将 XRR 曲线点的反射率映射到更高维度
        x = x.view(batch_size, XRR_size, 1)
        x = self.embedding(x)  # 输出形状：(batch_size, seq_len, embed_size)

        # 添加位置编码：加上位置编码来传递顺序信息
        x = x + self.positional_encoding  # 形状仍然是 (batch_size, seq_len, embed_size)

        # 转置输入，使其符合 Transformer 编码器的要求：(seq_len, batch_size, embed_size)
        x = x.transpose(0, 1)

        # 通过 Transformer 编码器进行处理
        x = self.transformer_encoder(x)

        # 对所有时间步的输出取平均，得到全局特征表示
        x = x.mean(dim=0)  # (batch_size, embed_size)

        # 通过输出层映射到材料参数
        x = self.output_layer(x)  # (batch_size, Mat_size)

        return x
'''



'''
    def __init__(self):
        super(XRRNet, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, stride=1, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        #self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)  
        self.drop1 = nn.Dropout(0.1)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=7, stride=1, padding=3)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.drop2 = nn.Dropout(0.1)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=7, stride=1, padding=3)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.res_conv = nn.Conv1d(64, 128, kernel_size=1)  
        self.drop3 = nn.Dropout(0.1)

        self.conv4 = nn.Conv1d(128, 256, kernel_size=7, stride=1, padding=3)
        self.bn4 = nn.BatchNorm1d(256)
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.drop4 = nn.Dropout(0.1)
        

        self.conv5 = nn.Conv1d(256, 512, kernel_size=7, stride=1, padding=3)
        self.bn5 = nn.BatchNorm1d(512)
        self.pool5 = nn.AdaptiveAvgPool1d(1)  # 自适应池化,输出 1 维
        self.drop5 = nn.Dropout(0.1)
        

        conv_output_size = 512  

        self.fc1 = nn.Linear(conv_output_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, Mat_size)

    def forward(self, x):
        x = x.unsqueeze(1)  # 调整输入维度，适配 Conv1d
        
        x = self.drop1((F.relu(self.bn1(self.conv1(x)))))
        x = self.drop2(self.pool2(F.relu(self.bn2(self.conv2(x)))))
        res = self.res_conv(x)  # 残差映射
        x = self.drop3(self.pool3(F.relu(self.bn3(self.conv3(x))+ res)))
        x = self.drop4(self.pool4(F.relu(self.bn4(self.conv4(x)))))
        x = self.drop5(self.pool5(F.relu(self.bn5(self.conv5(x)))))
        x = x.view(x.size(0), -1)   #展平成全连接输入
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  

        return x
'''


'''
    def __init__(self):                #残差连接MLP
        super(XRRNet, self).__init__()
        self.fc1 = nn.Linear(XRR_size, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 2048)
        self.bn2 = nn.BatchNorm1d(2048)
        self.fc3 = nn.Linear(2048, 1024)
        self.bn3 = nn.BatchNorm1d(1024)
        self.fc4 = nn.Linear(1024, Mat_size)  # 输出层
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x1 = self.dropout(nn.ReLU()(self.bn1(self.fc1(x))))
        x2 = self.dropout(nn.ReLU()(self.bn2(self.fc2(x1))))
        x3 = self.dropout(nn.ReLU()(self.bn3(self.fc3(x2) + x1)))  # 残差连接
        output = self.fc4(x3)  # 最后一层不加 ReLU
        return output
'''


'''#MLP全连接层
class XRRNet(nn.Module):
    def __init__(self):               #MLP
        super(XRRNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear( XRR_size,1000),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1000, 4000),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(4000, 2000),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2000, 400),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(400, Mat_size),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.model(x)
        return x
'''    


'''
class XRRNet(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, output_size=11, dropout=0.1):
        super(XRRNet, self).__init__()
        
        # Bi-directional LSTM
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        
        # Fully Connected Layer
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),  # 因为是双向 LSTM，所以 hidden_size 乘以 2
            nn.ReLU(),
            nn.Linear(256, output_size)  # 输出 11 个材料参数
        )

    def forward(self, x):
        x = x.view(batch_size, XRR_size, 1)
        # x shape: (batch_size, 200, 1)  # 200 个点，每个点是 1 维
        rnn_out, _ = self.rnn(x)  # LSTM 输出
        last_hidden = rnn_out[:, -1, :]  # 取最后一个时间步的输出
        
        # 通过全连接层
        output = self.fc(last_hidden)
        return output
'''


'''
class XRRNet(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, output_size=11, dropout=0.1):
        super(XRRNet, self).__init__()
        
        # 双向 RNN 替代 LSTM
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),  # 双向 RNN，hidden_size 乘以 2
            nn.ReLU(),
            nn.Linear(256, output_size)  # 输出 11 个材料参数
        )

    def forward(self, x):
        x = x.view(batch_size, XRR_size, 1)  # 保持 (batch_size, 200, 1)
        rnn_out, _ = self.rnn(x)  # RNN 输出
        last_hidden = rnn_out[:, -1, :]  # 取最后一个时间步的输出
        
        # 通过全连接层
        output = self.fc(last_hidden)
        return output
'''


'''
#BERT
class XRRNet(nn.Module):
    def __init__(self, n_layers=6):
        super(XRRNet, self).__init__()
        self.embedding = Embedding()
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(0.2),
            nn.Tanh(),
        )
        self.linear = nn.Linear(d_model, d_model)
        self.activ2 = gelu
        # fc2 is shared with embedding layer
        #embed_weight = self.embedding.tok_embed.weight
        self.fc2 = nn.Linear(d_model, Mat_size, bias=False)
        #self.fc2.weight = embed_weight

    def forward(self, input_ids):
        input_ids = input_ids.view(batch_size, XRR_size, 1)
        output = self.embedding(input_ids) # [bach_size, seq_len, d_model]
        for layer in self.layers:
            # output: [batch_size, max_len, d_model]
            output = layer(output)
        h_pooled = self.fc(output[:, 0]) # [batch_size, d_model]
        h_masked = self.activ2(self.linear(h_pooled)) # [batch_size, max_pred, d_model]
        logits_lm = self.fc2(h_masked) # [batch_size, max_pred, vocab_size]
        return logits_lm

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Linear(1, d_model).to(device)  # token embedding
        self.pos_embed = nn.Linear(1, d_model).to(device)  # position embedding
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        seq_len = x.size(1)
        x = x.to(device)
        pos = torch.arange(seq_len, dtype=torch.float32).to(device)
        pos = pos.unsqueeze(0).expand(batch_size, seq_len) # [seq_len] -> [batch_size, seq_len]
        pos = pos.unsqueeze(2)
        embedding = self.tok_embed(x) + self.pos_embed(pos)
        return self.norm(embedding)

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) # scores : [batch_size, n_heads, seq_len, seq_len]
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads).to(device)
        self.W_K = nn.Linear(d_model, d_k * n_heads).to(device)
        self.W_V = nn.Linear(d_model, d_v * n_heads).to(device)
        self.fcc = nn.Linear(n_heads * d_v, d_model).to(device)
        self.norm = nn.LayerNorm(d_model).to(device)
    def forward(self, Q, K, V):
        # q: [batch_size, seq_len, d_model], k: [batch_size, seq_len, d_model], v: [batch_size, seq_len, d_model]
        residual, batch_size = Q.to(device), Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # q_s: [batch_size, n_heads, seq_len, d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # k_s: [batch_size, n_heads, seq_len, d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # v_s: [batch_size, n_heads, seq_len, d_v]


        # context: [batch_size, n_heads, seq_len, d_v], attn: [batch_size, n_heads, seq_len, seq_len]
        context = ScaledDotProductAttention()(q_s, k_s, v_s)
        context = context.to(device)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v) # context: [batch_size, seq_len, n_heads, d_v]
        output = self.fcc(context)
        return self.norm(output + residual) # output: [batch_size, seq_len, d_model]

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff).to(device)
        self.fc2 = nn.Linear(d_ff, d_model).to(device)

    def forward(self, x):
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        return self.fc2(gelu(self.fc1(x)))

class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention().to(device)
        self.pos_ffn = PoswiseFeedForwardNet().to(device)

    def forward(self, enc_inputs):
        enc_outputs = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size, seq_len, d_model]
        return enc_outputs
'''

Hidden_size = 256
class XRRNet(nn.Module):
    '''
    def __init__(self, input_size = 1):
        super(XRRNet, self).__init__()
        self.hidden_size = Hidden_size
        self.num_layers = 3
        self.biLSTM = nn.LSTM(input_size, self.hidden_size, bidirectional = True, num_layers = self.num_layers, dropout = 0.2)
        self.fc_thi = nn.Sequential(
            nn.Linear(2*self.hidden_size, 2)
        )
        self.fc_rou = nn.Sequential(
            nn.Linear(2*self.hidden_size, 3)
        )
        self.fc_real = nn.Sequential(
            nn.Linear(2*self.hidden_size, 3)
        )
        self.fc_imag = nn.Sequential(
            nn.Linear(2*self.hidden_size, 3)
        )
        self.fc_hid1 = nn.Sequential(
            nn.Linear(self.hidden_size, 2*self.hidden_size),
            nn.ReLU(),
            nn.Linear(2*self.hidden_size, self.hidden_size)
            )
        self.fc_hid2 = nn.Sequential(
            nn.Linear(self.hidden_size, 2*self.hidden_size),
            nn.ReLU(),
            nn.Linear(2*self.hidden_size, self.hidden_size)
            )
        self.fc_hid3 = nn.Sequential(
            nn.Linear(self.hidden_size, 2*self.hidden_size),
            nn.ReLU(),
            nn.Linear(2*self.hidden_size, self.hidden_size)
            )
        self.fc_hid4 = nn.Sequential(
            nn.Linear(self.hidden_size, 2*self.hidden_size),
            nn.ReLU(),
            nn.Linear(2*self.hidden_size, self.hidden_size)
            )



    def attn(self, input, hidden_state):
        #input: [batch_size, seq_len, directions*hidden_size] 
        hidden_state = hidden_state.transpose(0, 1).contiguous().view(batch_size, -1, 1)
        scores = torch.matmul(input, hidden_state).squeeze(-1)
        scores = F.softmax(scores, 1)  #scores:[batch_size, seq_len]
        context = torch.matmul(input.transpose(1, 2), scores.unsqueeze(-1)).squeeze(-1)
        return context  #context:[batch_size, directions*hidden_size]

    


    def forward(self, x):
        x = x.view(batch_size, XRR_size, 1).transpose(0, 1)
        output, (hidden_state, cell_state) = self.biLSTM(x) #output:[XRR_size, batch_size, directions*hidden_size] hidden_state:[bidirection(2)*num_layers, batch_size, hidden_size]
        output = output.transpose(0, 1)
        hidden_state = hidden_state[-2:]
        thi = self.fc_thi(self.attn(output, self.fc_hid1(hidden_state)))
        rou = self.fc_rou(self.attn(output, self.fc_hid2(hidden_state)))
        real = self.fc_real(self.attn(output, self.fc_hid3(hidden_state)))
        imag = self.fc_imag(self.attn(output, self.fc_hid4(hidden_state)))
        return torch.cat((thi, rou, real, imag), dim = 1)
    '''
    def __init__(self, input_size = 1):
        super(XRRNet, self).__init__()
        self.hidden_size = Hidden_size
        self.num_layers = 3
        self.biLSTM = nn.LSTM(input_size, self.hidden_size, bidirectional = True, num_layers = self.num_layers, dropout = 0.2)
        self.result = nn.Sequential(
            nn.Linear(2*self.hidden_size, 100),
            nn.Sigmoid(),
            nn.Linear(100, 11)
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

       


if __name__ == '__main__':
    main()
















