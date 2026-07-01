import argparse
import os
import sys
import time

import Draw_line
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

XRR_size = 621
Fuzzy_SLD_size = 6
Mat_size = 17
batch_size = 64
num_epochs = 50

min_thi = [5.0, 5.0]
max_thi = [20.0, 20.0]
min_rou = [0.05, 0.05, 0.5, 0.5]
max_rou = [0.25, 0.35, 2.0, 2.0]
min_real_den = [5e-4, 5e-4, 1e-3]
max_real_den = [1e-2, 1e-2, 5e-3]
min_imag_den = [5e-5, 5e-5, 3e-5]
max_imag_den = [1e-3, 1e-3, 7e-5]
min_surface_params = [0.5, 2e-5, 1e-6]
max_surface_params = [1.2, 1.5e-3, 3e-5]
min_sigma_deg = [0.015]
max_sigma_deg = [0.025]
min_angle_offset = [-0.01]
max_angle_offset = [0.01]

PARAM_GROUPS = [
    ("thk", [0, 1]),
    ("rou", [2, 3, 4, 5]),
    ("re", [6, 7, 8]),
    ("im", [9, 10, 11]),
    ("dust", [12, 13, 14]),
    ("inst", [15, 16]),
    ("stack", list(range(15))),
]

Hidden_size = 256
Fuzzy_encoder_output_size = 128

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=[1, 2], default=1)
    parser.add_argument("--train_dir", type=str, default="")
    parser.add_argument("--test_dir", type=str, default="")
    parser.add_argument("--pretrained_path", type=str, default="")
    parser.add_argument("--num_epochs", type=int, default=num_epochs)
    return parser.parse_args()

def get_mat_bounds(device=None, dtype=torch.float32):
    min_range = torch.tensor(
        min_thi + min_rou + min_real_den + min_imag_den + min_surface_params + min_sigma_deg + min_angle_offset,
        dtype=dtype,
        device=device,
    )
    max_range = torch.tensor(
        max_thi + max_rou + max_real_den + max_imag_den + max_surface_params + max_sigma_deg + max_angle_offset,
        dtype=dtype,
        device=device,
    )
    return min_range, max_range

def get_fuzzy_bounds(device=None, dtype=torch.float32):
    min_range = torch.tensor(min_real_den + min_imag_den, dtype=dtype, device=device)
    max_range = torch.tensor(max_real_den + max_imag_den, dtype=dtype, device=device)
    return min_range, max_range

def Log_XRR(data_XRR):
    data_XRR = torch.clamp(data_XRR, min=1e-10)
    outputs = torch.log10(data_XRR + 1e-11)
    outputs = torch.nan_to_num(outputs, nan=-10.0, posinf=10.0, neginf=-10.0)
    return outputs

def Normalize_Mat(data_Mat):
    min_range, max_range = get_mat_bounds(device=data_Mat.device, dtype=data_Mat.dtype)
    data_Mat = (data_Mat - min_range) / (max_range - min_range)
    return data_Mat

def Normalize_Fuzzy_SLD(data_Fuzzy):
    min_range, max_range = get_fuzzy_bounds(device=data_Fuzzy.device, dtype=data_Fuzzy.dtype)
    data_Fuzzy = (data_Fuzzy - min_range) / (max_range - min_range)
    return data_Fuzzy

def Format_Group_MAE(abs_error_sum, sample_count):
    if sample_count <= 0:
        return "no valid samples"
    mae = abs_error_sum / sample_count
    return " ".join([f"{name}={float(mae[indices].mean()):.4f}" for name, indices in PARAM_GROUPS])

class XRRDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.files = sorted(os.listdir(data_dir))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.files[idx])
        data = np.loadtxt(file_path)
        expected_size = XRR_size + Fuzzy_SLD_size + Mat_size
        if data.size != expected_size:
            raise ValueError(
                f"Unexpected sample size in {file_path}: expected {expected_size}, got {data.size}. "
                "Regenerate the dataset with the updated dust-surface generator."
            )
        xrr_data = data[:XRR_size]
        fuzzy_sld_data = data[XRR_size:XRR_size + Fuzzy_SLD_size]
        mat_data = data[-Mat_size:]
        xrr_data = np.nan_to_num(xrr_data, nan=1e-10, posinf=1e-10, neginf=1e-10)
        xrr_data = np.maximum(xrr_data, 1e-10)
        mat_data = np.nan_to_num(mat_data, nan=0.0, posinf=0.0, neginf=0.0)
        xrr_data = Log_XRR(torch.tensor(xrr_data, dtype=torch.float))
        fuzzy_sld_data = Normalize_Fuzzy_SLD(torch.tensor(fuzzy_sld_data, dtype=torch.float))
        mat_data = torch.tensor(mat_data, dtype=torch.float)
        return xrr_data, fuzzy_sld_data, mat_data

class XRRNet(nn.Module):
    def __init__(self, xrr_input_size=1, fuzzy_sld_input_size=Fuzzy_SLD_size):
        super().__init__()
        self.hidden_size = Hidden_size
        self.num_layers = 3
        self.biLSTM = nn.LSTM(
            xrr_input_size,
            self.hidden_size,
            bidirectional=True,
            num_layers=self.num_layers,
            dropout=0.2,
        )
        self.fuzzy_encoder = nn.Sequential(
            nn.Linear(fuzzy_sld_input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, Fuzzy_encoder_output_size),
        )
        self.result = nn.Sequential(
            nn.Linear(2 * self.hidden_size + Fuzzy_encoder_output_size, 256),
            nn.Sigmoid(),
            nn.Linear(256, 128),
            nn.Sigmoid(),
            nn.Linear(128, Mat_size),
        )
        self.Attention = nn.ModuleList([MultiHeadAttention() for _ in range(6)])

    def forward(self, xrr_input, fuzzy_sld_input):
        batch_size_local = xrr_input.size(0)
        xrr_size_local = xrr_input.size(1)
        xrr_seq = xrr_input.view(batch_size_local, xrr_size_local, 1).transpose(0, 1)
        output, (hidden_state, _) = self.biLSTM(xrr_seq)
        hidden_state = hidden_state[-2:]
        xrr_features = hidden_state.transpose(0, 1).contiguous().view(batch_size_local, 1, 2 * Hidden_size)
        for layer in self.Attention:
            xrr_features = layer(xrr_features, output, output)
        xrr_features = xrr_features.squeeze(1)
        fuzzy_features = self.fuzzy_encoder(fuzzy_sld_input)
        combined = torch.cat([xrr_features, fuzzy_features], dim=1)
        result = self.result(combined)
        return result

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads=3):
        super().__init__()
        self.d_Q = 64
        self.d_K = 64
        self.d_V = 64
        self.heads = n_heads
        self.fc_Q = nn.Linear(2 * Hidden_size, self.heads * self.d_Q)
        self.fc_K = nn.Linear(2 * Hidden_size, self.heads * self.d_K)
        self.fc_V = nn.Linear(2 * Hidden_size, self.heads * self.d_V)
        self.recover = nn.Linear(self.heads * self.d_V, 2 * Hidden_size)
        self.LayerNorm1 = nn.LayerNorm(2 * Hidden_size)
        self.LayerNorm2 = nn.LayerNorm(2 * Hidden_size)
        self.PoswiseFeedforward = nn.Sequential(
            nn.Linear(2 * Hidden_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2 * Hidden_size),
        )

    def ScaleDotProductAttention(self, Q, K, V):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_K)
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)
        return output

    def forward(self, Q_input, K_input, V_input):
        batch_size_local = Q_input.size(0)
        residual = Q_input
        Q = self.fc_Q(residual).view(batch_size_local, -1, self.heads, self.d_Q).transpose(1, 2)
        K = self.fc_K(K_input.transpose(0, 1)).view(batch_size_local, -1, self.heads, self.d_K).transpose(1, 2)
        V = self.fc_V(V_input.transpose(0, 1)).view(batch_size_local, -1, self.heads, self.d_V).transpose(1, 2)
        output = self.ScaleDotProductAttention(Q, K, V)
        output = output.transpose(1, 2).contiguous().view(batch_size_local, -1, self.heads * self.d_V)
        output = self.recover(output)
        output = self.LayerNorm1(output + residual)
        output = self.LayerNorm2(output + self.PoswiseFeedforward(output))
        return output

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
    default_test_dir = (
        f"D:/Work_files/EUV-OCT/XRR/XRR_Software/DeepLearning/data/test_data/test_exp_transition_stage{stage}"
    )
    default_train_dir = (
        f"D:/Work_files/EUV-OCT/XRR/XRR_Software/DeepLearning/data/train_data/train_exp_transition_stage{stage}"
    )
    test_dir = args.test_dir if args.test_dir else default_test_dir
    train_dir = args.train_dir if args.train_dir else default_train_dir
    train_dataset = XRRDataset(train_dir)
    test_dataset = XRRDataset(test_dir)

    train_epochs = args.num_epochs
    learning_rate = 0.0007 if stage == 1 else 0.0001
    model_dir = r"D:\Work_files\EUV-OCT\XRR\XRR_Software\DeepLearning\XRR_FCNN\AEExperiment"

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    XRR_net = XRRNet()
    if stage == 2 and args.pretrained_path and os.path.exists(args.pretrained_path):
        try:
            XRR_net.load_state_dict(torch.load(args.pretrained_path, map_location="cpu"))
        except RuntimeError as exc:
            raise RuntimeError(
                "Pretrained checkpoint shape does not match the updated 17-parameter dust-surface model."
            ) from exc
    if torch.cuda.is_available():
        XRR_net.cuda()

    loss_fn = nn.L1Loss()
    if torch.cuda.is_available():
        loss_fn = loss_fn.cuda()
    optimizer = torch.optim.AdamW(XRR_net.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=train_epochs)

    total_loss_history = []
    test_loss_history = []
    session_timestamp = time.strftime("%Y%m%d_%H%M%S")

    for epoch in range(train_epochs):
        XRR_net.train()
        total_loss = 0.0
        valid_batches = 0
        train_param_abs_error_sum = torch.zeros(Mat_size, dtype=torch.float64)
        train_sample_count = 0
        for batch_XRR, batch_Fuzzy_SLD, batch_Mat in train_dataloader:
            batch_Mat = Normalize_Mat(batch_Mat)
            if (
                not torch.isfinite(batch_XRR).all()
                or not torch.isfinite(batch_Fuzzy_SLD).all()
                or not torch.isfinite(batch_Mat).all()
            ):
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
            train_param_abs_error_sum += torch.sum(torch.abs(output - batch_Mat), dim=0).detach().cpu().double()
            train_sample_count += batch_Mat.size(0)
            valid_batches += 1
        scheduler.step()

        train_epoch_loss = total_loss / max(valid_batches, 1)
        total_loss_history.append(train_epoch_loss)
        if epoch >= 1 and total_loss_history[epoch] < total_loss_history[epoch - 1]:
            torch.save(
                XRR_net.state_dict(),
                os.path.join(model_dir, f"XRR_net_dust_stage{stage}_train_{session_timestamp}.pth"),
            )
        print(f"epoch {epoch + 1}\tloss: {train_epoch_loss}")
        print(f"train group mae:\t{Format_Group_MAE(train_param_abs_error_sum, train_sample_count)}")

        XRR_net.eval()
        with torch.no_grad():
            test_loss = 0.0
            valid_test_batches = 0
            test_param_abs_error_sum = torch.zeros(Mat_size, dtype=torch.float64)
            test_sample_count = 0
            for batch_XRR, batch_Fuzzy_SLD, batch_Mat in test_dataloader:
                batch_Mat = Normalize_Mat(batch_Mat)
                if (
                    not torch.isfinite(batch_XRR).all()
                    or not torch.isfinite(batch_Fuzzy_SLD).all()
                    or not torch.isfinite(batch_Mat).all()
                ):
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
                test_param_abs_error_sum += torch.sum(torch.abs(output - batch_Mat), dim=0).detach().cpu().double()
                test_sample_count += batch_Mat.size(0)
                valid_test_batches += 1
        test_epoch_loss = test_loss / max(valid_test_batches, 1)
        test_loss_history.append(test_epoch_loss)
        if epoch >= 1 and test_loss_history[epoch] <= min(test_loss_history):
            torch.save(
                XRR_net.state_dict(),
                os.path.join(model_dir, f"XRR_net_dust_stage{stage}_test_{session_timestamp}.pth"),
            )
        print(f"test loss: {test_epoch_loss}")
        print(f"test group mae:\t{Format_Group_MAE(test_param_abs_error_sum, test_sample_count)}")

    Draw_line.Draw_line(
        (range(train_epochs), total_loss_history, test_loss_history),
        xlabel=["epoch"],
        ylabel=["Loss"],
        sub_title=["Dust-surface model"],
        legend=[{"loc": "upper right", "fontsize": 10, "title": "Calculation method"}],
        label=[["Train", "Test"]],
    )
    torch.save(
        XRR_net.state_dict(),
        os.path.join(model_dir, f"XRR_net_dust_stage{stage}_final_{session_timestamp}.pth"),
    )

if __name__ == "__main__":
    main()
