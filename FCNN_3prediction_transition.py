import argparse
import csv
import os
from datetime import datetime

import Draw_line
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

XRR_size = 621
Fuzzy_SLD_size = 6
Mat_size = 17
waveLength = 0.154056
Angle_init = np.linspace(0.4000, 3.5000, XRR_size, endpoint=True, dtype=np.float64)
Angle = np.float64(Angle_init / 180.0 * np.pi)

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

PARAM_NAMES = [
    "Thickness (TiO2)",
    "Thickness (SBN)",
    "Roughness (air/dust)",
    "Roughness (dust/TiO2)",
    "Roughness (TiO2/SBN)",
    "Roughness (SBN/Si)",
    "Real (TiO2)",
    "Real (SBN)",
    "Real (Si)",
    "Imag (TiO2)",
    "Imag (SBN)",
    "Imag (Si)",
    "Dust thickness",
    "Real (Dust)",
    "Imag (Dust)",
    "Sigma (deg)",
    "Angle offset (deg)",
]

Hidden_size = 256
Fuzzy_encoder_output_size = 128

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--test_file", type=str, default="")
    parser.add_argument("--exp_file", type=str, default="")
    parser.add_argument("--fuzzy_real_l1", type=float, default=None)
    parser.add_argument("--fuzzy_real_l2", type=float, default=None)
    parser.add_argument("--fuzzy_real_sub", type=float, default=None)
    parser.add_argument("--fuzzy_imag_l1", type=float, default=None)
    parser.add_argument("--fuzzy_imag_l2", type=float, default=None)
    parser.add_argument("--fuzzy_imag_sub", type=float, default=None)
    parser.add_argument("--save_excel", action="store_true")
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

def Normalize_Fuzzy_SLD(data_Fuzzy):
    min_range, max_range = get_fuzzy_bounds(device=data_Fuzzy.device, dtype=data_Fuzzy.dtype)
    data_Fuzzy = (data_Fuzzy - min_range) / (max_range - min_range)
    return data_Fuzzy

def Denormalize_Mat(data_Mat):
    min_range, max_range = get_mat_bounds(device=data_Mat.device, dtype=data_Mat.dtype)
    data_Mat = torch.clamp(data_Mat, 0.0, 1.0)
    data_Mat = data_Mat * (max_range - min_range) + min_range
    constrained = data_Mat.clone()
    dust_thickness = constrained[..., 12:13]
    constrained[..., 2:3] = torch.clamp(
        constrained[..., 2:3],
        min=min_rou[0],
        max=min(max_rou[0], float(max_surface_params[0])),
    )
    constrained[..., 3:4] = torch.clamp(
        constrained[..., 3:4],
        min=min_rou[1],
        max=min(max_rou[1], float(max_surface_params[0])),
    )
    constrained[..., 2:4] = torch.minimum(constrained[..., 2:4], dust_thickness)
    constrained[..., 2:3] = torch.clamp(constrained[..., 2:3], min=min_rou[0], max=max_rou[0])
    constrained[..., 3:4] = torch.clamp(constrained[..., 3:4], min=min_rou[1], max=max_rou[1])
    dust_real = constrained[..., 13:14]
    dust_imag_cap = torch.minimum(torch.full_like(dust_real, max_surface_params[2]), 0.05 * dust_real)
    dust_imag_cap = torch.maximum(dust_imag_cap, torch.full_like(dust_real, min_surface_params[2]))
    constrained[..., 14:15] = torch.clamp(constrained[..., 14:15], min=min_surface_params[2])
    constrained[..., 14:15] = torch.minimum(constrained[..., 14:15], dust_imag_cap)
    return constrained

def Log_XRR(data_XRR):
    data_XRR = torch.clamp(data_XRR, min=1e-10)
    outputs = torch.log10(data_XRR + 1e-11)
    outputs = torch.nan_to_num(outputs, nan=-10.0, posinf=10.0, neginf=-10.0)
    return outputs

def collect_fuzzy_from_args(args):
    values = [
        args.fuzzy_real_l1,
        args.fuzzy_real_l2,
        args.fuzzy_real_sub,
        args.fuzzy_imag_l1,
        args.fuzzy_imag_l2,
        args.fuzzy_imag_sub,
    ]
    prompts = [
        "Input fuzzy Re(SLD) for TiO2",
        "Input fuzzy Re(SLD) for SBN",
        "Input fuzzy Re(SLD) for Si",
        "Input fuzzy Im(SLD) for TiO2",
        "Input fuzzy Im(SLD) for SBN",
        "Input fuzzy Im(SLD) for Si",
    ]
    defaults = [5e-3, 5e-3, 2e-3, 5e-4, 5e-4, 5e-5]
    for idx, value in enumerate(values):
        if value is None:
            values[idx] = float(input(f"{prompts[idx]} [{defaults[idx]}]: ") or str(defaults[idx]))
    return np.array(values, dtype=np.float64)

def load_generated_sample(path):
    data = np.loadtxt(path)
    expected_size = XRR_size + Fuzzy_SLD_size + Mat_size
    if data.size != expected_size:
        raise ValueError(
            f"Unexpected sample size in {path}: expected {expected_size}, got {data.size}. "
            "Regenerate the dataset with the updated dust-surface model."
        )
    xrr_data = data[:XRR_size]
    fuzzy_sld_data = data[XRR_size:XRR_size + Fuzzy_SLD_size]
    ground_truth = data[-Mat_size:]
    return xrr_data, fuzzy_sld_data, ground_truth

def load_experimental_curve(path):
    angle_values = []
    reflectivity_values = []
    reflectivity_only = []
    with open(path, mode="r", newline="") as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            numeric_values = []
            for item in row:
                try:
                    numeric_values.append(float(item))
                except ValueError:
                    continue
            if len(numeric_values) >= 2:
                angle_values.append(numeric_values[0])
                reflectivity_values.append(numeric_values[1])
            elif len(numeric_values) == 1:
                reflectivity_only.append(numeric_values[0])

    if len(angle_values) >= 10 and len(angle_values) == len(reflectivity_values):
        angle_array = np.asarray(angle_values, dtype=np.float64)
        reflectivity_array = np.asarray(reflectivity_values, dtype=np.float64)
        order = np.argsort(angle_array)
        angle_array = angle_array[order]
        reflectivity_array = reflectivity_array[order]
        xrr_data = np.interp(Angle_init, angle_array, reflectivity_array, left=reflectivity_array[0], right=reflectivity_array[-1])
    else:
        if reflectivity_values:
            raw = np.asarray(reflectivity_values, dtype=np.float64)
        else:
            raw = np.asarray(reflectivity_only, dtype=np.float64)
        if raw.size == 0:
            raise ValueError(f"No numeric reflectivity data found in {path}")
        if raw.size >= XRR_size:
            xrr_data = raw[:XRR_size]
        else:
            xrr_data = np.pad(raw, (0, XRR_size - raw.size), mode="edge")
    xrr_data = np.nan_to_num(xrr_data, nan=1e-10, posinf=1e-10, neginf=1e-10)
    xrr_data = np.maximum(xrr_data, 1e-10)
    return xrr_data

def unpack_params(input_data):
    input_data = np.asarray(input_data, dtype=np.float64).copy()
    surface_thickness = float(np.clip(input_data[12], min_surface_params[0], max_surface_params[0]))
    input_data[2] = np.clip(input_data[2], min_rou[0], min(max_rou[0], surface_thickness))
    input_data[3] = np.clip(input_data[3], min_rou[1], min(max_rou[1], surface_thickness))
    main_thickness = np.asarray(input_data[0:2], dtype=np.float64)
    roughness = np.asarray(input_data[2:6], dtype=np.float64)
    real_main = np.asarray(input_data[6:9], dtype=np.float64)
    imag_main = np.asarray(input_data[9:12], dtype=np.float64)
    dust_real = float(np.clip(input_data[13], min_surface_params[1], max_surface_params[1]))
    dust_imag_cap = min(max_surface_params[2], 0.05 * dust_real)
    dust_imag_cap = max(dust_imag_cap, min_surface_params[2])
    dust_imag = float(np.clip(input_data[14], min_surface_params[2], dust_imag_cap))
    sigma_deg = float(input_data[15])
    angle_offset = float(input_data[16])
    return (
        main_thickness,
        roughness,
        real_main,
        imag_main,
        surface_thickness,
        dust_real,
        dust_imag,
        sigma_deg,
        angle_offset,
    )

def build_full_stack(input_data):
    (
        main_thickness,
        roughness,
        real_main,
        imag_main,
        surface_thickness,
        dust_real,
        dust_imag,
        _sigma_deg,
        _angle_offset,
    ) = unpack_params(input_data)

    thickness = np.array(
        [0.0, surface_thickness, main_thickness[0], main_thickness[1], 0.0],
        dtype=np.float64,
    )
    real_den = np.array(
        [0.0, dust_real, real_main[0], real_main[1], real_main[2]],
        dtype=np.float64,
    )
    imag_den = np.array(
        [0.0, dust_imag, imag_main[0], imag_main[1], imag_main[2]],
        dtype=np.float64,
    )
    return thickness, roughness, real_den, imag_den

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

def Materials(input_data):
    real_den, imag_den = None, None
    thickness, roughness, real_den, imag_den = build_full_stack(input_data)
    n = N_angle(real_den, imag_den, waveLength).flatten()
    r, R, Q = Angle_matrix(thickness, roughness, n, Angle, waveLength)
    return r, R, Q

def Reconstruct_R2(input_data, target_curve):
    from scipy.ndimage import gaussian_filter1d

    _, R_raw, _ = Materials(input_data)
    sigma_deg = float(input_data[15])
    angle_offset = float(input_data[16])
    step_size = Angle_init[1] - Angle_init[0]
    conv_sigma = sigma_deg / step_size
    R_conv = gaussian_filter1d(R_raw, sigma=conv_sigma, mode="nearest")
    eps = 1e-10
    y = np.maximum(np.asarray(target_curve, dtype=np.float64), eps)
    offset_min = max(-0.01, angle_offset - 0.003)
    offset_max = min(0.01, angle_offset + 0.003)
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
    best_params = (angle_offset, 1.0, bg_min)
    for off in offsets:
        shifted = np.interp(Angle_init, Angle_init + off, R_conv, left=R_conv[0], right=R_conv[-1])
        shifted = np.maximum(shifted, eps)
        denom = float(np.dot(shifted, shifted)) + 1e-20
        for bg in bg_grid:
            scale = float(max(np.dot(shifted, y - bg) / denom, 1e-6))
            pred = np.maximum(scale * shifted + bg, eps)
            err = np.mean((np.log10(pred) - np.log10(y)) ** 2)
            if err < best_err:
                best_err = err
                best_curve = pred
                best_params = (float(off), scale, float(bg))
    return best_curve, best_params

def N_angle(Re_sld, Im_sld, waveLength):
    k0 = np.float64(np.power(waveLength, 2) / (2 * np.pi))
    n = 1 - k0 * Re_sld + 1j * k0 * Im_sld
    return np.array(n)

def Angle_matrix(thickness, roughness, n, incidence_angle, waveLength):
    lay_num = len(thickness)
    k0 = 2 * np.pi / waveLength
    len_ang = len(incidence_angle)
    k = np.zeros((lay_num, len_ang), dtype=complex)
    for j in range(lay_num):
        k[j] = -np.sqrt((k0 * n[j]) ** 2 - (k0 * n[0] * np.cos(incidence_angle)) ** 2)

    m = np.zeros((lay_num - 1, len_ang), dtype=complex)
    p = np.zeros((lay_num - 1, len_ang), dtype=complex)
    ek = np.zeros((lay_num - 1, len_ang), dtype=complex)
    for j in range(lay_num - 1):
        k_add = k[j] + k[j + 1]
        k_sub = k[j] - k[j + 1]
        p[j] = k_add / (2 * k[j]) * np.exp(-0.5 * (k_sub * roughness[j]) ** 2)
        m[j] = k_sub / (2 * k[j]) * np.exp(-0.5 * (k_add * roughness[j]) ** 2)
        ek[j] = 1j * k[j] * thickness[j]

    h1 = np.exp(-ek)
    h2 = 1 / h1
    M1T = np.array([[p, m], [m, p]])
    M1 = np.transpose(M1T, [2, 3, 0, 1])
    zeros = np.zeros((lay_num - 1, len_ang), dtype=complex)
    H1T = np.array([[h1, zeros], [zeros, h2]])
    H1 = np.transpose(H1T, [2, 3, 0, 1])
    tra_Matrix = np.matmul(H1, M1)
    transferMatrix = np.tile(np.array([[1, 0], [0, 1]], dtype=complex), (len_ang, 1, 1))
    for j in range(lay_num - 1):
        transferMatrix = np.matmul(transferMatrix, tra_Matrix[j])

    r = transferMatrix[:, 0, 1] / transferMatrix[:, 1, 1]
    R = np.abs(r) ** 2
    Q = 0.2 * k0 * np.sin(incidence_angle) * np.abs(n[0])
    return r, R, Q

def main():
    args = parse_args()
    if not args.model_path:
        raise ValueError("Please provide --model_path for the dust-surface model.")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    if not args.test_file and not args.exp_file:
        raise ValueError("Please provide --test_file or --exp_file.")

    if args.exp_file:
        xrr_data = load_experimental_curve(args.exp_file)
        fuzzy_sld_data = collect_fuzzy_from_args(args)
        ground_truth = None
    else:
        if not os.path.exists(args.test_file):
            raise FileNotFoundError(f"Test sample not found: {args.test_file}")
        xrr_data, fuzzy_sld_data, ground_truth = load_generated_sample(args.test_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = XRRNet()
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint shape does not match the updated 17-parameter dust-surface model. "
            "Use a checkpoint retrained with the new dataset layout."
        ) from exc
    model.to(device)
    model.eval()

    xrr_data = np.nan_to_num(xrr_data, nan=1e-10, posinf=1e-10, neginf=1e-10)
    xrr_data = np.maximum(xrr_data, 1e-10)
    R1 = xrr_data

    xrr_pre = Log_XRR(torch.tensor(xrr_data, dtype=torch.float32))
    xrr_input = xrr_pre.view(1, XRR_size).to(device)
    fuzzy_input = Normalize_Fuzzy_SLD(torch.tensor(fuzzy_sld_data, dtype=torch.float32)).view(1, Fuzzy_SLD_size).to(device)

    with torch.no_grad():
        predicted_params = model(xrr_input, fuzzy_input)

    cover_data = Denormalize_Mat(predicted_params[0]).detach().cpu().numpy()
    print(f"Predicted parameters:\n{cover_data}")
    if ground_truth is not None:
        absolute_error = np.abs(cover_data - ground_truth)
        relative_error = np.where(np.abs(ground_truth) > 1e-8, absolute_error / np.abs(ground_truth), np.nan)
        print(f"Ground truth:\n{ground_truth}")
        print(f"Absolute error:\n{absolute_error}")
        print(f"Relative error:\n{relative_error}")
        print(f"Mean relative error (finite only): {np.nanmean(relative_error):.4f}")
    else:
        absolute_error = None
        relative_error = None

    R2, fit_params = Reconstruct_R2(cover_data, R1)
    print(f"Curve compensation: offset={fit_params[0]:.6f} deg, scale={fit_params[1]:.6f}, background={fit_params[2]:.3e}")

    Draw_line.Draw_line(
        (Angle_init, R1, R2),
        xlabel=["ang(deg)"],
        yscale="log",
        ylabel=["R"],
        sub_title=[f"Dust-surface model, Wavelength={waveLength}nm"],
        legend=[{"loc": "upper right", "fontsize": 10, "title": "Comparison"}],
        label=[["Experiment", "Predicted"]],
    )

    save_excel = args.save_excel
    if not save_excel:
        save_excel = input("Enter 1 to save Excel output: ") == "1"
    if save_excel:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_data = {"Parameter": PARAM_NAMES, "Predicted": cover_data}
        if ground_truth is not None:
            excel_data["Ground truth"] = ground_truth
            excel_data["Absolute error"] = absolute_error
            excel_data["Relative error"] = relative_error
        df = pd.DataFrame(excel_data)
        output_name = f"{timestamp}_dust_prediction.xlsx"
        df.to_excel(output_name, index=False)
        print(f"Saved {output_name}")

if __name__ == "__main__":
    main()
