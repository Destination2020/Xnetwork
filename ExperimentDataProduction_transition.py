import argparse
import os

import numpy as np
from scipy.ndimage import gaussian_filter1d

waveLength = 0.154056
XRR_size = 621
Fuzzy_SLD_size = 6
Mat_size = 17
Angle_init = np.linspace(0.4000, 3.5000, XRR_size, endpoint=True, dtype=np.float64)
Angle = Angle_init / 180 * np.pi
rng = np.random.default_rng()

min_thi = np.array([5.0, 5.0], dtype=np.float64)
max_thi = np.array([20.0, 20.0], dtype=np.float64)

min_rou = np.array([0.05, 0.05, 0.5, 0.5], dtype=np.float64)
max_rou = np.array([0.25, 0.35, 2.0, 2.0], dtype=np.float64)

min_real_den = np.array([5e-4, 5e-4, 1e-3], dtype=np.float64)
max_real_den = np.array([1e-2, 1e-2, 5e-3], dtype=np.float64)
min_imag_den = np.array([5e-5, 5e-5, 3e-5], dtype=np.float64)
max_imag_den = np.array([1e-3, 1e-3, 7e-5], dtype=np.float64)

min_surface_thickness = 0.5
max_surface_thickness = 1.2
min_dust_real = 2e-5
max_dust_real = 1.5e-3
min_dust_imag = 1e-6
max_dust_imag = 3e-5

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=[1, 2], default=1)
    parser.add_argument("--num_samples", type=int, default=1024)
    parser.add_argument("--output_dir", type=str, default="")
    return parser.parse_args()

def stage_config(stage):
    if stage == 3:
        return {
            "angle_scale_min": 0.9995,
            "angle_scale_max": 1.0005,
            "angle_offset_min": -0.003,
            "angle_offset_max": 0.003,
            "intensity_scale_min": 0.98,
            "intensity_scale_max": 1.02,
            "background_offset_min": -1.5e-7,
            "background_offset_max": 1.5e-7,
            "background_mean": 1.08e-6,
            "background_std": 1.0e-7,
            "noise_smooth_sigma": 0.8,
            "sigma_deg_min": 0.017,
            "sigma_deg_max": 0.023,
            "surface_thickness_max": 1.0,
            "surface_bias_power": 2.6,
            "dust_real_min": 2e-5,
            "dust_real_max": 1.5e-3,
            "dust_imag_min": 1e-6,
            "dust_imag_max": 3e-5,
        }
    return {
        "angle_scale_min": 0.998,
        "angle_scale_max": 1.002,
        "angle_offset_min": -0.01,
        "angle_offset_max": 0.01,
        "intensity_scale_min": 0.95,
        "intensity_scale_max": 1.05,
        "background_offset_min": -1.2e-7,
        "background_offset_max": 1.2e-7,
        "background_mean": 8.2e-7,
        "background_std": 2.7e-7,
        "noise_smooth_sigma": 1.0,
        "sigma_deg_min": 0.015,
        "sigma_deg_max": 0.025,
        "surface_thickness_max": 1.2,
        "surface_bias_power": 2.2,
        "dust_real_min": 2e-5,
        "dust_real_max": 1.5e-3,
        "dust_imag_min": 1e-6,
        "dust_imag_max": 3e-5,
    }

def sample_biased_thickness(min_value, max_value, bias_power):
    return float(min_value + (rng.random() ** bias_power) * (max_value - min_value))

def sample_log_uniform(min_value, max_value):
    if max_value <= min_value:
        return float(min_value)
    return float(np.exp(rng.uniform(np.log(min_value), np.log(max_value))))

def sample_main_materials():
    main_thickness = rng.random(size=2) * (max_thi - min_thi) + min_thi
    roughness = rng.normal(
        loc=np.array([0.12, 0.18, 1.0, 1.1]),
        scale=np.array([0.04, 0.06, 0.35, 0.40]),
    )
    roughness = np.clip(roughness, min_rou, max_rou)

    real_den = rng.random(size=3) * rng.random(size=3) * (max_real_den - min_real_den) + min_real_den
    max_imag = np.minimum(max_imag_den, 0.3 * real_den)
    imag_den = rng.random(size=3) * (max_imag - min_imag_den) + min_imag_den
    return main_thickness, roughness, real_den, imag_den

def sample_surface_params(cfg):
    surface_thickness = sample_biased_thickness(
        min_surface_thickness,
        cfg["surface_thickness_max"],
        cfg["surface_bias_power"],
    )
    surface_thickness = float(np.clip(surface_thickness, min_surface_thickness, max_surface_thickness))
    dust_real = sample_log_uniform(cfg["dust_real_min"], cfg["dust_real_max"])
    dust_imag_max = min(cfg["dust_imag_max"], 0.05 * dust_real)
    if dust_imag_max < cfg["dust_imag_min"]:
        dust_imag_max = cfg["dust_imag_min"]
    dust_imag = sample_log_uniform(cfg["dust_imag_min"], dust_imag_max)
    return surface_thickness, dust_real, dust_imag

def apply_surface_roughness_constraint(roughness, surface_thickness):
    constrained = np.array(roughness, dtype=np.float64, copy=True)
    surface_limit = max(min_surface_thickness, float(surface_thickness))
    constrained[0] = np.clip(constrained[0], min_rou[0], min(max_rou[0], surface_limit))
    constrained[1] = np.clip(constrained[1], min_rou[1], min(max_rou[1], surface_limit))
    return constrained

def build_fuzzy_sld(real_den, imag_den):
    fuzzy_real_den = np.clip(real_den * rng.uniform(0.5, 1.5, size=3), min_real_den, max_real_den)
    fuzzy_imag_den = np.clip(imag_den * rng.uniform(0.5, 1.5, size=3), min_imag_den, max_imag_den)
    return fuzzy_real_den, fuzzy_imag_den

def build_full_stack(main_thickness, roughness, main_real, main_imag, surface_thickness, dust_real, dust_imag):
    thickness = np.array(
        [0.0, surface_thickness, main_thickness[0], main_thickness[1], 0.0],
        dtype=np.float64,
    )
    real_den = np.array(
        [0.0, dust_real, main_real[0], main_real[1], main_real[2]],
        dtype=np.float64,
    )
    imag_den = np.array(
        [0.0, dust_imag, main_imag[0], main_imag[1], main_imag[2]],
        dtype=np.float64,
    )
    return thickness, roughness, real_den, imag_den

def pack_labels(
    main_thickness,
    roughness,
    main_real,
    main_imag,
    surface_thickness,
    dust_real,
    dust_imag,
    sigma_deg,
    angle_offset,
):
    return np.concatenate(
        (
            main_thickness,
            roughness,
            main_real,
            main_imag,
            [surface_thickness, dust_real, dust_imag, sigma_deg, angle_offset],
        ),
        axis=0,
    )

def main():
    args = parse_args()
    cfg = stage_config(args.stage)
    output_dir = args.output_dir if args.output_dir else (
        f"D:/Work_files/EUV-OCT/XRR/XRR_Software/DeepLearning/data/train_data/train_exp_transition_stage{args.stage}"
    )
    os.makedirs(output_dir, exist_ok=True)

    for num in range(args.num_samples):
        main_thickness, roughness, main_real, main_imag = sample_main_materials()
        surface_thickness, dust_real, dust_imag = sample_surface_params(cfg)
        roughness = apply_surface_roughness_constraint(roughness, surface_thickness)
        fuzzy_real_den, fuzzy_imag_den = build_fuzzy_sld(main_real, main_imag)

        angle_scale = rng.uniform(cfg["angle_scale_min"], cfg["angle_scale_max"])
        angle_offset = rng.uniform(cfg["angle_offset_min"], cfg["angle_offset_max"])
        intensity_scale = rng.uniform(cfg["intensity_scale_min"], cfg["intensity_scale_max"])
        background_offset = rng.uniform(cfg["background_offset_min"], cfg["background_offset_max"])

        thickness, interface_roughness, real_den, imag_den = build_full_stack(
            main_thickness, roughness, main_real, main_imag, surface_thickness, dust_real, dust_imag
        )
        incidence_angle = (angle_scale * Angle_init + angle_offset) / 180 * np.pi
        n = N_angle(real_den, imag_den, waveLength).flatten()
        _, reflectivity, _ = Angle_matrix(thickness, interface_roughness, n, incidence_angle, waveLength)

        step_size = Angle_init[1] - Angle_init[0]
        sigma_deg = rng.uniform(cfg["sigma_deg_min"], cfg["sigma_deg_max"])
        conv_sigma = sigma_deg / step_size
        reflectivity = gaussian_filter1d(reflectivity, sigma=conv_sigma, mode="nearest")

        white_noise = rng.normal(0.0, cfg["background_std"], size=reflectivity.shape)
        smooth_noise = gaussian_filter1d(white_noise, sigma=cfg["noise_smooth_sigma"], mode="nearest")
        reflectivity = intensity_scale * reflectivity + cfg["background_mean"] + background_offset + smooth_noise
        reflectivity = np.maximum(reflectivity, 1e-10)

        labels = pack_labels(
            main_thickness,
            roughness,
            main_real,
            main_imag,
            surface_thickness,
            dust_real,
            dust_imag,
            sigma_deg,
            angle_offset,
        )
        sample = np.concatenate((reflectivity, fuzzy_real_den, fuzzy_imag_den, labels), axis=0)
        np.savetxt(os.path.join(output_dir, f"{num+1:07d}.txt"), sample)

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

if __name__ == "__main__":
    main()
