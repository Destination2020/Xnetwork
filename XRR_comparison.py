import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import Calculate
import Draw_line


def calculate_XRR_with_SLD(Model, waveLength, incidence_angle_input, 
                           HWHM=0.0, background_constant=0.0,
                           method=1, kind=1, resolution_mode='q_fractional'):
    """
    计算 XRR 曲线（角度扫描），输入 SLD 实部和虚部
    
    参数:
        Model: 层状结构模型，格式 [['Material', thickness, roughness, Re_SLD, Im_SLD], ...]
               或者 [['Material', thickness, roughness, density], ...]（当 Re_SLD<0 时从密度计算）
               thickness 单位 nm, roughness 单位 nm
               Re_SLD 和 Im_SLD 单位 Å⁻²
        waveLength: 波长（nm）
        incidence_angle_input: 角度范围（度）
        HWHM: 半高半宽
            - resolution_mode='angle': 角度空间HWHM（度）
            - resolution_mode='q_constant': Q空间HWHM（Å⁻¹）
            - resolution_mode='q_fractional': δQ_HWHM/Q比值（REFLEX方式）
        background_constant: 常数背景强度值
        method: 计算方法，1=传递矩阵法，2=一次反射近似法
        kind: 数据来源，1=Henke, 2=NIST, 3=RATB
        resolution_mode: 分辨率展宽模式
            - 'angle': 角度空间卷积（原始方式）
            - 'q_constant': Q空间常数δQ卷积
            - 'q_fractional': Q空间δQ/Q逐点展宽（REFLEX方式）
    
    返回:
        incidence_angle_input: 角度数组
        R_processed: 处理后的反射率数组
        R_raw: 原始反射率数组（未处理）
    """
    Energy = [1239.841984 / waveLength]
    
    incidence_angle = np.float64(incidence_angle_input / 180.0 * np.pi)
    
    Layer_system = Calculate.Layer_model()
    
    for layer in Model:
        Layer_system.add_lay_with_sld(layer, Energy, kind)
    
    if method == 1:
        r, R_raw, Q = Calculate.Angle_matrix(Layer_system, incidence_angle, waveLength)
    elif method == 2:
        r, R_raw, Q = Calculate.Angle_neglect(Layer_system, incidence_angle, waveLength)
    else:
        raise ValueError("method 必须是 1 或 2")
    
    R_processed = R_raw.copy()
    
    if HWHM > 0:
        R_processed = Calculate.apply_resolution_broadening(
            R_processed, incidence_angle_input, HWHM,
            mode=resolution_mode, Q=Q
        )
    
    if background_constant > 0:
        R_processed += background_constant
    
    R_processed = np.maximum(R_processed, 0)
    
    return incidence_angle_input, R_processed, R_raw


def load_experimental_data(csv_file):
    """
    加载实验数据
    
    参数:
        csv_file: CSV 文件路径
    
    返回:
        angle: 角度数组（度）
        intensity: 实验强度数组
    """
    data = pd.read_csv(csv_file, header=None)
    angle = data.iloc[:, 0].values
    intensity = data.iloc[:, 1].values
    return angle, intensity


def plot_comparison(Model, waveLength, incidence_angle_input, experimental_file,
                    HWHM=0.0, background_constant=0.0,
                    method=1, kind=1, resolution_mode='q_fractional',
                    filename=None, yscale='log', xscale='linear'):
    """
    计算 XRR 曲线并与实验数据对比绘图
    
    参数:
        Model: 层状结构模型，格式 [['Material', thickness, roughness, Re_SLD, Im_SLD], ...]
        waveLength: 波长（nm）
        incidence_angle_input: 角度范围（度）
        experimental_file: 实验数据 CSV 文件路径
        HWHM: 半高半宽（含义取决于resolution_mode）
        background_constant: 常数背景强度值
        method: 计算方法
        kind: 数据来源
        resolution_mode: 分辨率展宽模式（'angle', 'q_constant', 'q_fractional'）
        filename: 保存文件名（可选）
        yscale: y轴缩放类型
        xscale: x轴缩放类型
    """
    angle_calc, R_processed, R_raw = calculate_XRR_with_SLD(
        Model, waveLength, incidence_angle_input,
        HWHM, background_constant, method, kind, resolution_mode
    )
    
    exp_angle, exp_intensity = load_experimental_data(experimental_file)
    
    R_interp = np.interp(exp_angle, angle_calc, R_processed)
    
    Draw_line.Draw_line(
        (exp_angle, exp_intensity, R_interp),
        xlabel=['θ(°)'],
        xscale=xscale,
        yscale=yscale,
        ylabel=['Intensity (counts)'],
        sub_title=[f'λ={waveLength}nm, HWHM={HWHM}°, Bg={background_constant:.0e}'],
        legend=[{'loc': 'upper right', 'fontsize': 10, 'title': 'Data type'}],
        label=[['Experimental', 'Calculated']],
        filename=filename
    )
    
    return exp_angle, R_interp, exp_angle, exp_intensity


if __name__ == "__main__":
    Model = [
        ['Air', 0.0, 0.65, 0.0, 0.0],
        ['TiO2', 9.7, 2.739, 4.1e-3, 1.73e-5],
        ['0.75SrNb2O6 + 0.25BaNb2O6', 8.976, 0.1, 4.1e-3, 1.54e-5],
        ['Si', 0.0, 0.0, 2.01e-3, 4.57e-5]
    ]
    
    waveLength = 0.154
    
    experimental_file = os.path.join(current_dir, 'doublelayer1.csv')
    
    exp_angle_temp, _ = load_experimental_data(experimental_file)
    angle_min, angle_max = exp_angle_temp[0], exp_angle_temp[-1]
    incidence_angle_input = exp_angle_temp
    
    HWHM = 0.03  # δQ_HWHM/Q比值（REFLEX方式），对应δQ_FWHM/Q = 2%
    # 旧版角度空间HWHM = 0.014° 的等效Q空间参数:
    # dQ_hwhm = Calculate.angle_hwhm_to_q_hwhm(0.014, waveLength) ≈ 0.002 Å⁻¹
    # 在Q=0.1 Å⁻¹处: δQ_HWHM/Q ≈ 0.02
    resolution_mode = 'q_fractional'
    background_constant = 7e-7
    
    angle_calc, R_processed, exp_angle, exp_intensity = plot_comparison(
        Model=Model,
        waveLength=waveLength,
        incidence_angle_input=incidence_angle_input,
        experimental_file=experimental_file,
        HWHM=HWHM,
        background_constant=background_constant,
        method=1,
        kind=1,
        resolution_mode=resolution_mode
    )
    
    print(f"计算完成！")
    print(f"模型结构: {Model}")
    print(f"波长: {waveLength} nm")
    print(f"分辨率展宽模式: {resolution_mode}")
    if resolution_mode == 'q_fractional':
        print(f"δQ_HWHM/Q: {HWHM} (δQ_FWHM/Q: {HWHM*2})")
    elif resolution_mode == 'q_constant':
        print(f"δQ_HWHM: {HWHM} Å⁻¹")
    else:
        print(f"HWHM: {HWHM}°")
    print(f"常数背景强度: {background_constant}")
    print(f"角度范围: {angle_min}° - {angle_max}°")
    print(f"数据点数: {len(incidence_angle_input)}")
