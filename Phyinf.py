import torch


"""
    Vectorized version of the XRR forward model calculation.
    
    Args:
        parameters (torch.Tensor): A tensor of shape (batch_size, 15) containing
                                   normalized material parameters [0, 1].
        min_max_ranges (dict): A dictionary containing the min and max values for
                               thickness, roughness, Re_sld, and Im_sld.
                               e.g., {'min_thi': [...], 'max_thi': [...], ...}
    
    Returns:
        torch.Tensor: A tensor of shape (batch_size, 401) containing the
                      log10 of the calculated reflectivity curves.
"""

def Phyinf(parameters, min_rangeMat, max_rangeMat):
    device = parameters.device
    c_dtype = torch.cfloat
    batch_num = parameters.shape[0]

    # --- 1. 定义常量 (Constants) ---
    waveLength = 0.154
    k0 = (2 * torch.pi) / waveLength
    k1 = waveLength**2 / (2 * torch.pi)
    
    # 入射角, 形状: (1, 401) for broadcasting
    incidence_angle_inner = torch.linspace(0.5, 4, 512, device=device)
    incidence_angle_inner = (incidence_angle_inner / 180 * torch.pi).view(1, -1)
    # --- 2. 参数反归一化 (Parameter De-normalization) ---
    min_rangeMat = torch.tensor(min_rangeMat, dtype = torch.float, device = device)
    max_rangeMat = torch.tensor(max_rangeMat, dtype = torch.float, device = device)
    parameters = min_rangeMat + parameters*(max_rangeMat - min_rangeMat)
    # --- 3. 重组参数以匹配物理层 (Reshape Parameters for Layers) ---
    # 使用 unsqueeze 和 cat 在 batch 维度上操作
    zeros_batch_1d = torch.zeros(batch_num, 1, device=device) # (batch_size, 1)

    # 厚度: [vac, L1, L2, L3, sub], 形状: (batch_size, 5)
    thickness = torch.cat([zeros_batch_1d, parameters[:, 0:3], zeros_batch_1d], dim=1)
    # 粗糙度: [vac-L1, L1-L2, L2-L3, L3-sub], 形状: (batch_size, 4)
    roughness = parameters[:, 3:7]
    # 实部 SLD: [vac, L1, L2, L3, L4(sub)], 形状: (batch_size, 5)
    Re_sld = torch.cat([zeros_batch_1d, parameters[:, 7:11]], dim=1)
    # 虚部 SLD: [vac, L1, L2, L3, L4(sub)], 形状: (batch_size, 5)
    Im_sld = torch.cat([zeros_batch_1d, parameters[:, 11:15]], dim=1)
    
    # 增加一个维度以进行后续广播, e.g., (batch_size, 5, 1)
    thickness = thickness.unsqueeze(-1)
    roughness = roughness.unsqueeze(-1)
    Re_sld = Re_sld.unsqueeze(-1)
    Im_sld = Im_sld.unsqueeze(-1)

    # --- 4. 计算每层的物理属性 (Calculate Layer Properties) ---
    # 折射率 n, 形状: (batch_size, 5, 1)
    n = 1 - k1 * Re_sld.to(c_dtype) + 1j * k1 * Im_sld.to(c_dtype)
    lay_num = n.shape[1] # 5
    
    # 计算所有层的 k_z (波矢的z分量)
    # 利用 PyTorch 的广播机制一次性计算
    # n 的形状:          (batch_size, 5, 1)
    # incidence_angle: (1, 401)
    # 广播后, k 的形状为 (batch_size, 5, 401)
    n_substrate = n[:, 0:1, :] # 形状: (batch_size, 1, 1)
    k_cos_term = (k0 * n_substrate * torch.cos(incidence_angle_inner))**2
    k = -torch.sqrt((k0 * n)**2 - k_cos_term)

    # --- 5. 计算传递矩阵 (Calculate Transfer Matrices) ---
    # 这里的计算是针对 '界面' (interface) 的，所以有 lay_num - 1 个
    num_interfaces = lay_num - 1 # 4

    # 使用切片代替循环来获取相邻层
    k_top = k[:, :-1, :]    # (batch_size, 4, 401) Layers 0-3
    k_bottom = k[:, 1:, :]  # (batch_size, 4, 401) Layers 1-4
    
    k_add = k_top + k_bottom
    k_sub = k_top - k_bottom

    # p 和 m 矩阵元素, 形状: (batch_size, 4, 401)
    p = k_add / (2 * k_top) * torch.exp(-0.5 * (k_sub * roughness)**2)
    m = k_sub / (2 * k_top) * torch.exp(-0.5 * (k_add * roughness)**2)
    
    # 按照您的原始逻辑, ek 使用 k_top 和 thickness_top
    # 注意: thickness[0] (真空层) 为0, 所以 ek 的第一项为0
    ek = 1j * k_top * thickness[:, :-1, :] # 形状: (batch_size, 4, 401)
    h1 = torch.exp(-ek)
    h2 = 1 / h1

    # 构建 2x2 矩阵 M 和 H
    # 最终形状为: (batch_size, num_interfaces, len_ang, 2, 2)
    zeros_mat = torch.zeros_like(p)
    # 构建 M 矩阵
    M_row1 = torch.stack([p, m], dim=-1)
    M_row2 = torch.stack([m, p], dim=-1)
    M = torch.stack([M_row1, M_row2], dim=-2)
    # 构建 H 矩阵
    H_row1 = torch.stack([h1, zeros_mat], dim=-1)
    H_row2 = torch.stack([zeros_mat, h2], dim=-1)
    H = torch.stack([H_row1, H_row2], dim=-2)

    # 接口的传递矩阵 tra_Matrix = H @ M
    # 这是批处理矩阵乘法, 作用于最后两个维度
    tra_Matrix = H @ M # 形状: (batch_size, 4, 401, 2, 2)
    
    # --- 6. 迭代计算总传递矩阵 (Iteratively Compute Total Transfer Matrix) ---
    # 这是唯一需要保留的循环，因为它是一个顺序乘积
    len_ang = incidence_angle_inner.shape[1]
    
    # 初始化一个单位矩阵的批次
    transferMatrix = torch.eye(2, dtype=c_dtype, device=device).view(1, 1, 2, 2)
    transferMatrix = transferMatrix.expand(batch_num, len_ang, -1, -1)
    
    for j in range(num_interfaces):
        # 提取第 j 个界面的所有批次和角度的矩阵
        # tra_Matrix[:, j, :, :, :] -> 形状: (batch_size, 401, 2, 2)
        transferMatrix = transferMatrix @ tra_Matrix[:, j, :, :, :]

    # --- 7. 计算最终反射率 (Calculate Final Reflectivity) ---
    # r 的形状: (batch_size, 401)
    r = transferMatrix[..., 0, 1] / transferMatrix[..., 1, 1]
    R = torch.abs(r)**2
    
    # 添加一个小的 epsilon 防止 log(0)
    R_all = torch.log10(R + 1e-11)
   
    return R_all
