import numpy as np
import material_parameter as mp
#The parameters of each layer, including Re_sld,Im_sld,thickness ,roughness and description
class Layer_info: 
    def __init__(self, Re_sld, Im_sld, thickness, roughness, description):
        self.Re_sld = Re_sld       #A^-2
        self.Im_sld = Im_sld       #A^-2
        self.thk = thickness       #A
        self.rough = roughness     #A
        self.desc = description
#Handling of input materials
class Layer_model:
    def __init__(self):
        self.LayerMatrix = []
    def add_lay(self, Layer, Energy,kind):         #根据输入的结构矩阵指定每一层的参数
        desc = Layer[0]
        thk = Layer[1]
        rough = Layer[2]
        density = Layer[3]
        Mat = mp.Material(desc, density, Energy,kind)
        Re_sld, Im_sld = Mat.SLD()
        LayerN = Layer_info(Re_sld,Im_sld,thk,rough,desc)
        self.LayerMatrix.append(LayerN)

    def add_lay_with_sld(self, Layer, Energy, kind):
        desc = Layer[0]
        thk = Layer[1]
        rough = Layer[2]
        if len(Layer) == 5:
            Re_sld = Layer[3]
            Im_sld = Layer[4]
        else:
            if desc == 'Air' and Layer[3] == -1:
                Re_sld = 0.0
                Im_sld = 0.0
            else:
                Mat = mp.Material(desc, Layer[3], Energy, kind)
                Re_sld_arr, Im_sld_arr = Mat.SLD()
                Re_sld = float(Re_sld_arr[0])
                Im_sld = float(Im_sld_arr[0])
        LayerN = Layer_info(Re_sld, Im_sld, thk, rough, desc)
        self.LayerMatrix.append(LayerN)

    def add_model(self, Model, Energy,kind):
        for layerX in Model:
            self.add_lay(layerX, Energy,kind)

    def add_model_with_sld(self, Model, Energy, kind):
        for layerX in Model:
            self.add_lay_with_sld(layerX, Energy, kind)
    
    


#传递矩阵calculate refractive index,delta and beta of each layer
class Layer_parameter:
    def __init__(self,n,delta,beta):
        self.n = n
        self.delta = delta
        self.beta = beta
    

class Layer_para_cal:
    def __init__(self):
        self.Layer_para = []

    def ref_index_matrix_angle(self,Re_sld,Im_sld,waveLength):
        k0 = np.float64(np.power(waveLength,2)/(2*np.pi))
        if isinstance(Re_sld, (list, np.ndarray)):
            delta = k0*Re_sld[0]
            beta = k0*Im_sld[0]
        else:
            delta = k0*Re_sld
            beta = k0*Im_sld
        refractive_index = 1-delta + 1j*beta
        layer_p = Layer_parameter(refractive_index,delta,beta)
        self.Layer_para.append(layer_p)
    
    def ref_index_matrix_energy(self,Re_sld,Im_sld,waveLength):
        #delta = np.zeros(len(waveLength),dtype=float)
        #beta = np.zeros(len(waveLength),dtype=float)
        #refractive_index = np.zeros(len(waveLength),dtype=complex)
        pi_2 = np.float64(2*np.pi)
        delta = np.power(waveLength,2)*Re_sld/pi_2
        beta = np.power(waveLength,2)*Im_sld/pi_2
        refractive_index = 1-(delta-beta*1j)
        layer_p = Layer_parameter(refractive_index,delta,beta)
        self.Layer_para.append(layer_p)

#隐藏代码适合阅读
'''       
def Angle_matrix(Layer_system,incidence_angle,waveLength):
    lay_num = len(Layer_system.LayerMatrix)
    Layer_n = Layer_para_cal()    #k,n,delta,beta of each layer
    for LayerX in Layer_system.LayerMatrix:
        Layer_n.ref_index_matrix_angle(LayerX.Re_sld,LayerX.Im_sld,waveLength)

    #calculate k of ecah layer of each incidence angle
    k = np.zeros((lay_num,len(incidence_angle)), dtype=complex)              
    for j in range(lay_num):
        for i in range(len(incidence_angle)):
            k[j][i] = -np.sqrt((2*np.pi/waveLength)**2*np.power(Layer_n.Layer_para[j].n,2)-(2*np.pi*Layer_n.Layer_para[0].n/waveLength)**2*np.cos(incidence_angle[i])**2)
                
    m = np.zeros((lay_num-1,len(incidence_angle)), dtype=complex)
    p = np.zeros((lay_num-1,len(incidence_angle)), dtype=complex)
    ek = np.zeros((lay_num-1,len(incidence_angle)), dtype=complex)
    for j in range(lay_num-1):
        for i in range(len(incidence_angle)):             
            #p[j][i] = (Layer_n.Layer_para[j+1].n**2*k[j][i]+Layer_n.Layer_para[j].n**2*k[j+1][i])/(2*Layer_n.Layer_para[j+1].n**2*k[j][i])
            #m[j][i] = (Layer_n.Layer_para[j+1].n**2*k[j][i]-Layer_n.Layer_para[j].n**2*k[j+1][i])/(2*Layer_n.Layer_para[j+1].n**2*k[j][i])
            p[j][i] = (k[j][i]+k[j+1][i])/(2*k[j][i])*np.exp(-1.0*np.power(k[j+1][i]-k[j][i],2)*Layer_system.LayerMatrix[j].rough**2/2)
            m[j][i] = (k[j][i]-k[j+1][i])/(2*k[j][i])*np.exp(-1.0*np.power(k[j+1][i]+k[j][i],2)*Layer_system.LayerMatrix[j].rough**2/2)
            ek[j][i] = complex(0,k[j][i]*Layer_system.LayerMatrix[j].thk)

    transferMatrix = []
    for i in range(len(incidence_angle)):
        tra_Matrix = np.array([[1,0],[0,1]], dtype=complex)
        for j in range(lay_num-1):
            h1 = np.exp(-ek[j][i])
            h2 = np.exp(ek[j][i])
            M1 = np.array([[p[j][i],m[j][i]],[m[j][i],p[j][i]]])       #refractive Matrix
            H1 = np.array([[h1,0],[0,h2]])     #transmission Matrix
            tra_Matrix = np.dot(tra_Matrix,H1)
            tra_Matrix = np.dot(tra_Matrix,M1)
        transferMatrix.append(tra_Matrix)

    r = []
    for tra_Matrix in transferMatrix:
        r.append(tra_Matrix[0][1]/tra_Matrix[1][1])
    R = np.zeros(len(incidence_angle))
    Q = np.zeros(len(incidence_angle))
    for i in range(len(incidence_angle)):
        R[i] = np.abs(r[i])**2
        Q[i] = 4*np.pi*np.sin(incidence_angle[i])/(waveLength*10)*np.abs(Layer_n.Layer_para[0].n)
    
    return r,R,Q
'''



def Angle_matrix(Layer_system,incidence_angle,waveLength):
    lay_num = len(Layer_system.LayerMatrix)
    Layer_n = Layer_para_cal()    #k,n,delta,beta of each layer
    for LayerX in Layer_system.LayerMatrix:
        Layer_n.ref_index_matrix_angle(LayerX.Re_sld,LayerX.Im_sld,waveLength)
    k0 = np.float64(2*np.pi/waveLength)
    len_ang = len(incidence_angle)
    #calculate k of ecah layer of each incidence angle
    k = np.zeros((lay_num,len_ang), dtype=complex)              
    for j in range(lay_num):
            k[j] = -np.sqrt((k0*Layer_n.Layer_para[j].n)**2-(k0*Layer_n.Layer_para[0].n*np.cos(incidence_angle))**2)

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
            p[j] = k_add/(2*k[j])*np.exp(-0.5*(k_sub*Layer_system.LayerMatrix[j].rough)**2)
            m[j] = k_sub/(2*k[j])*np.exp(-0.5*(k_add*Layer_system.LayerMatrix[j].rough)**2)
            ek[j] = 1j*k[j]*Layer_system.LayerMatrix[j].thk

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
    Q = 0.2*k0*np.sin(incidence_angle)*np.abs(Layer_n.Layer_para[0].n)  #Dividing by 10 is for the unit as A^-1
    return r,R,Q


   
def Energy_matrix(Layer_system,incidence_angle,waveLength):
    lay_num = len(Layer_system.LayerMatrix)
    len_wav = len(waveLength)
    k0 = np.float64(2*np.pi/waveLength)
    Layer_n = Layer_para_cal()    #k,n,delta,beta of each layer
    for LayerX in Layer_system.LayerMatrix:
        Layer_n.ref_index_matrix_energy(LayerX.Re_sld,LayerX.Im_sld,waveLength)

    #calculate k of ecah layer of each incidence angle
    k = np.zeros((lay_num,len_wav), dtype=complex)
    for j in np.arange(lay_num):
        k[j] = -k0*np.sqrt((Layer_n.Layer_para[j].n)**2-(Layer_n.Layer_para[0].n*np.cos(incidence_angle))**2)
        mask = (Layer_n.Layer_para[j].n < 0.0)
        k[j][mask] = -k[j][mask]
    
    m = np.zeros((lay_num-1,len_wav), dtype=complex)
    p = np.zeros((lay_num-1,len_wav), dtype=complex)
    ek = np.zeros((lay_num-1,len_wav), dtype=complex)
    for j in range(lay_num-1):
            '''
            p[j] = (Layer_n.Layer_para[j+1].n**2*k[j]+Layer_n.Layer_para[j].n**2*k[j+1])/(2*Layer_n.Layer_para[j+1].n**2*k[j])
            m[j] = (Layer_n.Layer_para[j+1].n**2*k[j]-Layer_n.Layer_para[j].n**2*k[j+1])/(2*Layer_n.Layer_para[j+1].n**2*k[j])
            '''
            k_add = k[j] + k[j+1]
            k_sub = k[j] - k[j+1]
            p[j] = k_add/(2*k[j])*np.exp(-0.5*(k_sub*Layer_system.LayerMatrix[j].rough)**2)
            m[j] = k_sub/(2*k[j])*np.exp(-0.5*(k_add*Layer_system.LayerMatrix[j].rough)**2)
            ek[j] = 1j*k[j]*Layer_system.LayerMatrix[j].thk

    h1 = np.exp(-ek)
    h2 = 1/h1
    M1T = np.array([[p, m], [m, p]])
    M1 = np.transpose(M1T, [2, 3, 0, 1])
    zeros = np.zeros((lay_num-1, len_wav), dtype = complex)
    H1T = np.array([[h1, zeros], [zeros, h2]])
    H1 = np.transpose(H1T, [2, 3, 0, 1])
    tra_Matrix = np.matmul(H1, M1)
    transferMatrix = np.tile(np.array([[1,0], [0, 1]], dtype = complex), (len_wav, 1, 1))
    for j in np.arange(lay_num-1):
        transferMatrix = np.matmul(transferMatrix, tra_Matrix[j])

    r = transferMatrix[:, 0, 1]/transferMatrix[:, 1, 1]
    R = np.abs(r)**2
    Q = 0.2*k0*np.sin(incidence_angle)*np.abs(Layer_n.Layer_para[0].n)  #A^-1

    return r,R,Q

    
def Energy_neglect(Layer_system,incidence_angle,waveLength):
    lay_num = len(Layer_system.LayerMatrix)
    len_wav = len(waveLength)
    k0 = 2*np.pi/waveLength
    theta = np.zeros((lay_num, len_wav),dtype=complex)
    Layer_n = Layer_para_cal()    #k,n,delta,beta of each layer
    for LayerX in Layer_system.LayerMatrix:
        Layer_n.ref_index_matrix_energy(LayerX.Re_sld,LayerX.Im_sld,waveLength)
    for j in range(lay_num):
        if j==0:
            theta[0] = np.pi/2-incidence_angle
        else:
            theta[j] = np.arcsin(Layer_n.Layer_para[j-1].n/Layer_n.Layer_para[j].n*np.sin(theta[j-1]))
    rs = np.zeros((lay_num-1,len_wav), dtype=complex)
    r = np.zeros(len_wav, dtype=complex)
    qd = np.zeros(len_wav, dtype=complex)
    for j in range(lay_num-1):
        n_last = Layer_n.Layer_para[j].n * np.cos(theta[j])
        n_next = Layer_n.Layer_para[j+1].n * np.cos(theta[j+1])
        rs[j] = (n_last - n_next) / (n_last + n_next)
        qd = qd + 2*k0*Layer_system.LayerMatrix[j].thk * n_last
        r = r+rs[j]*np.exp(1j*qd)
    R = np.abs(r)**2
    Q = 0.2 * k0 * np.sin(incidence_angle) * np.abs(Layer_n.Layer_para[0].n)
    
    return r,R,Q
    '''
def Energy_neglect(Layer_system,incidence_angle,waveLength):
    lay_num = len(Layer_system.LayerMatrix)
    len_wav = len(waveLength)
    k0 = 2*np.pi/waveLength
    theta = np.zeros((lay_num, len_wav),dtype=complex)
    Layer_n = Layer_para_cal()    #k,n,delta,beta of each layer
    for LayerX in Layer_system.LayerMatrix:
        Layer_n.ref_index_matrix_energy(LayerX.Re_sld,LayerX.Im_sld,waveLength)
    for j in range(lay_num):
        if j==0:
            theta[0] = incidence_angle
        else:
            theta[j] = np.arccos(Layer_n.Layer_para[j-1].n/Layer_n.Layer_para[j].n*np.cos(theta[j-1]))
    rs = np.zeros((lay_num-1,len(waveLength)), dtype=complex)
    r = np.zeros(len(waveLength), dtype=complex)
    qd = np.zeros(len(waveLength), dtype=complex)
    for j in range(lay_num-1):
        n_last = Layer_n.Layer_para[j].n * np.sin(theta[j])
        n_next = Layer_n.Layer_para[j+1].n * np.sin(theta[j+1])
        rs[j] = (n_last - n_next) / (n_last + n_next)
        qd = qd + 2*k0*Layer_system.LayerMatrix[j].thk * n_last
        r = r+rs[j]*np.exp(1j*qd)
    R = np.abs(r)**2
    Q = 0.2 * k0 * np.sin(incidence_angle) * np.abs(Layer_n.Layer_para[0].n)
    
    return r,R,Q
    '''

def Angle_neglect(Layer_system,incidence_angle,waveLength):
    lay_num = len(Layer_system.LayerMatrix)
    len_ang = len(incidence_angle)
    k0 = np.float64(2*np.pi/waveLength)
    Layer_n = Layer_para_cal()    #k,n,delta,beta of each layer
    for LayerX in Layer_system.LayerMatrix:
        Layer_n.ref_index_matrix_angle(LayerX.Re_sld,LayerX.Im_sld,waveLength)
    theta = np.zeros((lay_num,len_ang), dtype=complex) 
    for j in range(lay_num):
        if j==0:
            theta[0] = np.float64(np.pi/2-incidence_angle)
        else:
            theta[j] = np.arcsin(Layer_n.Layer_para[j-1].n/Layer_n.Layer_para[j].n*np.sin(theta[j-1]))
    rs = np.zeros((lay_num-1,len_ang), dtype=complex)
    r = np.zeros(len(incidence_angle), dtype=complex)
    qd = np.zeros(len(incidence_angle), dtype=complex)
    for j in range(lay_num-1):
        n_last = Layer_n.Layer_para[j].n * np.cos(theta[j])
        n_next = Layer_n.Layer_para[j+1].n * np.cos(theta[j+1])
        rs[j] = (n_last - n_next) / (n_last + n_next)
        qd = qd + 2 *k0* Layer_system.LayerMatrix[j].thk * n_last
        r = r+rs[j]*np.exp(1j*qd)
    
    R = np.abs(r)**2
    Q = 0.2 * k0 * np.sin(incidence_angle) * np.abs( Layer_n.Layer_para[0].n)
    
    return r,R,Q


def Energy_transmission(Layer_system,incidence_angle,waveLength):
    lay_num = len(Layer_system.LayerMatrix)
    len_wav = len(waveLength)
    k0 = 2*np.pi/waveLength
    Layer_n = Layer_para_cal()    #k,n,delta,beta of each layer
    for LayerX in Layer_system.LayerMatrix:
        Layer_n.ref_index_matrix_energy(LayerX.Re_sld,LayerX.Im_sld,waveLength)

    #calculate k of ecah layer of each incidence angle
    k = np.zeros((lay_num,len_wav), dtype=complex)
    for j in range(lay_num):
        k[j] = -np.sqrt((k0*Layer_n.Layer_para[j].n)**2- \
                           (k0*Layer_n.Layer_para[0].n*np.cos(incidence_angle))**2)
                
    m = np.zeros((lay_num-1,len(waveLength)), dtype=complex)
    p = np.zeros((lay_num-1,len(waveLength)), dtype=complex)
    ek = np.zeros((lay_num-1,len(waveLength)), dtype=complex)
    for j in range(lay_num-1):
        k_add = k[j]+k[j+1]
        k_sub = k[j]-k[j+1]
        p[j] = (k_add)/(2*k[j])*np.exp(-0.5*(k_sub*Layer_system.LayerMatrix[j].rough)**2)               
        m[j] = (k_sub)/(2*k[j])*np.exp(-0.5*(k_add*Layer_system.LayerMatrix[j].rough)**2) 
        ek[j] = 1j*k[j]*Layer_system.LayerMatrix[j].thk

    h1 = np.exp(-ek)
    h2 = 1/h1
    M1T = np.array([[p, m], [m, p]])
    M1 = np.transpose(M1T, [2, 3, 0, 1])
    zeros = np.zeros((lay_num-1, len_wav), dtype = complex)
    H1T = np.array([[h1, zeros], [zeros, h2]])
    H1 = np.transpose(H1T, [2, 3, 0, 1])
    tra_Matrix = np.matmul(H1, M1)
    transferMatrix = np.tile(np.array([[1, 0], [0, 1]]), (len_wav, 1, 1))
    for j in range(lay_num-1):
        transferMatrix = transferMatrix * tra_Matrix[j]
    
    t = 1/transferMatrix[:, 1, 1]
    T = np.abs(t)**2

    return t,T


def apply_resolution_broadening(R, x_axis, HWHM, mode='angle', Q=None):
    """
    应用仪器分辨率展宽效果

    参数:
        R: 反射率数组
        x_axis: 对应的x轴数据（角度，度）
        HWHM: 半高半宽（Half Width at Half Maximum）
            - mode='angle': 角度空间HWHM（度），常数
            - mode='q_constant': Q空间HWHM（Å⁻¹），常数δQ
            - mode='q_fractional': Q空间HWHM/Q比值，即δQ_HWHM/Q（REFLEX方式）
              注: REFLEX/refnx通常使用FWHM/Q，转换为HWHM/Q需除以2
              例: REFLEX中δQ_FWHM/Q = 2% → HWHM = 0.01
        mode: 展宽模式
            - 'angle': 角度空间高斯卷积，常数HWHM（原始方式）
            - 'q_constant': Q空间高斯卷积，常数δQ_HWHM
            - 'q_fractional': Q空间逐点高斯展宽，δQ_HWHM = HWHM × Q_i（REFLEX方式）
        Q: Q值数组（Å⁻¹），mode为'q_constant'或'q_fractional'时必须提供

    返回:
        R_broadened: 展宽后的反射率数组
    """
    if HWHM <= 0:
        return R

    if mode == 'angle':
        sigma = HWHM / np.sqrt(2 * np.log(2))
        dx = np.mean(np.diff(x_axis))
        window_size = int(4 * sigma / dx)
        if window_size % 2 == 0:
            window_size += 1
        if window_size < 3:
            window_size = 3
        x_kernel = np.linspace(-window_size * dx / 2, window_size * dx / 2, window_size)
        gaussian_kernel = np.exp(-x_kernel**2 / (2 * sigma**2))
        gaussian_kernel /= gaussian_kernel.sum()
        R_broadened = np.convolve(R, gaussian_kernel, mode='same')
        return R_broadened

    elif mode == 'q_constant':
        if Q is None:
            raise ValueError("Q空间展宽模式需要提供Q数组")
        sigma = HWHM / np.sqrt(2 * np.log(2))
        dQ = np.mean(np.diff(Q))
        window_size = int(4 * sigma / dQ)
        if window_size % 2 == 0:
            window_size += 1
        if window_size < 3:
            window_size = 3
        q_kernel = np.linspace(-window_size * dQ / 2, window_size * dQ / 2, window_size)
        gaussian_kernel = np.exp(-q_kernel**2 / (2 * sigma**2))
        gaussian_kernel /= gaussian_kernel.sum()
        R_broadened = np.convolve(R, gaussian_kernel, mode='same')
        return R_broadened

    elif mode == 'q_fractional':
        if Q is None:
            raise ValueError("Q空间展宽模式需要提供Q数组")
        R_broadened = np.zeros_like(R, dtype=float)
        for i in range(len(Q)):
            dQ_hwhm_i = HWHM * Q[i]
            if dQ_hwhm_i <= 0:
                R_broadened[i] = R[i]
                continue
            sigma_i = dQ_hwhm_i / np.sqrt(2 * np.log(2))
            dQ_arr = np.abs(Q - Q[i])
            cutoff = 4 * sigma_i
            mask = dQ_arr <= cutoff
            if not np.any(mask):
                R_broadened[i] = R[i]
                continue
            weights = np.exp(-dQ_arr[mask]**2 / (2 * sigma_i**2))
            weights /= weights.sum()
            R_broadened[i] = np.sum(R[mask] * weights)
        return R_broadened

    else:
        raise ValueError(f"不支持的展宽模式: {mode}")


def angle_hwhm_to_q_hwhm(HWHM_angle_deg, waveLength_nm):
    """
    将角度空间的HWHM转换为Q空间的HWHM（小角度近似，cosθ≈1）

    参数:
        HWHM_angle_deg: 角度空间HWHM（度）
        waveLength_nm: 波长（nm）

    返回:
        dQ_hwhm: Q空间HWHM（Å⁻¹）
    """
    HWHM_rad = HWHM_angle_deg * np.pi / 180.0
    dQ_hwhm = 4.0 * np.pi * HWHM_rad / (waveLength_nm * 10.0)
    return dQ_hwhm


def apply_background_noise(R, x_axis=None, background_type='power_law', 
                           background_params=None, add_random_noise=False,
                           noise_level=1e-6, random_seed=None):
    """
    添加背景噪声
    
    参数:
        R: 原始反射率数组
        x_axis: x轴数据（角度或能量），某些背景类型需要
        background_type: 背景类型
            - 'constant': 常数背景
            - 'power_law': 幂律背景（XRR实验常见，随角度增大）
            - 'exponential': 指数衰减背景
            - 'custom': 自定义背景（需提供background_params为数组）
        background_params: 背景参数
            - 'constant': [background_level]
            - 'power_law': [scale, exponent]，background = scale * x^(-exponent)
            - 'exponential': [scale, decay]，background = scale * exp(-x/decay)
            - 'custom': 与R等长的数组
        add_random_noise: 是否添加随机噪声
        noise_level: 随机噪声水平（标准差）
        random_seed: 随机种子（用于可重复性）
    
    返回:
        R_with_background: 添加背景和噪声后的反射率
    """
    if background_params is None:
        background_params = [1e-6]
    
    if background_type == 'constant':
        background = background_params[0] * np.ones_like(R)
        
    elif background_type == 'power_law':
        if x_axis is None:
            raise ValueError("power_law 背景类型需要提供 x_axis 参数")
        scale = background_params[0] if len(background_params) > 0 else 1e-6
        exponent = background_params[1] if len(background_params) > 1 else 2.0
        x_safe = np.where(x_axis > 0, x_axis, 1e-10)
        background = scale * np.power(x_safe, -exponent)
        
    elif background_type == 'exponential':
        if x_axis is None:
            raise ValueError("exponential 背景类型需要提供 x_axis 参数")
        scale = background_params[0] if len(background_params) > 0 else 1e-6
        decay = background_params[1] if len(background_params) > 1 else 1.0
        background = scale * np.exp(-x_axis / decay)
        
    elif background_type == 'custom':
        if len(background_params) != len(R):
            raise ValueError("custom 背景需要与R等长的数组")
        background = np.array(background_params)
        
    else:
        raise ValueError(f"不支持的背景类型: {background_type}")
    
    R_with_background = R + background
    
    if add_random_noise:
        if random_seed is not None:
            np.random.seed(random_seed)
        noise = np.random.poisson(R_with_background / noise_level) * noise_level
        R_with_background = R_with_background + noise
    
    R_with_background = np.maximum(R_with_background, 0)
    
    return R_with_background

