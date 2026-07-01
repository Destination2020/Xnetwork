import numpy as np
import Draw_line
from scipy.ndimage import gaussian_filter1d
import argparse
#定义N层材料的厚度、粗糙度、SLDs
#计算curves，500个点
#输出保存为txt文件

#with open("./XRR_FCNN/FCNN3/config.yaml", "r") as file:
#        config = yaml.safe_load(file)
waveLength = 0.154056
num_layer = 2
Angle_init = np.linspace(0.4000, 3.5000, 621, endpoint=True, dtype = np.float64)
#Angle_init = np.linspace(0.50002, 3.50002, 700, endpoint=True, dtype = np.float64)
#Angle_init = np.geomspace(0.1, 4, 400, dtype = np.float64)
Angle = Angle_init/180*np.pi
rng = np.random.default_rng()
min_thi = np.array([0, 5, 5, 0],  dtype = np.float64)
max_thi = np.array([0, 15, 15, 0],  dtype = np.float64)
min_rou = np.array([0, 0.0, 0.0],  dtype = np.float64)
max_rou = np.array([0.5, 1.0, 1.0],  dtype = np.float64)
'''
min_f1 = np.array([0.0, 8.0, 8.0, 8.0],  dtype = np.float64)
max_f1 = np.array([0.0, 80.0, 80.0, 80.0],  dtype = np.float64)
min_f2 = np.array([0.0, 0.1, 0.1, 0.1],  dtype = np.float64)
max_f2 = np.array([0.0, 10, 10, 10],  dtype = np.float64)
min_real_den = np.array([0.0, 1e-3, 1e-3, 1e-3],  dtype = np.float64)
max_real_den = np.array([0.0, 2e-2, 2e-2, 2e-2],  dtype = np.float64)
max_imag_den = np.array([0.0, 2e-3, 2e-3, 2e-3],  dtype = np.float64)
min_imag_den = np.array([0.0, 2e-5, 2e-5, 2e-5],  dtype = np.float64)
'''
min_real_den = np.array([0.0, 5e-4, 5e-4, 5e-4],  dtype = np.float64)
max_real_den = np.array([0.0, 1e-2, 1e-2, 1e-2],  dtype = np.float64)
max_imag_den = np.array([0.0, 1e-3, 1e-3, 1e-4],  dtype = np.float64)
min_imag_den = np.array([0.0, 5e-5, 5e-5, 1e-5],  dtype = np.float64)

import os

def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--stage", type=int, choices=[1, 2], default=1)
        parser.add_argument("--num_samples", type=int, default=1024)
        parser.add_argument("--output_dir", type=str, default="")
        return parser.parse_args()

def stage_config(stage):
        if stage == 1:
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
                        "sigma_deg_max": 0.023
                }
        return {
                "angle_scale_min": 0.998,
                "angle_scale_max": 1.002,
                "angle_offset_min": -0.01,
                "angle_offset_max": 0.01,
                "intensity_scale_min": 0.95,
                "intensity_scale_max": 1.05,
                "background_offset_min": -4e-7,
                "background_offset_max": 4e-7,
                "background_mean": 1.08e-6,
                "background_std": 1.6e-7,
                "noise_smooth_sigma": 1.0,
                "sigma_deg_min": 0.015,
                "sigma_deg_max": 0.025
        }

def main():
        args = parse_args()
        cfg = stage_config(args.stage)
        output_dir = args.output_dir if args.output_dir else f'D:/Work_files/EUV-OCT/XRR/XRR_Software/DeepLearning/data/train_data/train_exp_stage{args.stage}'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        for num in range(args.num_samples):
                thickness = rng.random(size = (1,4)) * (max_thi - min_thi) + min_thi
                thickness = thickness.flatten()
                roughness = np.float64(rng.normal(0.5, 0.2, size = (1,3)) * (max_rou - min_rou) + min_rou)
                roughness = np.array([max(min(r, m),n) for r, m, n in zip(roughness.flatten(), max_rou.flatten(), min_rou.flatten())])
                mask = roughness[:-1]>thickness[1:-1]
                roughness[:-1][mask] = 0.1 * thickness[1:-1][mask]

                '''
                f1_den = rng.normal(45, 20, size = (1,4))
                f1_den = np.float64([min(max(r,m),n) for r, m, n in zip(f1_den.flatten(), min_f1.flatten(), max_f1.flatten())])
                f2_den = rng.normal(0.08, 0.03, size = (1,4))*f1_den
                f2_den = np.float64([min(max(r,m),n) for r, m, n in zip(f2_den.flatten(), min_f2.flatten(), max_f2.flatten())])
                real_den = (0.00169700331398766*0.1*f1_den).flatten()
                imag_den = (0.00169700331398766*0.1*f2_den).flatten()
                imag_den = imag_den.flatten()
                '''
                real_den = (rng.random(size = (1,4)) * rng.random(size = (1,4)) * (max_real_den - min_real_den) + min_real_den).flatten()
                max_imag = [min(i) for i in zip(max_imag_den, 0.3*real_den)]
                imag_den = (rng.random(size = (1,4)) * (max_imag - min_imag_den) + min_imag_den).flatten()
                fuzzy_real_den = real_den * rng.uniform(0.5, 1.5, size=4)
                fuzzy_imag_den = imag_den * rng.uniform(0.5, 1.5, size=4)
                angle_scale = rng.uniform(cfg["angle_scale_min"], cfg["angle_scale_max"])
                angle_offset = rng.uniform(cfg["angle_offset_min"], cfg["angle_offset_max"])
                intensity_scale = rng.uniform(cfg["intensity_scale_min"], cfg["intensity_scale_max"])
                background_offset = rng.uniform(cfg["background_offset_min"], cfg["background_offset_max"])
                incidence_angle = (angle_scale * Angle_init + angle_offset)/180*np.pi
                n = N_angle(real_den, imag_den, waveLength)
                n = n.flatten()
                r, R, Q = Angle_matrix(thickness, roughness, n, incidence_angle, waveLength)

                step_size = Angle_init[1] - Angle_init[0]
                sigma_deg = rng.uniform(cfg["sigma_deg_min"], cfg["sigma_deg_max"])

                conv_sigma = sigma_deg / step_size

                R = gaussian_filter1d(R, sigma=conv_sigma, mode='nearest')

                white_noise = rng.normal(0.0, cfg["background_std"], size=R.shape)
                smooth_noise = gaussian_filter1d(white_noise, sigma=cfg["noise_smooth_sigma"], mode='nearest')
                R = intensity_scale * R + cfg["background_mean"] + background_offset + smooth_noise
                R = np.maximum(R, 1e-10)
                XRR_data = list(np.concatenate((R, fuzzy_real_den[1:], fuzzy_imag_den[1:], thickness[1:-1], roughness, real_den[1:], imag_den[1:], [sigma_deg, angle_offset]), axis =0))
                np.savetxt(os.path.join(output_dir, f'{num+1:07d}.txt'), XRR_data)
   
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
            p[j] = k_add/(2*k[j])*np.exp(-0.5*(k_sub*roughness[j])**2)
            m[j] = k_sub/(2*k[j])*np.exp(-0.5*(k_add*roughness[j])**2)
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
