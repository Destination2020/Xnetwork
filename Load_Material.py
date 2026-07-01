#输入材料文件，利用消光系数数和折射系数来进行反射率计算
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import math
import sys
# import material_parameter as mp
import re
# import Calculate
import Draw_line
import store_data
# import circle_str
import time
# import Transmission
from datetime import datetime
from scipy.interpolate import interp1d

def main():
        #[材料名称， 厚度nm 粗糙度nm  数据来源] 数字0代表折射率为1+0*1j，否则要给出来源文件地址
    Model = [
        ['Air', 0.0, 0.0, 0], 
        #['Ag', 10.0, 0.0, "D:/Work_files/EUV-OCT/XRR/XRR_Software/Material_lib/Ag_Stahrenberg.csv"]
        ['SiO2', 0.0, 0.00, "D:/Work_files/EUV-OCT/XRR/XRR_Software/Material_lib/Rodriguez-de_Marcos.csv"]
    ]
    thi = [mat[1] for mat in Model ]
    rou = [mat[2] for mat in Model ]


    # incidence_angle1_input = np.linspace(82, 78, 401, endpoint = True, dtype = np.float64)
    incidence_angle1_input = np.arange(0.1, 90, 0.1, dtype = np.float64)
    waveLength1 = 193.368                                            #输入定量的波长，单位是nm,使用Henke时不要小于0.413和大于41.3
    con_n1 = 5 #int(1 / 0.01 + 1)
    incidence_angle1 = np.float64(incidence_angle1_input/180.0*np.pi)
    n_k_lib = Material_n_k(Model , [waveLength1])
    r1, R1, Q1 = Angle_matrix_s(thi, rou, n_k_lib.refn, incidence_angle1, waveLength1) 
    r3, R3, Q3 = Angle_matrix_p(thi, rou, n_k_lib.refn, incidence_angle1, waveLength1)
    r1_angle = np.float64(np.angle(r1, deg = True))
    r3_angle = np.float64(np.angle(r3, deg = True))
    r_angle1 = r3_angle - r1_angle
    R1_con = con1d_n(R1, con_n1)
    R3_con = con1d_n(R3, con_n1)
    Rsub_S1 = abs(R1 - R1_con)
    Rdiv_S1 = Rsub_S1/R1
    Rsub_P1 = abs(R3 - R3_con)
    Rdiv_P1 = Rsub_P1/R3
    r_con1 = con1d_n(r_angle1, con_n1) 
    rsub1 = abs(r_angle1 - r_con1)                                                  

    incidence_angle2_input = 80                               #输入定量的角度
    incidence_angle2 = incidence_angle2_input/180*np.pi   
    # waveLength2 = np.linspace(190, 240, 2001, endpoint = True, dtype = np.float64)
    waveLength2 = np.arange(190-0.5, 240+0.501, 0.005,  dtype = np.float64)
    con_n2 = 5 #int(0.5 / 0.005 + 1)
    n_k_lib = Material_n_k(Model , waveLength2)
    r2, R2, Q2 = Energy_matrix_s(thi, rou, n_k_lib.refn, incidence_angle2, waveLength2)
    r4, R4, Q4 = Energy_matrix_p(thi, rou, n_k_lib.refn, incidence_angle2, waveLength2)
    R2_con = con1d_n(R2, con_n2)
    R4_con = con1d_n(R4, con_n2)
    r2_angle = np.angle(r2, deg = True)
    r4_angle = np.angle(r4, deg = True)
    r_angle2 = r4_angle - r2_angle
    r_con2 = con1d_n(r_angle2, con_n2) 
    Rsub_S2 = abs(R2 - R2_con)
    Rdiv_S2 = Rsub_S2/R2
    Rsub_P2 = abs(R4 - R4_con)
    Rdiv_P2 = Rsub_P2/R4
    rsub2 = abs(r_angle2 - r_con2)


    Draw_line.Draw_line((90-incidence_angle1_input, r1_angle, r3_angle),\
                        xlabel=['\u03B8(°)' ],xscale = 'linear', yscale = 'linear', \
                        ylabel=['Angle'], title='Phase_angle_P-S',sub_title=[f'Wavelength={waveLength1}nm'], \
                            legend = [{'loc': 'upper right', 'fontsize': 10, 'title': 'Calculation type'}],\
                        yscale_max=1.005, yscale_min=0.995,label=[['S', 'P']]) 

    '''
    #详见Draw_line.py文件，注意每次使用都会存储数据图
    con_n1_half_l = int(math.floor(con_n1/2))
    con_n1_half_r = int(math.floor((con_n1-1)/2))
    Draw_line.Draw_line((90-incidence_angle1_input[con_n1:-con_n1], R1[con_n1:-con_n1], R3[con_n1:-con_n1], R1_con[con_n1:-con_n1], R3_con[con_n1:-con_n1]),\
                        (90-incidence_angle1_input[con_n1:-con_n1], Rsub_S1[con_n1:-con_n1], Rsub_P1[con_n1:-con_n1]),\
                        (90-incidence_angle1_input[con_n1:-con_n1], Rdiv_S1[con_n1:-con_n1], Rdiv_P1[con_n1:-con_n1]),\
                            xlabel=['\u03B8(°)', '\u03B8(°)', '\u03B8(°)'],xscale = 'linear', yscale = 'log', \
                        ylabel=['R', 'R','R'], title='Reflectance_angle' ,sub_title=[f'Wavelength={waveLength1}nm',  f'Wavelength={waveLength1}nm', f'Wavelength={waveLength1}nm'], \
                            legend = [{'loc': 'upper right', 'fontsize': 10, 'title': 'Calculation type'}, \
                                      {'loc': 'upper right', 'fontsize': 10, 'title': 'Calculation type'}, \
                                      {'loc': 'upper right', 'fontsize': 10, 'title': 'Calculation type'}], \
                        yscale_max=1.002, yscale_min=0.998,label=[['S','P','S_con', 'P_con' ],\
                                                                   ['S:abs_error','P:abs_error'], \
                                                                    ['S:rel_error', 'P:rel_error']]) 
    
    con_n2_half_l = int(math.floor(con_n2/2))
    con_n2_half_r = int(math.floor((con_n2-1)/2))
    Draw_line.Draw_line((waveLength2[con_n2:-con_n2], R2[con_n2:-con_n2], R4[con_n2:-con_n2], R2_con[con_n2:-con_n2], R4_con[con_n2:-con_n2]),\
                        (waveLength2[con_n2:-con_n2], Rsub_S2[con_n2:-con_n2], Rsub_P2[con_n2:-con_n2]),\
                        (waveLength2[con_n2:-con_n2], Rdiv_S2[con_n2:-con_n2], Rdiv_P2[con_n2:-con_n2]),\
                            xlabel=[ '\u03BB(nm)', '\u03BB(nm)', '\u03BB(nm)'],xscale = 'linear', yscale = 'log', \
                        ylabel=['R', 'R','R'], title='Reflectance_wavelength' ,sub_title=[f'Angel={incidence_angle2_input}°', f'Angel={incidence_angle2_input}°', f'Angel={incidence_angle2_input}°'], \
                            legend = [{'loc': 'upper right', 'fontsize': 10, 'title': 'Calculation type'}, \
                                      {'loc': 'upper right', 'fontsize': 10, 'title': 'Calculation type'}, \
                                      {'loc': 'upper right', 'fontsize': 10, 'title': 'Calculation type'}], \
                        yscale_max=1.002, yscale_min=0.998,label=[['S','P','S_con', 'P_con' ],\
                                                                    ['S:abs_error','P:abs_error'],\
                                                                    ['S:rel_error', 'P:rel_error']]) 
    Draw_line.Draw_line((90-incidence_angle1_input[con_n1:-con_n1], r_angle1[con_n1:-con_n1], r_con1[con_n1:-con_n1]),\
                        (90-incidence_angle1_input[con_n1:-con_n1], rsub1[con_n1:-con_n1]),\
                        xlabel=['\u03B8(°)' ,'\u03B8(°)',],xscale = 'linear', yscale = 'log', \
                        ylabel=['Angle', 'Angle'], title='Phase_angle_P-S',sub_title=[f'Wavelength={waveLength1}nm',  f'Wavelength={waveLength1}nm'], \
                            legend = [{'loc': 'upper right', 'fontsize': 10, 'title': 'Calculation type'}, \
                                      {'loc': 'upper right', 'fontsize': 10, 'title': 'Calculation type'}],\
                        yscale_max=1.005, yscale_min=0.995,label=[['P-S', 'P-S_con'], ['P-S:abs_error']]) 

    Draw_line.Draw_line((waveLength2[con_n2:-con_n2], r_angle2[con_n2:-con_n2], r_con2[con_n2:-con_n2]),\
                        (waveLength2[con_n2:-con_n2], rsub2[con_n2:-con_n2]),\
                        xlabel=['\u03BB(nm)', '\u03BB(nm)'],xscale = 'linear', yscale = 'log', \
                        ylabel=['Angle', 'Angle'], title='Phase_waveP-S',sub_title=[f'Angel={incidence_angle2_input}°', f'Angel={incidence_angle2_input}°'], \
                            legend = [{'loc': 'upper right', 'fontsize': 10, 'title': 'Calculation type'}, \
                                      {'loc': 'upper right', 'fontsize': 10, 'title': 'Calculation type'}],\
                        yscale_max=1.005, yscale_min=0.995,label=[['P-S', 'P-S_con'],['P-S:abs_error']])
    time.sleep(0.1)
    if input('请输入数字1来保存')=='1':
        store_data.store_data(90-incidence_angle1_input,waveLength1,R1,R1_con,Rsub_S1,Rdiv_S1, R3, R3_con,Rsub_P1,Rdiv_P1, r_angle1,r_con1,rsub1,  \
                                Columns=['Angle(°)','Wavelength(nm)','R: S_R','R:S_con','S绝对误差','S相对误差','R :P_R','R :P_con','P绝对误差','P相对误差', 'P-S_angle','P-S_angle_con','P-S相位绝对误差'])
        time.sleep(0.1)
        store_data.store_data(90-incidence_angle2_input,waveLength2,R2,R2_con,Rsub_S2,Rdiv_S2, R4, R4_con,Rsub_P2,Rdiv_P2, r_angle2,r_con2,rsub2,  \
                                Columns=['Angle(°)','Wavelength(nm)','R: S_R','R:S_con','S绝对误差','S相对误差','R :P_R','R :P_con','P绝对误差','P相对误差', 'P-S_angle','P-S_angle_con','P-S相位绝对误差'])
    '''
    return

def con1d_5(R):
    kernel = np.ones(5)/5
    R_con = np.convolve(R, kernel, mode = 'same')
    R_con[:2] = R[:2]
    R_con[-2:] = R[-2:]
    return R_con
     

def con1d_n(R,n):
    kernel = np.ones(n)/n
    R_con = np.convolve(R, kernel, mode = 'same')
    R_con[:math.floor((n-1)/2)] = R[:math.floor((n-1)/2)]
    R_con[-math.floor(n/2):] = R[-math.floor(n/2):]
    return R_con
     

class Material_n_k:
    def __init__(self, Model, waveLength):
        self.waveLength = waveLength
        self.len = len(Model)
        self.dir = self.Dir(Model)
        self.wave, self.n, self.k = self.get_n_k(self.dir)
        self.n_inter, self.k_inter = self.interpolation_cubic(self.wave, self.n, self.k)
        self.refn = self.n_inter + 1j*self.k_inter

    def Dir(self, Model):
        dir = []
        for i in range(len(Model)):
            dir.append(Model[i][-1])
        return dir

    def get_n_k(self, dir_Mat):
        waveLength, n, k = [], [], []
        for dir_name in dir_Mat:
            if dir_name == 0:
                waveLength.append(np.array([0.0]))
                n.append(np.array([1.0]))
                k.append(np.array([0.0]))
            elif dir_name.lower().endswith('.csv'):
                df = pd.read_csv(dir_name)
                Mat_n_k = list(df['n'])
                Mat_n_k_idx = Mat_n_k.index('k')
                Mat_n = Mat_n_k[:Mat_n_k_idx]
                Mat_k = Mat_n_k[Mat_n_k_idx + 1:]
                wave = list(df['wl'])
                wave_idx = wave.index('wl')
                wave_Length = wave[:wave_idx]
                wave_Length = np.array([1000*np.float64(x) for x in wave_Length])
                Mat_n = np.array([np.float64(x) for x in Mat_n])
                Mat_k = np.array([np.float64(x) for x in Mat_k])

                waveLength.append(wave_Length)
                n.append(Mat_n)
                k.append(Mat_k)
        return waveLength, n, k

    def interpolation(self, wave_all, n_all, k_all):
        wave_input = self.waveLength
        n_output, k_output = [], []
        for sam in range(len(wave_all)):
            n_input, k_input = [], []
            wave = wave_all[sam]
            n = n_all[sam]
            k = k_all[sam]
            if len(wave) ==1 and wave[0]==0.0 :
                n_output.append(np.ones(len(wave_input)))
                k_output.append(np.zeros(len(wave_input)))
            else:
                raise_errors = 1
                for i in range(len(wave_input)):
                    k_idx = 0
                    for index,num in enumerate(wave):
                        if index==0:
                            if wave_input[i]==num:
                                n_input.append(n[index])
                                k_input.append(k[index])
                                k_idx = 1
                                break
                        elif wave_input[i]==num:
                            n_input.append(n[index])
                            k_input.append(k[index])
                            k_idx = 1
                            break
                        elif wave_input[i]<num and wave_input[i]>wave[index-1]:
                            liner_n = n[index-1]+(n[index]-n[index-1])/(num-wave[index-1])*(wave_input[i]-wave[index-1])
                            liner_k = k[index-1]+(k[index]-k[index-1])/(num-wave[index-1])*(wave_input[i]-wave[index-1])
                            n_input.append(liner_n)
                            k_input.append(liner_k)
                            k_idx = 1
                            break
                if k_idx==0:
                    raise ValueError("输入波长超出文件中波长范围！")
                n_output.append(np.array(np.float64(n_input)))
                k_output.append(np.array(np.float64(k_input)))
        return np.array(n_output), np.array(k_output)

    def interpolation_cubic(self, wave_all, n_all, k_all):
        wave_input = self.waveLength
        n_output, k_output = [], []
        for sam in range(len(wave_all)):
            wave = wave_all[sam]
            if len(wave) ==1 and wave[0]==0.0 :
                n_output.append(np.ones(len(wave_input)))
                k_output.append(np.zeros(len(wave_input)))
            else:
                f_n = interp1d(wave, n_all[sam], kind = 'cubic')
                f_k = interp1d(wave, k_all[sam], kind = 'cubic')
                if max(wave_input)>max(wave) or min(wave_input)<min(wave):
                    raise ValueError("输入波长超出文件中波长范围！")
                n_output.append(f_n(wave_input))
                k_output.append(f_k(wave_input))
        return np.array(n_output), np.array(k_output)     
        

def Energy_matrix_p(thi, rou, n, incidence_angle,waveLength):
    lay_num = len(thi)
    len_wav = len(waveLength)
    k0 = np.float64(2*np.pi/waveLength)
    #calculate k of ecah layer of each incidence angle
    k = np.zeros((lay_num,len_wav), dtype=complex)
    for j in np.arange(lay_num):
        k[j] = -k0*np.sqrt((n[j])**2-(n[0]*np.cos(incidence_angle))**2)
    
    m = np.zeros((lay_num-1,len_wav), dtype=complex)
    p = np.zeros((lay_num-1,len_wav), dtype=complex)
    ek = np.zeros((lay_num-1,len_wav), dtype=complex)
    for j in range(lay_num-1):
            p[j] = (n[j+1]**2*k[j]+n[j]**2*k[j+1])/(2*n[j+1]**2*k[j])
            m[j] = (n[j+1]**2*k[j]-n[j]**2*k[j+1])/(2*n[j+1]**2*k[j])
            '''
            k_add = k[j] + k[j+1]
            k_sub = k[j] - k[j+1]
            p[j] = k_add/(2*k[j])*np.exp(-0.5*(k_sub*rou[j])**2)
            m[j] = k_sub/(2*k[j])*np.exp(-0.5*(k_add*rou[j])**2)
            ek[j] = 1j*k[j]*thi[j]
            '''
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
    R = np.float64(np.abs(r)**2)
    Q = 0.2*k0*np.sin(incidence_angle)*np.abs(n[0])  #A^-1

    return r,R,Q

def Angle_matrix_p(thickness, roughness, n, incidence_angle, waveLength):
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
            
            p[j] = (n[j+1]**2*k[j]+n[j]**2*k[j+1])/(2*n[j+1]**2*k[j])
            m[j] = (n[j+1]**2*k[j]-n[j]**2*k[j+1])/(2*n[j+1]**2*k[j])
            '''
            k_add = k[j]+k[j+1]
            k_sub = k[j]-k[j+1]
            p[j] = k_add/(2*k[j])*np.exp(-0.5*(k_sub*roughness[j])**2)
            m[j] = k_sub/(2*k[j])*np.exp(-0.5*(k_add*roughness[j])**2)
            ek[j] = 1j*k[j]*thickness[j]
            '''
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
    R = np.float64(np.abs(r)**2)
    Q = 0.2*k0*np.sin(incidence_angle)*np.abs(n[0])  #Dividing by 10 is for the unit as A^-1
    return r,R,Q

def Energy_matrix_s(thi, rou, n, incidence_angle,waveLength):
    lay_num = len(thi)
    len_wav = len(waveLength)
    k0 = np.float64(2*np.pi/waveLength)
    #calculate k of ecah layer of each incidence angle
    k = np.zeros((lay_num,len_wav), dtype=complex)
    for j in np.arange(lay_num):
        k[j] = -k0*np.sqrt((n[j])**2-(n[0]*np.cos(incidence_angle))**2)
    
    m = np.zeros((lay_num-1,len_wav), dtype=complex)
    p = np.zeros((lay_num-1,len_wav), dtype=complex)
    ek = np.zeros((lay_num-1,len_wav), dtype=complex)
    for j in range(lay_num-1):
            k_add = k[j] + k[j+1]
            k_sub = k[j] - k[j+1]
            p[j] = k_add/(2*k[j])*np.exp(-0.5*(k_sub*rou[j])**2)
            m[j] = k_sub/(2*k[j])*np.exp(-0.5*(k_add*rou[j])**2)
            ek[j] = 1j*k[j]*thi[j]
        
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
    R = np.float64(np.abs(r)**2)
    Q = 0.2*k0*np.sin(incidence_angle)*np.abs(n[0])  #A^-1

    return r,R,Q

def Angle_matrix_s(thickness, roughness, n, incidence_angle, waveLength):
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
    R = np.float64(np.abs(r)**2)
    Q = 0.2*k0*np.sin(incidence_angle)*np.abs(n[0])  #Dividing by 10 is for the unit as A^-1
    return r,R,Q





if __name__ == "__main__":
    main()       












