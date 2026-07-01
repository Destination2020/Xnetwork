import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import math
import sys
#from anaklasis import ref          #anaklasis 
import material_parameter as mp
import re
import Calculate
#import ana_cal
import Draw_line
import store_data
import circle_str
import time
from datetime import datetime

def main():
    #Each layer's infomation of the material
    #输入每层结构的参数，厚度和粗糙度的单位是nm，密度的单位是g/cm^3
    #结构的名称基本形式为'Air','SiO2'形式，元素首字母要大写，单个元素也要大写，对于混合物可按照比例采用'0.2Fe2O3 + 0.8CaCO3'形式
    #输入单质可将密度设为-1，调用默认密度;对于化合物需要输入对应密度，否则调用首元素的密度作为该化合物的密度
    #Model = circle_str.circle_str()

    # 创建一个密度从2.0到2.2线性变化的100层碳层
    
    Model = [
        ['Air', 0.0, 0.33, -1],
        ['TiO2', 10.3, 0.7, 4.2],
        ['0.75SrNb2O6 + 0.25BaNb2O6', 8.276, 0.5, 5.28],
        ['Si', 0.0, 0.0, -1]
    ]

    '''
    Model=[
        # description thk rough density
        ['Air', 0.0, 0.0, -1],
        ['Au', 40.0, 1.0, -1],
        ['Si', 200.0, 0.0, -1],
        ['Fe', 4.0, 0.0, -1],
        ['SiO2', 200.0, 0.0, -1],
        ['Na2O', 0.0, 0.0, 2.27],
        ['P2O5', 0.0, 0.0, 2.39],
        ['Mo', 0.0, 0.0, -1],
        ['HfO2', 0.0, 0.0, 9.68],
        ['Al2O3', 0.0, 0.0, 4],
        ['TiN', 0.0, 0.0, 5.4],
        ['GaAs', 0.0, 0.0, 5.32],
        ['ZrO2', 0.0, 0.0, 5.68]
    ]
    '''

    #以下采用的物理量带有1的是表示角度改变，波长固定；带有2的表示波长改变，角度固定
    #%%
    kind = 1   #"0"为分区调用(Henke>NIST>RATB)"1"为Henke,"2"为NIST数据，"3"为RATB数据，Henke:30-30000eV,Nist:11-433000eV,Ratb:1-1MeV范围大小是浮动的，如果遇到能量范围调整请求，可以尝试向中心缩小能量范围。
    method = 1  #“1”使用的计算方法是传递矩阵法，“2”使用的计算方法是一次反射近似法，省略掉了层状结构间的多次反射。
    #Transmission.Transmission([['SiO2', 20.0, 0.0, 2.2],['Si',300.0,0.0,-1],['SiO2', 20.0, 0.0, 2.2]], np.linspace(35,400,1001),[90], 1)  #透过率计算,具体看Transmission.py



    #%%能量和波长固定，角度改变
    
    #incidence_angle1_input = np.array(list(np.geomspace(0.5, 7, 1000, endpoint=False, dtype = np.float64)) + list(np.linspace(1, 4, 300,endpoint=True, dtype = np.float64)))
    #incidence_angle1_input = np.linspace(0.0001, 1, 512, dtype = np.float64)
    incidence_angle1_input = np.geomspace(0.1,7,1024, dtype = np.float64)                   #输入角度范围，np.linspace(5,30,1001)/180*np.pi是5°到35°之间生成1001个采样点
    waveLength1 = 0.154                                            #输入定量的波长，单位是nm,使用Henke时不要小于0.413和大于41.3
    
    incidence_angle1 = np.float64(incidence_angle1_input/180.0*np.pi)                                                        
    Energy1 = [1239.841984/waveLength1] 
    Layer_system = Calculate.Layer_model()
    Layer_system.add_model(Model, Energy1,kind) 
    
    if method==1:              #对传递矩阵和一次反射近似的方法数据都进行了记录，方便比较
        r1,R1,Q1 = Calculate.Angle_matrix(Layer_system,incidence_angle1,waveLength1)          #matrix后缀的是传递矩阵法,neglect后缀是一次反射近似法
        #r1_,R1_,Q1_ = Calculate.Angle_neglect(Layer_system,incidence_angle1,waveLength1)     
    elif method==2:
        r1,R1,Q1 = Calculate.Angle_neglect(Layer_system,incidence_angle1,waveLength1)
        r1_,R1_,Q1_ = Calculate.Angle_matrix(Layer_system,incidence_angle1,waveLength1)
      
    


    #%%角度固定，能量和波长改变
    incidence_angle2_input = 0.4                               #输入定量的角度
    Energy2  =  np.linspace(3000,1000,512, dtype = np.float64)                          #输入能量范围，np.linspace(30,30000,1001)表示30eV到30000eV
    waveLength2 = 1239.841984/Energy2


    incidence_angle2 = incidence_angle2_input/180*np.pi                                  
    Layer_system = Calculate.Layer_model()
    Layer_system.add_model(Model, Energy2,kind) 
    
    if method==1:
        r2,R2,Q2 = Calculate.Energy_matrix(Layer_system,incidence_angle2,waveLength2) 
        r2_,R2_,Q2_ = Calculate.Energy_neglect(Layer_system,incidence_angle2,waveLength2) 
    elif method==2:
        r2,R2,Q2 = Calculate.Energy_neglect(Layer_system,incidence_angle2,waveLength2)
        r2_,R2_,Q2_ = Calculate.Energy_matrix(Layer_system,incidence_angle2,waveLength2)



    #%%画图和存储数据
    Draw_line.Draw_line((incidence_angle1_input, R1),(waveLength2, R2), (Energy2, R2), xlabel=['\u03B8(°)', '\u03BB(nm)','E(eV)'],xscale = 'linear', yscale = 'log',
                        ylabel=['R', 'R', 'R'], sub_title=[f'Wavelength={waveLength1}nm',f'Angel={incidence_angle2_input}°', f'Angel={incidence_angle2_input}°'],  
                        legend = [{'loc': 'upper right', 'fontsize': 10, 'title': 'Calculation method'},{'loc': 'upper right', 'fontsize': 10, 'title': 'Calculation method'},{'loc': 'upper right', 'fontsize': 10, 'title': 'Calculation method'}],
                        label=[['matrix'],['matrix'],['matrix']])    #详见Draw_line.py文件，注意每次使用都会存储数据图
    '''
    Draw_line.Draw_line((Q1, R1,R1_),(Q2,R2,R2_), xlabel=['Q(A^-1)', 'Q(A^-1)'],    #Q的单位是Angstrom=0.1nm
                        ylabel=['R', 'R'], yscale='log', sub_title=[f'Wavelength={waveLength1_input}nm',f'Angel={incidence_angle2_input/np.pi*180}°'],  
                        legend = [{'loc': 'upper right', 'fontsize': 10, 'title': 'Calculation method'},{'loc': 'upper right', 'fontsize': 10, 'title': 'Calculation method'}],
                        label=[['matrix','neglect'],['matrix','neglect']])    #详见Draw_line.py文件，注意每次使用都会存储数据图
    '''
    store_data.store_txt(Model)
    storeint = input('输入1将数据存储为表格形式')
    if storeint=='1':
        Initial_real, Initial_imag = store_data.get_sld(Model, Energy1, 1)
        # Model: [Air, TiO2, SBN, Au, Mo, Si] -> 6 layers
        # Initial_real, Initial_imag: 6 elements corresponding to each layer

        # Build parameter arrays
        params = np.array([
            Model[1][1], Model[2][1], Model[3][1], Model[4][1],       # TiO2, SBN, Au, Mo thickness
            Model[0][2], Model[1][2], Model[2][2], Model[3][2], Model[4][2],  # Air, TiO2, SBN, Au, Mo roughness
            Initial_real[1], Initial_real[2], Initial_real[3], Initial_real[4], Initial_real[5],  # TiO2~Si Re_sld
            Initial_imag[1], Initial_imag[2], Initial_imag[3], Initial_imag[4], Initial_imag[5]   # TiO2~Si Im_sld
        ])

        num_points = len(R1)
        num_params = len(params)

        # Create 2D array: each row has [R, param1, param2, ..., param16]
        Initial_data = np.zeros((num_points, 1 + num_params))
        Initial_data[:, 0] = R1
        for i in range(num_params):
            Initial_data[:, i + 1] = params[i]

        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        np.savetxt(f'D:/Work_files/EUV-OCT/XRR/XRR_Software/DeepLearning/XRR_FCNN/FCNN_LSTM3/{current_time}.txt', Initial_data, fmt='%.6e')
        print(f"保存成功，请查收")
    else:
        print(f"没有保存")
    
    
if __name__ == "__main__":
    main()














    