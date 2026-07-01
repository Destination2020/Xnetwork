import pandas as pd
import numpy as np
from datetime import datetime
import Calculate
import material_parameter as mp
import time
#可以输入多个数据及其对应的名称
def store_data(*Datas, Columns=None):
    Data_XRR = []                                                    #store the set of energy_change data as Excel file
    n = len(Datas)
    length = 0
    Column = []
    if (Columns!=[]) and (len(Columns)!=n):
        raise ValueError("数据名称要和数据量相匹配")
    for i, Data in enumerate(Datas):
        if isinstance(Data, (list, tuple, np.ndarray)):
            length = len(Data)
        elif length==0:
            length = 1
        else:
            pass
        if Columns==None:
                Column.append(f'Data{i+1}')
    for j in range(length):
        data_XRR = []
        for i, Data in enumerate(Datas):
            if isinstance(Data, (list, tuple, np.ndarray)):
                data_XRR.append(Data[j])
            else:
                data_XRR.append(Data)
        Data_XRR.append(data_XRR)
    if Columns==None:
        Columns=Column
    df = pd.DataFrame(Data_XRR, columns=Columns)
    time.sleep(0.1)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")  # 生成当前时间的时间戳
    filename = f'Data_{current_time}.xlsx'
    df.to_excel(filename,index = False)
    return
 
def store_txt(Model):
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")  # 生成当前时间的时间戳
    filename = f'Model_{current_time}.txt'
    with open(filename, "w") as f:
        for row in Model:
            f.write("\t".join(map(str, row)) + "\n")  # 每行写入列表中的一行
    return

#材料结构，能量范围，数据来源
def store_n(Model,Energy,kind):
    Energy = np.array(Energy)
    Layer_system = Calculate.Layer_model()
    Layer_system.add_model(Model, Energy,kind) 
    lay_num = len(Layer_system.LayerMatrix)
    Layer_n = Calculate.Layer_para_cal()    #k,n,delta,beta of each layer
    waveLength = 1239.841984/Energy
    Data_n = []
    Columns = []
    Re_sld = []
    Im_sld = []
    for LayerX in Layer_system.LayerMatrix:
        if len(Energy)==1:
            Layer_n.ref_index_matrix_angle(LayerX.Re_sld,LayerX.Im_sld,waveLength)   
        else:
            Layer_n.ref_index_matrix_energy(LayerX.Re_sld,LayerX.Im_sld,waveLength)
        Re_sld.append(LayerX.Re_sld)
        Im_sld.append(LayerX.Im_sld) 
    if len(Energy)==1:
        data_n = []
        data_n.append(Energy[0])
        Columns.append(f'Energy(eV)')
        data_n.append(waveLength[0])
        Columns.append(f'waveLength(nm)')
        for i in range(lay_num):
            data_n.append(Layer_n.Layer_para[i].n)
            data_n.append(Re_sld[i])
            data_n.append(Im_sld[i])
            Columns.append(f'[{Layer_system.LayerMatrix[i].desc}]:n')
            Columns.append(f'[{Layer_system.LayerMatrix[i].desc}]:Re_sld')
            Columns.append(f'[{Layer_system.LayerMatrix[i].desc}]:Im_sld')
        Data_n.append(data_n)
    else:
        Columns.append(f'Energy(eV)')
        Columns.append(f'waveLength(nm)')
        for i in range(lay_num):
            Columns.append(f'[{Layer_system.LayerMatrix[i].desc}]:n')
        for j in range(len(Energy)):
            data_n = []
            data_n.append(Energy[j])
            data_n.append(waveLength[j])
            for i in range(lay_num):
                data_n.append(Layer_n.Layer_para[i].n[j])
            Data_n.append(data_n)
    df = pd.DataFrame(Data_n, columns=Columns)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")  # 生成当前时间的时间戳
    filename = f'n_{current_time}.xlsx'
    df.to_excel(filename,index = False)
    return
        
def get_sld(Model,Energy,kind):
    Energy = np.array(Energy)
    Layer_system = Calculate.Layer_model()
    Layer_system.add_model(Model, Energy,kind) 
    lay_num = len(Layer_system.LayerMatrix)
    Layer_n = Calculate.Layer_para_cal()    #k,n,delta,beta of each layer
    waveLength = 1239.841984/Energy
    Data_n = []
    Columns = []
    Re_sld = []
    Im_sld = []
    for LayerX in Layer_system.LayerMatrix:
        Re_sld.append(LayerX.Re_sld)
        Im_sld.append(LayerX.Im_sld) 

    return np.array(Re_sld).flatten(), np.array(Im_sld).flatten()




    