#f1 and f2 of each material,  ues liner interpolation and log interpolation ,respectively return Re_sld and Im_sld
import numpy as np
import re
import sys
import Matpack


class Material:
    def __init__(self, desc, density, Energy,kind):
        self.Energy = Energy
        self.desc = desc
        self.density = density
        self.coe = np.float64(6.02214076e23*1e6*2.8179403e-15*1e-18)             #电子半径等系数，单位为/nm^-2
        self.Mat = Matpack.Matpack()
        self.kind = kind

    def Analyse(self):
        Material = {}
        formulas = re.split(r'\s*\+\s*',self.desc)
        for formula in formulas:
            formula_num = re.search(r'^\b\d+\.*\d*[e\-]*\d*', formula) 
            fuhao = re.search(r'%', formula)
            if fuhao!=None:
                if formula_num!=None:
                    formula_num = float(formula_num.group())*0.01
                else:
                    formula_num = 1
            else:
                if formula_num!=None:
                    formula_num = float(formula_num.group())
                else:
                    formula_num = 1
            formula_mat = re.findall(r'([A-Z][a-z]*)(\d*)', formula)
            for element, num in formula_mat:
                if num == '':
                    num = 1*formula_num
                else:
                    num = float(num)*formula_num
                if Material.get(element):
                    Material[element] = num+Material[element]
                else:
                    Material[element] = num
        return Material

    def SLD(self):
        Material = self.Analyse()
        for key in Material.keys():
            if key not in self.Mat:
                print(f"输入元素无法计算！")
                sys.exit()
        if (self.density<0) or (self.density==None) or (self.density==''):
            if self.desc in self.Mat:
                den = self.Mat.get(self.desc)[0]
            else:
                name = list(Material.items())
                name = name[0][0]
                den = self.Mat.get(name)[0]
                print(f"系统中没有预存输入物质密度，将元素{name}的密度{den}g/cm^3作为物质密度。")
        else:
            den = self.density
        Ma = 0
        f1, f2 = np.zeros(len(self.Energy), dtype = np.float64), np.zeros(len(self.Energy), dtype = np.float64)
        for keys, values in Material.items():
            ma1 = self.Mat.get(keys)
            Ma = Ma+values*ma1[1]
            f11, f22 = self.switch_desc(keys)
            f1 = f1+values*f11
            f2 = f2+values*f22
        Re_sld, Im_sld = self.sld(den,Ma,f1,f2)
        return Re_sld, Im_sld
    
    def readfile(self, filename,kind_part):
        def parse_data(line):
            num = re.findall(r'[\+\-]*\d+\.*\d*[Ee\-\+]*\d*', line)
            if (kind_part==1) or (kind_part==2):
                return np.float64(num[0]), np.float64(num[1]), np.float64(num[2])
            elif (kind_part==3):
                return np.float64(num[0]), np.float64(num[4]), -np.float64(num[1])
            
        with open(filename,'r',encoding='utf-8') as file:
            data = np.array([parse_data(line) for line in file.readlines()[1:]])
        if (kind_part==1) or (kind_part==2):
            mask = (data[:,1]>-200) & (data[:,1]<200) & (data[:, 2]<200)
            eV = data[:, 0][mask]
            f1 = data[:, 1][mask]
            f2 = data[:, 2][mask]
        elif(kind_part==3):
            eV, f1, f2 = [], [], []
            for i in range(data.shape[0]):
                if (i>=1) and (i<=data.shape[0]-2):
                    if (data[i, 1] >= 0.2*(data[i+1, 1] + data[i-1, 1])) and (data[i, 1] <= 5*(data[i+1, 1] + data[i-1, 1])) and (data[i, 2] <= 5*(data[i+1, 2] + data[i-1, 2])) and (data[i, 2] >= 0.2*(data[i+1, 2] + data[i-1, 2])):
                        eV.append(data[i, 0])
                        f1.append(data[i, 1])
                        f2.append(data[i, 2])
                else:
                    eV.append(data[i, 0])
                    f1.append(data[i, 1])
                    f2.append(data[i, 2])
                        
        return np.array(eV), np.array(f1), np.array(f2)

    '''
    def readfile(self, filename,kind_part):
        if (kind_part==1) or (kind_part==2):
            with open(filename,'r',encoding='utf-8') as file:
                k = 0
                eVV, f11, f22 = [], [], []
                for line in file.readlines():
                    if k==0:
                        k = 1
                    else:
                        num = re.findall(r'[\+\-]*\d+\.*\d*[Ee\-\+]*\d*', line)
                        eVV.append(float(num[0]))
                        f11.append(float(num[1]))
                        f22.append(float(num[2]))
            eV, f1, f2 = [], [], []
            for i in range(len(f11)):
                if (f11[i]>-200) and (f22[i]<200) and (f11[i]<200):
                    eV.append(eVV[i])
                    f1.append(f11[i])
                    f2.append(f22[i])
        elif (kind_part==3):
            with open(filename,'r',encoding='utf-8') as file:
                k = 0
                eVV, f11, f22 = [], [], []
                for line in file.readlines():
                    if k==0:
                        k = 1
                    else:
                        num = re.findall(r'[\+\-]*\d+\.*\d*[Ee\-\+]*\d*', line)
                        eVV.append(float(num[0]))
                        f11.append(float(num[4]))
                        f22.append(-float(num[1]))
            eV, f1, f2 = [], [], []
            for i in range(len(f11)):
                if (f11[i]>-200) and (f22[i]<200) and (f11[i]<200):
                    eV.append(eVV[i])
                    f1.append(f11[i])
                    f2.append(f22[i])
        return np.array(eV), np.array(f1), np.array(f2)
    '''

    def sld(self,density,weight,f1,f2):
        sld_coe = np.float64(density/weight*self.coe)
        Re_sld = f1*sld_coe
        Im_sld = f2*sld_coe
        return Re_sld, Im_sld

    def interpolation(self,eV, f1, f2, eV_part=None):
        if eV_part==None:
            f1_input, f2_input = [], []
            raise_errors = 1
            for i in range(len(self.Energy)):
                k = 0
                for index,num in enumerate(eV):
                    if index==0:
                        if self.Energy[i]==num:
                            f1_input.append(f1[index])
                            f2_input.append(f2[index])
                            k = 1
                            break
                    elif self.Energy[i]==num:
                        f1_input.append(f1[index])
                        f2_input.append(f2[index])
                        k = 1
                        break
                    elif self.Energy[i]<num and self.Energy[i]>eV[index-1]:
                        liner_Re = f1[index-1]+(f1[index]-f1[index-1])/(num-eV[index-1])*(self.Energy[i]-eV[index-1])
                        if (f2[index-1]>0) and (f2[index]>0):
                            log_Im = np.exp(np.log(f2[index-1])+(np.log(f2[index])-np.log(f2[index-1]))/(num-eV[index-1])*(self.Energy[i]-eV[index-1]))
                        else:
                            if f2[index-1]<=0:
                                f2[index-1] = 1e-6
                            if f2[index]<=0:
                                f2[index] = 1e-6
                            log_Im = np.exp(np.log(f2[index-1])+(np.log(f2[index])-np.log(f2[index-1]))/(num-eV[index-1])*(self.Energy[i]-eV[index-1]))
                        f1_input.append(liner_Re)
                        f2_input.append(log_Im)
                        k = 1
                        break
                if k==0:
                    raise_errors = 0
                    f1_input.append(-9999.0)
                    f2_input.append(1e-6)
            if raise_errors==0:
                raise ValueError("请调整能量范围Henke:30-30000,Nist:11-432900,Ratb:1-10000000或者边缘值向内靠拢")
        else:
            f1_input, f2_input = [], []
            raise_errors = 1
            for i in range(len(eV_part)):
                k = 0
                for index,num in enumerate(eV):
                    if index==0:
                        if eV_part[i]==num:
                            f1_input.append(f1[index])
                            f2_input.append(f2[index])
                            k = 1
                            break
                    elif eV_part[i]==num:
                        f1_input.append(f1[index])
                        f2_input.append(f2[index])
                        k = 1
                        break
                    elif eV_part[i]<num and eV_part[i]>eV[index-1]:
                        liner_Re = f1[index-1]+(f1[index]-f1[index-1])/(num-eV[index-1])*(eV_part[i]-eV[index-1])
                        if (f2[index-1]>0) and (f2[index]>0):
                            log_Im = np.exp(np.log(f2[index-1])+(np.log(f2[index])-np.log(f2[index-1]))/(num-eV[index-1])*(eV_part[i]-eV[index-1]))
                        else:
                            if f2[index-1]<=0:
                                f2[index-1] = 1e-6
                            if f2[index]<=0:
                                f2[index] = 1e-6
                            log_Im = np.exp(np.log(f2[index-1])+(np.log(f2[index])-np.log(f2[index-1]))/(num-eV[index-1])*(eV_part[i]-eV[index-1]))
                        f1_input.append(liner_Re)
                        f2_input.append(log_Im)
                        k = 1
                        break
                if k==0:
                    raise_errors = 0
                    f1_input.append(-9999.0)
                    f2_input.append(1e-6)
            if raise_errors==0:
                raise ValueError("请调整能量范围在1-10000000ev之间，或者边缘值向内靠拢")
        return np.array(np.float64(f1_input)), np.array(np.float64(f2_input))

    def AIR(self):
        f1 = np.zeros(len(self.Energy), dtype = np.float64)
        f2 = np.zeros(len(self.Energy), dtype = np.float64)
        return f1, f2

    def AU(self):
        name = 'au'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    
    def SI(self):
        name = 'si'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    
    def AC(self):
        name = 'ac'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    def AG(self):
        name = 'ag'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    def AL(self):
        name = 'al'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    def AR(self):
        name = 'ar'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    def AS(self):
        name = 'as'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    def B(self):
        name = 'b'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    def BA(self):
        name = 'ba'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    def BE(self):
        name = 'be'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    def BI(self):
        name = 'bi'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    def BR(self):
        name = 'br'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    def C(self):
        name = 'c'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    def CA(self):
        name = 'ca'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    def CD(self):
        name = 'cd'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    def CE(self):
        name = 'ce'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    def CL(self):
        name = 'cl'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    def CO(self):
        name = 'co'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    def CR(self):
        name = 'CR'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    def CS(self):
        name = 'cs'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    def CU(self):
        name = 'cu'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    def DY(self):
        name = 'dy'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    def ER(self):
        name = 'er'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    def EU(self):
        name = 'eu'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    def F(self):
        name = 'f'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    def FE(self):
        name = 'fe'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    def FR(self):
        name = 'fr'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    def GA(self):
        name = 'ga'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    def GD(self):
        name = 'gd'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    def GE(self):
        name = 'ge'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    def H(self):
        name = 'h'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    def HE(self):
        name = 'he'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    def HF(self):
        name = 'hf'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    def HG(self):
        name = 'hg'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    def HO(self):
        name = 'ho'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    def I(self):
        name = 'i'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    def IN(self):
        name = 'in'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    def IR(self):
        name = 'ir'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    def K(self):
        name = 'k'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    def KR(self):
        name = 'kr'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    def LA(self):
        name = 'la'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    def LI(self):
        name = 'li'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    def LU(self):
        name = 'lu'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    def MG(self):
        name = 'mg'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    def MN(self):
        name = 'mn'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    def MO(self):
        name = 'mo'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    def N(self):
        name = 'n'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    def NA(self):
        name = 'na'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    def NB(self):
        name = 'nb'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    def ND(self):
        name = 'nd'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    def NE(self):
        name = 'ne'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    def NI(self):
        name = 'ni'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    def O(self):
        name = 'o'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    
    def OS(self):
        name = 'os'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    def P(self):
        name = 'p'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    def PA(self):
        name = 'pa'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    def PB(self):
        name = 'pb'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    def PD(self):
        name = 'pd'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    def PM(self):
        name = 'pm'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    def PO(self):
        name = 'po'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    def PR(self):
        name = 'pr'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    def PT(self):
        name = 'pt'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    def RA(self):
        name = 'ra'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    def RB(self):
        name = 'rb'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    def RE(self):
        name = 're'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    def Rh(self):
        name = 'rh'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    def RN(self):
        name = 'rn'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    def RU(self):
        name = 'ru'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    def S(self):
        name = 's'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    def SB(self):
        name = 'sb'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    def SC(self):
        name = 'sc'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    def SE(self):
        name = 'se'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    def SM(self):
        name = 'sm'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    def SN(self):
        name = 'sn'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    def SR(self):
        name = 'sr'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    def TA(self):
        name = 'ta'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    def TB(self):
        name = 'tb'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    def TC(self):
        name = 'tc'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    def TE(self):
        name = 'te'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    def TH(self):
        name = 'th'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    def TI(self):
        name = 'ti'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    def TL(self):
        name = 'tl'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    def TM(self):
        name = 'tm'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    def U(self):
        name = 'u'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    def V(self):
        name = 'v'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    def W(self):
        name = 'w'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    def XE(self):
        name = 'xe'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    def Y(self):
        name = 'y'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    def YB(self):
        name = 'yb'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    def ZN(self):
        name = 'zn'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
    def ZR(self):
        name = 'zr'
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff')
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt')
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
#%%
    def switch_desc(self, desc):
        if desc=='air' or desc=='Air' or desc=='AIR':
            f1, f2 = self.AIR()
            return f1, f2
        name = desc.lower()
        if self.kind==1:
            eV, f1, f2 = self.readfile(f'./sf/{name}.nff', self.kind)
            f1_input, f2_input = self.interpolation(eV, f1, f2)
            return f1_input, f2_input
        elif self.kind==2:
            eV, f1, f2 = self.readfile(f'./sf/{name}.txt', self.kind)
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
        elif self.kind==3:
            eV, f1, f2 = self.readfile(f'./sf/{name}_RATB.txt', self.kind)
            f1_input, f2_input = self.interpolation(eV*1000, f1, f2)
            return f1_input, f2_input
        elif self.kind==0:
            eV1, eV2, eV3 = [], [], []
            f11_input, f21_input, f12_input, f22_input, f13_input, f23_input = [],[],[],[],[],[]
            for i in range(len(self.Energy)):
                if (self.Energy[i]>=30) and (self.Energy[i]<=30000):
                    eV1.append(self.Energy[i])
                elif (self.Energy[i]>=11) and (self.Energy[i]<=432900):
                    eV2.append(self.Energy[i])
                elif (self.Energy[i]>=1.0) and (self.Energy[i]<=1e7):
                    eV3.append(self.Energy[i])
            if len(eV1)>=1:
                eV, f1, f2 = self.readfile(f'./sf/{name}.nff', 1)
                f11_input, f21_input = self.interpolation(eV, f1, f2, eV_part=eV1)
            if len(eV2)>=1:
                eV, f1, f2 = self.readfile(f'./sf/{name}.txt', 2)
                f12_input, f22_input = self.interpolation(eV*1000, f1, f2,eV_part=eV2)
            if len(eV3)>=1:
                eV, f1, f2 = self.readfile(f'./sf/{name}_RATB.txt', 3)
                f13_input, f23_input = self.interpolation(eV*1000, f1, f2, eV_part=eV3)
            f1_input = list(f11_input)+list(f12_input)+list(f13_input)
            f2_input = list(f21_input)+list(f22_input)+list(f23_input)
            return np.array(np.float64(f1_input)),np.array(np.float64(f2_input))
            
        
        if desc=='au' or desc=='Au' or desc=='AU':
            f1, f2 = self.AU()
            return f1, f2
        elif desc=='air' or desc=='Air' or desc=='AIR':
            f1, f2 = self.AIR()
            return f1, f2
        elif desc=='si' or desc=='Si' or desc=='SI':
            f1, f2 = self.SI()
            return f1, f2
        elif desc=='ac' or desc=='Ac' or desc=='AC':
            f1, f2 = self.AC()
            return f1, f2
        elif desc=='ag' or desc=='Ag' or desc=='AG':
            f1, f2 = self.AG()
            return f1, f2
        elif desc=='al' or desc=='Al' or desc=='AL':
            f1, f2 = self.AL()
            return f1, f2
        elif desc=='ar' or desc=='Ar' or desc=='AR':
            f1, f2 = self.AR()
            return f1, f2
        elif desc=='as' or desc=='As' or desc=='AS':
            f1, f2 = self.AS()
            return f1, f2
        elif desc=='at' or desc=='At' or desc=='AT':
            f1, f2 = self.AT()
            return f1, f2
        elif desc=='B' or desc=='b':
            f1, f2 = self.B()
            return f1, f2
        elif desc=='ba' or desc=='Ba' or desc=='BA':
            f1, f2 = self.BA()
            return f1, f2
        elif desc=='be' or desc=='Be' or desc=='BE':
            f1, f2 = self.BE()
            return f1, f2
        elif desc=='bi' or desc=='Bi' or desc=='BI':
            f1, f2 = self.BI()
            return f1, f2
        elif desc=='br' or desc=='Br' or desc=='BR':
            f1, f2 = self.BR()
            return f1, f2
        elif desc=='C' or desc=='c':
            f1, f2 = self.C()
            return f1, f2
        elif desc=='ca' or desc=='Ca' or desc=='CA':
            f1, f2 = self.CA()
            return f1, f2
        elif desc=='cd' or desc=='Cd' or desc=='CD':
            f1, f2 = self.CD()
            return f1, f2
        elif desc=='ce' or desc=='Ce' or desc=='CE':
            f1, f2 = self.CE()
            return f1, f2
        elif desc=='cl' or desc=='Cl' or desc=='CL':
            f1, f2 = self.CL()
            return f1, f2
        elif desc=='co' or desc=='Co' or desc=='CO':
            f1, f2 = self.CO()
            return f1, f2
        elif desc=='cr' or desc=='Cr' or desc=='CR':
            f1, f2 = self.CR()
            return f1, f2
        elif desc=='cs' or desc=='Cs' or desc=='CS':
            f1, f2 = self.CS()
            return f1, f2
        elif desc=='cu' or desc=='Cu' or desc=='CU':
            f1, f2 = self.CU()
            return f1, f2
        elif desc=='dy' or desc=='Dy' or desc=='DY':
            f1, f2 = self.DY()
            return f1, f2
        elif desc=='er' or desc=='Er' or desc=='ER':
            f1, f2 = self.ER()
            return f1, f2
        elif desc=='eu' or desc=='Eu' or desc=='EU':
            f1, f2 = self.EU()
            return f1, f2
        elif desc=='f' or desc=='F':
            f1, f2 = self.F()
            return f1, f2
        elif desc=='fe' or desc=='Fe' or desc=='FE':
            f1, f2 = self.FE()
            return f1, f2
        elif desc=='fr' or desc=='Fr' or desc=='FR':
            f1, f2 = self.FR()
            return f1, f2
        elif desc=='ga' or desc=='Ga' or desc=='GA':
            f1, f2 = self.GA()
            return f1, f2
        elif desc=='gd' or desc=='Gd' or desc=='GD':
            f1, f2 = self.GD()
            return f1, f2
        elif desc=='ge' or desc=='Ge' or desc=='GE':
            f1, f2 = self.GE()
            return f1, f2
        elif desc=='h' or desc=='H':
            f1, f2 = self.H()
            return f1, f2
        elif desc=='he' or desc=='He' or desc=='HE':
            f1, f2 = self.HE()
            return f1, f2
        elif desc=='hf' or desc=='Hf' or desc=='HF':
            f1, f2 = self.HF()
            return f1, f2
        elif desc=='hg' or desc=='Hg' or desc=='HG':
            f1, f2 = self.HG()
            return f1, f2
        elif desc=='ho' or desc=='Ho' or desc=='HO':
            f1, f2 = self.HO()
            return f1, f2
        elif desc=='i' or desc=='I':
            f1, f2 = self.I()
            return f1, f2
        elif desc=='in' or desc=='In' or desc=='IN':
            f1, f2 = self.IN()
            return f1, f2
        elif desc=='ir' or desc=='Ir' or desc=='IR':
            f1, f2 = self.IR()
            return f1, f2
        elif desc=='k' or desc=='K':
            f1, f2 = self.K()
            return f1, f2
        elif desc=='kr' or desc=='Kr' or desc=='KR':
            f1, f2 = self.KR()
            return f1, f2
        elif desc=='la' or desc=='La' or desc=='LA':
            f1, f2 = self.LA()
            return f1, f2
        elif desc=='li' or desc=='Li' or desc=='LI':
            f1, f2 = self.LI()
            return f1, f2
        elif desc=='lu' or desc=='Lu' or desc=='LU':
            f1, f2 = self.LU()
            return f1, f2
        elif desc=='mg' or desc=='Mg' or desc=='MG':
            f1, f2 = self.MG()
            return f1, f2
        elif desc=='mn' or desc=='Mn' or desc=='MN':
            f1, f2 = self.MN()
            return f1, f2
        elif desc=='mo' or desc=='Mo' or desc=='MO':
            f1, f2 = self.MO()
            return f1, f2
        elif desc=='n' or desc=='N':
            f1, f2 = self.N()
            return f1, f2
        elif desc=='na' or desc=='Na' or desc=='NA':
            f1, f2 = self.NA()
            return f1, f2
        elif desc=='nb' or desc=='Nb' or desc=='NB':
            f1, f2 = self.NB()
            return f1, f2
        elif desc=='nd' or desc=='Nd' or desc=='ND':
            f1, f2 = self.ND()
            return f1, f2
        elif desc=='ne' or desc=='Ne' or desc=='NE':
            f1, f2 = self.NE()
            return f1, f2
        elif desc=='ni' or desc=='Ni' or desc=='NI':
            f1, f2 = self.NI()
            return f1, f2
        elif desc=='O' or desc=='o':
            f1, f2 = self.O()
            return f1, f2
        elif desc=='os' or desc=='Os' or desc=='OS':
            f1, f2 = self.OS()
            return f1, f2
        elif desc=='p' or desc=='P':
            f1, f2 = self.P()
            return f1, f2
        elif desc=='pa' or desc=='Pa' or desc=='PA':
            f1, f2 = self.PA()
            return f1, f2
        elif desc=='pb' or desc=='Pb' or desc=='PB':
            f1, f2 = self.PB()
            return f1, f2
        elif desc=='pd' or desc=='Pd' or desc=='PD':
            f1, f2 = self.PD()
            return f1, f2
        elif desc=='pm' or desc=='Pm' or desc=='PM':
            f1, f2 = self.PM()
            return f1, f2
        elif desc=='po' or desc=='Po' or desc=='PO':
            f1, f2 = self.PO()
            return f1, f2
        elif desc=='pr' or desc=='Pr' or desc=='PR':
            f1, f2 = self.PR()
            return f1, f2
        elif desc=='pt' or desc=='Pt' or desc=='PT':
            f1, f2 = self.PT()
            return f1, f2
        elif desc=='ra' or desc=='Ra' or desc=='RA':
            f1, f2 = self.RA()
            return f1, f2
        elif desc=='rb' or desc=='Rb' or desc=='RB':
            f1, f2 = self.RB()
            return f1, f2
        elif desc=='re' or desc=='Re' or desc=='RE':
            f1, f2 = self.RE()
            return f1, f2
        elif desc=='rh' or desc=='Rh' or desc=='RH':
            f1, f2 = self.RH()
            return f1, f2
        elif desc=='rn' or desc=='Rn' or desc=='RN':
            f1, f2 = self.RN()
            return f1, f2
        elif desc=='ru' or desc=='Ru' or desc=='RU':
            f1, f2 = self.RU()
            return f1, f2
        elif desc=='s' or desc=='S':
            f1, f2 = self.S()
            return f1, f2
        elif desc=='sb' or desc=='Sb' or desc=='SB':
            f1, f2 = self.SB()
            return f1, f2
        elif desc=='sc' or desc=='Sc' or desc=='SC':
            f1, f2 = self.SC()
            return f1, f2
        elif desc=='se' or desc=='Se' or desc=='SE':
            f1, f2 = self.SE()
            return f1, f2
        elif desc=='sm' or desc=='Sm' or desc=='SM':
            f1, f2 = self.SM()
            return f1, f2
        elif desc=='sn' or desc=='Sn' or desc=='SN':
            f1, f2 = self.SN()
            return f1, f2
        elif desc=='sr' or desc=='Sr' or desc=='SR':
            f1, f2 = self.SR()
            return f1, f2
        elif desc=='ta' or desc=='Ta' or desc=='TA':
            f1, f2 = self.TA()
            return f1, f2
        elif desc=='tb' or desc=='Tb' or desc=='TB':
            f1, f2 = self.TB()
            return f1, f2
        elif desc=='tc' or desc=='Tc' or desc=='TC':
            f1, f2 = self.TC()
            return f1, f2
        elif desc=='te' or desc=='Te' or desc=='TE':
            f1, f2 = self.TE()
            return f1, f2
        elif desc=='th' or desc=='Th' or desc=='TH':
            f1, f2 = self.TH()
            return f1, f2
        elif desc=='ti' or desc=='Ti' or desc=='TI':
            f1, f2 = self.TI()
            return f1, f2
        elif desc=='tl' or desc=='Tl' or desc=='TL':
            f1, f2 = self.TL()
            return f1, f2
        elif desc=='tm' or desc=='Tm' or desc=='TM':
            f1, f2 = self.TM()
            return f1, f2
        elif desc=='u' or desc=='U':
            f1, f2 = self.U()
            return f1, f2
        elif desc=='v' or desc=='V':
            f1, f2 = self.V()
            return f1, f2
        elif desc=='w' or desc=='W':
            f1, f2 = self.W()
            return f1, f2
        elif desc=='xe' or desc=='Xe' or desc=='XE':
            f1, f2 = self.XE()
            return f1, f2
        elif desc=='y' or desc=='Y':
            f1, f2 = self.Y()
            return f1, f2
        elif desc=='yb' or desc=='Yb' or desc=='YB':
            f1, f2 = self.YB()
            return f1, f2
        elif desc=='zn' or desc=='Zn' or desc=='ZN':
            f1, f2 = self.ZN()
            return f1, f2
        elif desc=='zr' or desc=='Zr' or desc=='ZR':
            f1, f2 = self.ZR()
            return f1, f2
        else:
            print("不包含输入材料，请重新输入。")
            sys.exit()