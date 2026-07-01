import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from matplotlib.ticker import LogLocator
import time
#x_yDatas可以输入多个，输入为数组格式(x1,y1,y11...), (x2,y2,y22...)...可以连续输入多对，对应多张图中多条数据
#xlabel ylabel定义两坐标轴的标目,如果有多张图，定义相应数量的坐标轴标目，figsize代表图形大小，默认是（6,6）
#只有一张数据图时，xlabel,ylabel,sub_title,legend,label也需要采用列表格式，如xlabel=['Wavelength(nm)'],ylabel=['R'],sub_title=['Name'],legend = [{'loc': 'upper right', 'fontsize': 10, 'title': 'Calculation method'}],label=[['Line']]
#xlabel,ylabel,sub_title,legend,label不输入会调用默认定义
#如果想要定义保存的图形名称，请调用函数时加入filename='需要的名字';没有定义名字，默认使用时间命名
#xscale yscale 同plt.xscale,plt.yscale，可以使用'log'等  yscale_min控制图像的最低点下移多少倍，yscale_max控制上移
#自动存储数据图，存在程序所在目录下

def Draw_line(*x_yDatas, title='Data Plot', xlabel='X Axis', ylabel='Y label', yscale_min=1,yscale_max=1, figsize=(6,6), filename=None, xscale='linear', yscale='linear', sub_title='figure',legend=None, label=None):
    n = len(x_yDatas)
    if n==0:
        raise ValueError("至少有一条数据对")
    if ((len(xlabel)!=len(ylabel)) or (len(xlabel)!=n)):
        raise ValueError("坐标轴标目要和数据量相匹配")
    if legend!=None:
        if len(legend)!=n:
            raise ValueError("图例数目要和图形数量相匹配")
    fig, axes = plt.subplots(1, n, figsize=figsize*np.array([n,1]))
    fig.suptitle(title, fontsize=16)  # 设置整个图形的标题
    cmap = plt.get_cmap('tab10')                #线条颜色盘
    if n==1:
        axes = [axes]
        
    for i,Datas in enumerate(x_yDatas):
        for j in range(len(Datas)-1):
            color = cmap((3*i+j)%10)
            if label==None:
                axes[i].plot(Datas[0], Datas[j+1], color=color, label=f'Line{j+1}')    
            else:
                axes[i].plot(Datas[0], Datas[j+1], color=color, label=label[i][j])    
        axes[i].set_xlabel(xlabel[i], fontsize=13)
        axes[i].set_ylabel(ylabel[i], fontsize=13)
        axes[i].set_xscale(xscale)
        axes[i].set_yscale(yscale)
        
        y_min = min(min(y) for y in Datas[1:])
        y_max = max(max(y) for y in  Datas[1:])
        if y_min>0:
            y_axes1 = y_min*yscale_min 
        else:
            y_axes1 = y_min*(1/yscale_min)
        if y_max>0:
            y_axes2 = y_max*yscale_max
        else:
            y_axes2 = y_max*(1/yscale_max)   
        axes[i].set_ylim(y_axes1, y_axes2)

        if (n>1) and (sub_title=='figure'):
            axes[i].set_title(f'figure {i}')
        elif (n==1) and (sub_title=='figure'):
            pass
        else:
            axes[i].set_title(sub_title[i], fontsize=14)
        if (n==1) and (legend==None):
            pass
        elif legend==None:
            axes[i].legend()
        else:
            axes[i].legend(**legend.pop(0))
        axes[i].grid(True, which='both', axis='both', linestyle='--', color='0.5')

    plt.xticks(fontsize=12, color='black', fontname='Times New Roman')
    plt.yticks(fontsize=12, color='black', fontname='Times New Roman')    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 留出一些空间给标题,调整子图间的间距，避免重叠
    
    time.sleep(0.1)
    if filename==None:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")  # 生成当前时间的时间戳
        filename = f'{current_time}.png'
    
    plt.savefig(filename, dpi = 300)
    plt.show(block=False)


    