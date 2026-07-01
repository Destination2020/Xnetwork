import numpy as np
def circle_str():
    mu_top = 4.14
    mu_bottom = 2.76
    sigma =0.0
    rou = 0.0
    period = 40
    gua_top = np.random.normal(mu_top, sigma, period)
    gua_bottom = np.random.normal(mu_bottom, sigma, period)
    Model = [['Air',0.0, rou, -1]]
    for i in range(period):
        Model = Model+[['Si', gua_top[i], rou, 2.33]]+[['Mo', gua_bottom[i], rou, 10.28]]
    Model = Model+[['SiO2', 0.0, 0.0, 2.2]]
    return Model
    
    
