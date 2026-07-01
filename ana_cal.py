from anaklasis import ref          #anaklasis 
def ana_cal(model_A):
    project='none'
    resolution=[0.001]
    background = [1.0e-9]
    scale = [1.0]
    qmax = [0.7]
    patches=[1.0]
    global_param = []
    system_A=[model_A]
    results_A = ref.calculate(project, resolution, 
	patches, system_A, global_param, 
	background, scale, qmax, plot=False)
    return results_A

'''
    Model_B=[
        #  Re_sld  Im_sld   thk rough solv description
        [ 0.00e-5, 0.00e-7,  0 , 0.0, 0.0, 'air'],
        [ 2.005539e-5, 4.5666092e-7,  400 , 0.0, 0.0, 'Si'] ,         #Au_Re_sld = (19.32*196.9666*6.02214076*10^(23))*10^6*2.8179403*10^(-15)*f1(15.36)
        [ 1.24813614e-4, 1.28370013e-5, 20,  0.0, 0.0, 'Au'],           #Au_Im_sld = (19.32*196.9666*6.02214076*10^(23))*10^6*2.8179403*10^(-15)*f1(14.76)
        [ 2.005539e-5, 4.5666092e-7,  250 , 0.0, 0.0, 'Si'] ,
        [ 1.24813614e-4, 1.28370013e-5, 20,  0.0, 0.0, 'Au'],
        [ 2.005539e-5, 4.5666092e-7,  0.0 , 0.0, 0.0, 'Si'] 
    ]
    results_A = ana_cal.ana_cal(Model_B)            #anaklasis package
    plt.plot(results_A[("reflectivity")][:,0],results_A[("reflectivity")][:,1],color='#000000',alpha=0.5, linewidth=2, label='Anaklasis_contrast')

    '''
    