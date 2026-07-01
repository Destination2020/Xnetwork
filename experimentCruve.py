import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# Add path for Calculate module
sys.path.append(r'd:\Work_files\EUV-OCT\XRR\XRR_Software\Software')
import Calculate
import Draw_line

def main():
    # 1. Load Experimental Data
    angleExperiment = []
    reflectivityExperiment = []
    
    file_name = r'd:\Work_files\EUV-OCT\XRR\XRR_Software\DeepLearning\XRR_FCNN\AEExperiment\doublelayerAE.csv'
    
    print(f"Loading data from {file_name}...")
    try:
        with open(file_name, mode='r', newline='') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                if len(row) >= 2:
                    try:
                        angleExperiment.append(float(row[0]))
                        reflectivityExperiment.append(float(row[1]))
                    except ValueError:
                        continue # Skip header or malformed lines
    except FileNotFoundError:
        print(f"Error: File {file_name} not found.")
        return

    angle = np.array(angleExperiment)
    reflectivity = np.array(reflectivityExperiment)
    rng = np.random.default_rng()
    angle_scale = rng.uniform(0.998, 1.002)
    angle_offset = rng.uniform(-0.01, 0.01)
    intensity_scale = rng.uniform(0.95, 1.05)
    background_offset = rng.uniform(-4e-7, 4e-7)
    
    # Normalize reflectivity if needed (based on original code logic)
    # The original code used index 44, which causes values > 1. 
    # Commenting out as requested by user to keep original values.
    # if len(reflectivity) > 44:
    #    reflectivity = reflectivity / reflectivity[44]
    
    # 2. Setup Simulation Model
    # [Material Name, Thickness(nm?), Roughness(nm?), Density]
    # Note: Units depend on Calculate.py implementation. Usually Thickness/Roughness in Angstrom or nm?
    # Looking at Calculate.py, it takes inputs as is. 
    # In ExperimentDataProduction.py, thickness is ~20-80. 
    # Here Model has 10.3, 8.276. If units are nm, 10.3nm = 103A.
    # Let's assume the Model definition is what the user wants.
    
    Model = [
        ['Air', 0.0, 0.33, -1],
        ['TiO2', 10.3, 0.7, 4.2],
        ['0.75SrNb2O6 + 0.25BaNb2O6', 8.276, 0.5, 5.28],
        ['Si', 0.0, 0.0, -1],
    ]
    
    waveLength1 = 0.154056
    incidence_angle = (angle_scale * angle + angle_offset) / 180 * np.pi
    Energy1 = [1239.841984 / waveLength1]
    
    # 3. Calculate Theoretical Curve
    # Change CWD to Software directory because material_parameter.py uses relative paths for 'sf' folder
    software_dir = r'd:\Work_files\EUV-OCT\XRR\XRR_Software\Software'
    original_cwd = os.getcwd()
    
    try:
        if os.path.exists(software_dir):
            os.chdir(software_dir)
        else:
            print(f"Warning: Software directory {software_dir} not found. Calculation might fail.")

        Layer_system = Calculate.Layer_model()
        Layer_system.add_model(Model, Energy1, 1)
        
        # Calculate.Angle_matrix returns r, R, Q
        # Note: Calculate.Angle_matrix might return lists or arrays
        r1, R1, Q1 = Calculate.Angle_matrix(Layer_system, incidence_angle, waveLength1)
        R1 = np.array(R1)
    
    except Exception as e:
        print(f"Calculation Error: {e}")
        return
    finally:
        # Always restore CWD
        os.chdir(original_cwd)
    
    step_size = angle[1] - angle[0]
    sigma_deg = rng.uniform(0.015, 0.025)
    sigma_pixels = sigma_deg / step_size
    print(f"Step size: {step_size:.5f} deg")
    print(f"Sigma (deg): {sigma_deg:.5f}")
    print(f"Sigma (pixels/points): {sigma_pixels:.5f}")
    print(f"Angle scale: {angle_scale:.6f}")
    print(f"Angle offset (deg): {angle_offset:.5f}")
    print(f"Intensity scale: {intensity_scale:.5f}")
    print(f"Background offset: {background_offset:.3e}")
    R_smeared = gaussian_filter1d(R1, sigma=sigma_pixels, mode='nearest')
    background_mean = 1.08e-6
    background_std = 3.6e-7
    white_noise = rng.normal(0.0, background_std, size=R1.shape)
    smooth_noise = gaussian_filter1d(white_noise, sigma=1.0, mode='nearest')
    R_smeared = intensity_scale * R_smeared + background_mean + background_offset + smooth_noise
    R_smeared = np.maximum(R_smeared, 1e-10)
    Draw_line.Draw_line(
        (list(angle), list(reflectivity), R_smeared), 
        xlabel=['deg'], 
        yscale='log',
        ylabel=['Reflectivity'], 
        sub_title=[f'Comparison'],
        legend=[{'loc': 'upper right', 'fontsize': 10, 'title': 'Legend'}],
        label=[['Experiment', 'Simulation (Smeared+Noise)']]
    )
    
    # Optional: Save comparison data
    # data_block = np.column_stack((angle, reflectivity, R_smeared))
    # np.savetxt('AEExperiment_comparison.dat', data_block, delimiter='\t', fmt='%.8e', header='Angle\tExp_R\tSim_R')

if __name__ == '__main__':
    main()
