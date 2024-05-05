import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
import pandas as pd
import math

def open_csv(file):
    
    df = pd.read_csv(file,sep=';',decimal=',')
    angles = df['Angle'].values
    gain = df['Gain'].values
    
    return angles,gain

def main():
    
    freq = int(input("Indique la frecuencia graficar: ")) 
    mic_number = 4
    mic_angle_sep = 90
    csv_file = f'CSV_files/Directivity_{freq}.csv'
    angles, gain = open_csv(csv_file)
    angles = np.degrees(angles).astype(np.float32)
    angles4 = angles.copy()
    gain4 = gain.copy()
    for i in range(mic_number):
        for angle in angles:
            
            ang = angle + mic_angle_sep*i
            if ang > 360:
                ang = ang - 360        
            index_orig = np.where(angles == angle)[0]       
            index = np.where(angles == ang)[0]
            index_n = np.where(angles4 == ang)[0]
            
            if len(index) == 0 and len(index_n) == 0:
                angles4 = np.append(angles4,ang)
                gain4 = np.append(gain4,gain[index_orig])               
            elif len(index) != 0:
                if gain[index_orig] >= gain4[index]:
                    gain4[index] = gain[index_orig] 
            elif len(index) == 0 and len(index_n) != 0:
                if gain[index_orig] >= gain4[index_n]:
                    gain4[index_n] = gain[index_orig]                     
    
    
    # Sort values before plotting and saving
    sorted_indices = np.argsort(angles4)
    angles4 = np.array(angles4)[sorted_indices]
    gain4 = np.array(gain4)[sorted_indices]
    
    if gain4[len(gain4)-1] > gain4[0]:
        gain4[0] = gain4[len(gain4)-1]

    
    # Angles to radians
    angles = np.deg2rad(angles)
    angles4 = np.deg2rad(angles4)

    angle_interp = np.linspace(0, 2*np.pi, 360)
    gain_interp = np.interp(angle_interp,angles,gain)
    gain_interp4 = np.interp(angle_interp,angles4,gain4)

    plt.figure(figsize=(10,5))
    ax = plt.subplot(121, projection='polar')
    ax.plot(angle_interp, gain_interp)
    ax.set_ylim(-25,0)
    ax.set_title(f"Directivity of 1 microphone at {freq} Hz")
    ax.grid(True)    
    ax = plt.subplot(122, projection='polar')
    ax.plot(angle_interp, gain_interp4)
    ax.set_ylim(-25,0)
    ax.set_title(f"Directivity of 4 microphones at {freq} Hz")
    ax.grid(True)
    plt.savefig(f"Figures/Directivity_1vs4_{freq}.png")

    

                     
if __name__ == "__main__":
    main()
