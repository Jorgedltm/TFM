import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
import pandas as pd
import math

def set_dev():
    
    sd.default.device[0] = 24
    sd.default.device[1] = 26
    sd.check_input_settings()
    sd.check_output_settings()
    print(sd.query_devices())

def record(testsignal,fs,inputChannels,outputChannels):

    sd.default.samplerate = fs
    sd.default.dtype = 'float32'
    print("Input channels:",  inputChannels)
    print("Output channels:", outputChannels)


    # Start the recording
    recorded = sd.playrec(testsignal, samplerate=fs, input_mapping = inputChannels,output_mapping = outputChannels)
    sd.wait()
      
    return recorded

def play(freq):
    
    duration = 3
    fs= 16000
    freq=freq
    silenceAtStart = 1
    numSamples = int(duration*fs)
    sine_signal = 0.5*np.sin(2*np.pi*(freq/fs)*np.arange(numSamples))
    sine_signal = np.resize(sine_signal,(numSamples,1))
    zerostart = np.zeros(shape = (silenceAtStart*fs,1))
    sine_signal = np.concatenate((zerostart, sine_signal), axis = 0)
    signal_fft = fft(sine_signal[16000:], axis=0)/len(sine_signal[16000:]) # Divide by the length to compensate the decay on amplitude on the calc of FFT.
    recorded = record(sine_signal,fs,[2],[2])
    recorded_fft = fft(recorded[16000:], axis=0)/len(recorded[16000:])
    gain = np.abs(recorded_fft[duration*freq]) / np.abs(signal_fft[duration*freq])
    #plt.plot(20*np.log10(2*np.abs(signal_fft)))
    #plt.show()
    return gain
    
def main():
    
    set_dev()
    freq = input("Indique la frecuencia de medida: ")
    freq = int(freq)
    medidas = 10
    angles, gain = [], []
    reps = 4
    
    for i in range(medidas):
        value_a = int(input(f"Cual es el ángulo de medida con respecto al micro?: "))
        value_g = 0
        for _ in range(reps):
           value_g += play(freq)
        value_g = value_g / reps
        angles.append(value_a*math.pi/180)
        print(f"Midiendo directividad para un ángulo de {angles[i]} y una frecuencia de {freq} Hz")
        gain.append(value_g)
    
    # Copy data of the first half of the circle to the second half
    for i in range(len(angles)):
       image_angle = 2*math.pi - angles[i]
       index = angles.index(angles[i])
       if image_angle != math.pi:
          angles.append(image_angle)
          gain.append(gain[index])

    gain = 10*np.log10(gain) # Normalize and calc gain in dB

    # Sort values before plotting and saving
    sorted_indices = np.argsort(angles)
    angles = np.array(angles)[sorted_indices]
    gain = np.array(gain)[sorted_indices]

    # Saving data to csv file to make tables
    directivity = f"CSV_files/Directivity_{freq}.csv"
    data = {'Freq': freq, 'Angle': angles, 'Gain': gain[:,0]}
    df = pd.DataFrame(data)
    df.to_csv(directivity, index=False, decimal=',', sep=';')
    angle_interp = np.linspace(0, 2*np.pi, 360)
    gain_interp = np.interp(angle_interp,angles,gain[:,0])
    plt.figure(figsize=(6,6))
    ax = plt.subplot(111, projection='polar')
    ax.plot(angle_interp, gain_interp)
    ax.set_ylim(-25,0)
    ax.set_title(f"Directivity of microphone at {freq} Hz")
    ax.grid(True)
    plt.savefig(f"Figures/Directivity_{freq}.png")
    
if __name__ == "__main__":
    main()
