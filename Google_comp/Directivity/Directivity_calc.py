import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
import pandas as pd
import math

def set_dev():
    
    sd.default.device[0] = 1
    sd.default.device[1] = 3
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
    numSamples = int(duration*fs)
    sine_signal = np.sin(2*np.pi*(freq/fs)*np.arange(numSamples)) + np.sin(2*np.pi*(261.63/fs)*np.arange(numSamples)) + np.sin(2*np.pi*(329.63/fs)*np.arange(numSamples))
    sine_signal = np.resize(sine_signal,(numSamples,1))
    signal_fft = fft(sine_signal, axis=0)/len(sine_signal) # Divide by the length to compensate the decay on amplitude on the calc of FFT.
    recorded = record(sine_signal,fs,[2],[2])
    recorded_fft = fft(recorded, axis=0)/len(recorded)
    gain = np.abs(recorded_fft[duration*freq]) / np.abs(signal_fft[duration*freq])
    #plt.plot(20*np.log10(2*np.abs(signal_fft)))
    #plt.show()
    return 10*math.log10(gain)
    
def main():
    
    set_dev()
    freq = input("Indique la frecuencia de medida: ")
    freq = int(freq)
    medidas = 8
    angles, gain = [], []
    
    for i in range(medidas):
        value = int(input(f"Cual es el ángulo de medida con respecto al micro?: "))
        angles.append(value*math.pi/180)
        print(f"Midiendo directividad para un ángulo de {angles[i]} y una frecuencia de {freq} Hz")
        value = play(freq)
        gain.append(value)
    
    # Copy 0 degree data to 360 degree for consistency
    index = angles.index(0)
    angles.append(2*math.pi)
    gain.append(gain[index])
    
    # Saving data to csv file to make tables
    directivity = f"Directivity_{freq}.csv"
    data = {'Freq': freq, 'Angle': angles, 'Gain': gain}
    df = pd.DataFrame(data)
    df.to_csv(directivity, index=False, decimal=',', sep=';')
    angle_interp = np.linspace(0, 2*np.pi, 500)
    gain_interp = np.interp(angle_interp,angles,gain)
    plt.figure(figsize=(6,6))
    ax = plt.subplot(111, projection='polar')
    ax.plot(angle_interp, gain_interp)
    ax.set_title(f"Directivity of microphone at {freq} Hz")
    ax.grid(True)
    plt.savefig(f"Directivity_{freq}.png")
    
if __name__ == "__main__":
    main()