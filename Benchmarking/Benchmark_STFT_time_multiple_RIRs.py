import time
import jax.numpy as jnp
import cupy as cp
import scipy
from cupyx.scipy.signal import stft as stft_cp
from jax.scipy.signal import stft as stft_jax
from scipy.signal import stft as stft_sp
from librosa import stft as stft_lb
from jax import device_put
import matplotlib.pyplot as plt
import numpy as np
import librosa
import torch
import pandas as pd

def RIR_format(RIR,num_samples,original_length):
    max_index = np.argmax(RIR)
    min_index = np.argmin(RIR)
    
    if max_index < min_index:
        direct_sound_index = max_index
    else:
        direct_sound_index = min_index

    RIR = np.concatenate((np.zeros(num_samples), RIR[direct_sound_index:]), dtype=np.float32)
    if len(RIR) > original_length:
        RIR = RIR[:original_length]
    elif len(RIR) < original_length:
        RIR = np.concatenate((RIR, np.zeros(original_length - len(RIR))), dtype=np.float32)
    return RIR

def RIR_to_gpu(RIR):
    RIR_cp = cp.asarray(RIR)
    RIR_tr = torch.tensor(RIR, dtype=torch.float32).cuda()
    return RIR_cp,RIR_tr

def main():
    libraries = ['Torch','CuPy','SciPy','Librosa']
    fs = 16000
    nLoads = 500
    RIR = np.load('../recorded/lastRecording/RIR.npy')
    RIR = RIR[:,0]
    distance = 300 # in cm
    num_samples = int((distance / 34300) * fs)
    original_length = int(0.2*fs)
    RIRs = np.empty((0,original_length))
    RIR_format(RIR,num_samples,original_length)

    # Benchmarking variables
    timmings, execution_time = [], []
    reps = 10

    # Measuring time in GPU
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Loading several RIRs into one single array
    for _ in range(nLoads):
        RIRs = np.vstack((RIRs,RIR))
        if (_+1)%100==0:
            print("Loading " + str(_) + " RIRs")
    print("-------FINISHED LOADING RIRs-------")

    RIRs_cp,RIRs_tr = RIR_to_gpu(RIRs)

    # Medida de tiempos
    window = torch.hann_window(128)
    window = window.cuda()
    print("-------TORCH CALCULATION-------")
    for _ in range(reps):
        start.record()
        RIR_fft_tr = torch.stft(RIRs_tr,n_fft=256,win_length=128,window=window,hop_length=64,return_complex=True)
        end.record()
        torch.cuda.synchronize()
        timmings.append(start.elapsed_time(end))
    execution_time.append([np.mean(timmings[1:]),np.std(timmings[1:])])
    RIR_fft_tr = RIR_fft_tr.cpu()
    RIR_fft_tr = RIR_fft_tr.numpy()
    del RIRs_tr,window,RIR_fft_tr

    timmings = []
    print("-------CUPY CALCULATION-------")
    for _ in range(reps):
        start.record()
        f_cp,t_cp,RIR_fft_cp = stft_cp(RIRs_cp,fs=fs,nperseg=128,noverlap=64,nfft=256)
        end.record()
        torch.cuda.synchronize()
        timmings.append(start.elapsed_time(end))
    execution_time.append([np.mean(timmings[1:]),np.std(timmings[1:])])
    del RIRs_cp,RIR_fft_cp

    timmings = []
    print("-------SCIPY CALCULATION-------")
    for _ in range(reps):
        start_time = time.perf_counter()
        f_scipy,t_scipy,RIR_fft = stft_sp(RIRs,fs=16000,nperseg=128,noverlap=64,nfft=256)
        end_time = time.perf_counter()
        timmings.append(end_time - start_time)
    execution_time.append([np.mean(timmings[1:])*1e3,np.std(timmings[1:])*1e3])
    del RIR_fft

    timmings = []
    print("-------LIBROSA CALCULATION-------")
    for _ in range(reps):
        start_time = time.perf_counter()
        RIR_fft_lib = stft_lb(RIRs,n_fft=256,hop_length=64,win_length=128)
        end_time = time.perf_counter()
        timmings.append(end_time - start_time)
    execution_time.append([np.mean(timmings[1:])*1e3,np.std(timmings[1:])*1e3])
    hamm_win = scipy.signal.get_window('hann', 128)
    scale = np.sqrt(hamm_win.sum()**2)
    RIR_fft_lib = RIR_fft_lib / scale
    #RIR_fft_tr = RIR_fft_tr / scale
    del RIRs, RIR_fft_lib

    '''    plt.figure(figsize=(30,50))
    for i in range(RIR_fft.shape[0]):
        # Torch plot
        plt.subplot(RIR_fft.shape[0], 5, 5*i+1,title="Torch spectrogram")
        librosa.display.specshow(librosa.amplitude_to_db(np.abs(RIR_fft_tr[i,:,:]),ref=np.max), sr=16000, hop_length=64, x_axis="time", y_axis="linear",cmap='jet')

        # Scipy ploy
        plt.subplot(RIR_fft.shape[0], 5, 5*i+3,title="Scipy spectrogram")
        librosa.display.specshow(librosa.amplitude_to_db(np.abs(RIR_fft[i,:,:]),ref=np.max), sr=16000, hop_length=64, x_axis="time", y_axis="linear",cmap='jet')

        # Librosa plot
        plt.subplot(RIR_fft.shape[0], 5, 5*i+4,title="Librosa spectrogram")
        librosa.display.specshow(librosa.amplitude_to_db(np.abs(RIR_fft_lib[i,:,:]),ref=np.max), sr=16000, hop_length=64, x_axis="time", y_axis="linear",cmap='jet')

        # CuPy plot
        plt.subplot(RIR_fft.shape[0], 5, 5*i+5,title="CuPy spectrogram")
        librosa.display.specshow(librosa.amplitude_to_db(np.abs(cp.asnumpy(RIR_fft_cp[i,:,:])),ref=np.max), sr=16000, hop_length=64, x_axis="time", y_axis="linear", cmap='jet')'''

    print("GPU Torch => Mean computing time:",round(execution_time[0][0],2),"ms","- Std deviation:",round(execution_time[0][1],2),"ms")
    print("GPU CuPy => Mean computing time:",round(execution_time[1][0],2),"ms","- Std deviation:",round(execution_time[1][1],2),"ms")
    print("CPU SciPy => Mean computing time:",round(execution_time[2][0],2),"ms","- Std deviation:",round(execution_time[2][1],2),"ms")
    print("CPU Librosa => Mean computing time:",round(execution_time[3][0],2),"ms","- Std deviation:",round(execution_time[3][1],2),"ms")
    
    
    # Load the values to a csv file
    for i in range(len(libraries)):
        new_line_data = {'Library': [libraries[i]], 'Num_reps': [nLoads],'Mean': [round(execution_time[i][0],2)], 'Std_dev': [round(execution_time[i][1],2)]}
        df = pd.DataFrame(new_line_data)
        df.to_csv('CSV_files/Benchmark_multiple_RIRs.csv', mode='a', header=False, sep=';', decimal=',', index=False)
    
    
    
if __name__ == "__main__":
  main()
  #plt.show()