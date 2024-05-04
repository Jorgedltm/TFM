import time
import os
import cupy as cp
import scipy
import gc
import tracemalloc
from cupyx.scipy.signal import stft as stft_cp
from scipy.signal import stft as stft_sp
from librosa import stft as stft_lb
from memory_profiler import profile
import sys
sys.path.append('..')
import _parseargs as parse
import matplotlib.pyplot as plt
import numpy as np
import librosa
import torch
import tensorflow as tf
import pandas as pd


def RIR_to_gpu(RIR,library):

    if library == "Torch":
       RIR_gpu = torch.tensor(RIR, dtype=torch.float32).cuda()
    elif library == "CuPy":
       RIR_gpu = cp.asarray(RIR,dtype=np.float32)
    elif library == "Tensorflow":
       with tf.device('/device:GPU:0'):
          RIR_gpu = tf.convert_to_tensor(RIR, dtype=tf.float32)

    return RIR_gpu

def main(nLoads,library):
    
    fs = 16000
    print(f"-------ITERATION WITH {nLoads} RIRs-------")
    #nLoads = 500
    args = parse._parse()
    if args.stftsel==True:
       RIR = np.load('../recorded/newrir_mics' + str(args.mics) + '_spk' + str(args.spk) + '/RIR.npy')
    else:
       RIR = np.load('../recorded/lastRecording/RIR.npy')

    RIR = RIR[:,0]
    RIRs = np.empty((0,len(RIR)))

    # Benchmarking variables
    timmings, execution_time = [], []
    reps = 5

    # Measuring time in GPU
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Loading several RIRs into one single array
    if os.path.exists(f"RIR_stack/RIRs_{nLoads}.npy"):
       RIRs = np.load(f"RIR_stack/RIRs_{nLoads}.npy")
       print("-------LOADED RIRs FROM PREVIOUS STACK-------")
    else:
       for _ in range(nLoads):
           RIRs = np.vstack((RIRs,RIR))
           if (_+1)%100==0:
               print("Loading " + str(_+1) + " RIRs")
       np.save(f"RIR_stack/RIRs_{nLoads}.npy",RIRs)
       print("-------FINISHED LOADING RIRs-------")

    
    # Time measurement
    if library == 'Torch':
       print("-------TORCH CALCULATION-------")
       window = torch.hann_window(128)
       window = window.cuda()
       RIRs_tr = RIR_to_gpu(RIRs,"Torch")
       del RIRs
       for _ in range(reps):
           try:
              start.record()
              RIR_fft_tr = torch.stft(RIRs_tr,n_fft=256,win_length=128,window=window,hop_length=64,return_complex=True)
              end.record()
              torch.cuda.synchronize()
              timmings.append(start.elapsed_time(end))
              del RIR_fft_tr
           except MemoryError:
              torch.cuda.empty_cache()
       execution_time.append([np.mean(timmings[2:]),np.std(timmings[2:])])
       torch.cuda.empty_cache()
       #RIR_fft_tr = RIR_fft_tr.cpu()
       #RIR_fft_tr = RIR_fft_tr.numpy()
       #hamm_win = scipy.signal.get_window('hann', 128)
       #scale = np.sqrt(hamm_win.sum()**2)
       #RIR_fft_tr = RIR_fft_tr / scale
       del RIRs_tr,window
      
    if library == 'Tensorflow':
       with tf.device('/device:GPU:0'):
          print("-------TENSORFLOW COMPUTING-------")
          timmings = []
          RIRs_tf = RIR_to_gpu(RIRs,"Tensorflow")
          del RIRs
          for _ in range(reps):
             try:
                start.record()
                RIR_fft_tf = tf.signal.stft(RIRs_tf,fft_length=256,frame_length=128,window_fn=tf.signal.hann_window,frame_step=64)
                end.record()
                torch.cuda.synchronize()
                timmings.append(start.elapsed_time(end))
                del RIR_fft_tf
             except MemoryError:
                gc.collect()
                torch.cuda.empty_cache()
          execution_time.append([np.mean(timmings[2:]),np.std(timmings[2:])])
          torch.cuda.empty_cache()
       #RIR_fft_tf = RIR_fft_tf.numpy()
       #RIR_fft_tf = RIR_fft_tf / scale
       del RIRs_tf

    if library == 'CuPy':
       print("-------CUPY CALCULATION-------")
       timmings = []
       RIRs_cp = RIR_to_gpu(RIRs,"CuPy")
       print("Size of array in memory:", RIRs_cp.nbytes*1e-9, "Gbytes"," - ", RIRs.nbytes*1e-9, "Gbytes")
       del RIRs
       for _ in range(reps):
           try:
              start.record()
              f_cp,t_cp,RIR_fft_cp = stft_cp(RIRs_cp,fs=fs,nperseg=128,noverlap=64,nfft=256)
              end.record()
              torch.cuda.synchronize()
              timmings.append(start.elapsed_time(end))
              del f_cp,t_cp,RIR_fft_cp
              cp.get_default_memory_pool().free_all_blocks()
              cp.get_default_pinned_memory_pool().free_all_blocks()
           except MemoryError:
              cp.get_default_memory_pool().free_all_blocks()
              cp.get_default_pinned_memory_pool().free_all_blocks()
       execution_time.append([np.mean(timmings[2:]),np.std(timmings[2:])])
       del RIRs_cp
    
    if library == 'SciPy':
       print("-------SCIPY CALCULATION-------")
       timmings = []
       for _ in range(reps):
           try:
              start_time = time.perf_counter()
              f_scipy,t_scipy,RIR_fft = stft_sp(RIRs,fs=16000,nperseg=128,noverlap=64,nfft=256)
              end_time = time.perf_counter()
              timmings.append(end_time - start_time)
              del RIR_fft,f_scipy,t_scipy
           except MemoryError:
              gc.collect()
       execution_time.append([np.mean(timmings[2:])*1e3,np.std(timmings[2:])*1e3])
       del RIRs

    if library == 'Librosa':
       print("-------LIBROSA CALCULATION-------")
       timmings = []
       for _ in range(reps):
           try:
              start_time = time.perf_counter()
              RIR_fft_lib = stft_lb(RIRs,n_fft=256,hop_length=64,win_length=128)
              end_time = time.perf_counter()
              timmings.append(end_time - start_time)
              del RIR_fft_lib
           except MemoryError:
              gc.collect()
       execution_time.append([np.mean(timmings[2:])*1e3,np.std(timmings[2:])*1e3])
       #RIR_fft_lib = RIR_fft_lib / scale
       del RIRs
    
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
    '''
    print("GPU Torch => Mean computing time:",round(execution_time[0][0],2),"ms","- Std deviation:",round(execution_time[0][1],2),"ms")
    print("GPU Tensorflow => Mean computing time:",round(execution_time[1][0],2),"ms","- Std deviation:",round(execution_time[1][1],2),"ms")
    print("GPU CuPy => Mean computing time:",round(execution_time[2][0],2),"ms","- Std deviation:",round(execution_time[2][1],2),"ms")
    print("CPU SciPy => Mean computing time:",round(execution_time[3][0],2),"ms","- Std deviation:",round(execution_time[3][1],2),"ms")
    print("CPU Librosa => Mean computing time:",round(execution_time[4][0],2),"ms","- Std deviation:",round(execution_time[4][1],2),"ms")
    '''
    
    # Load the values to a csv file
    for i in range(len(execution_time)):
        new_line_data = {'Library': [library], 'Num_reps': [nLoads],'Mean': [round(execution_time[i][0],2)], 'Std_dev': [round(execution_time[i][1],2)]}
        df = pd.DataFrame(new_line_data)
        df.to_csv('CSV_files/Benchmark_multiple_RIRs_jn.csv', mode='a', header=False, sep=';', decimal=',', index=False)
    
    if 'RIRs' in locals():
       del RIRs
    

if __name__ == "__main__":
    
    loads = [500,1000,2500,5000,7500,10000,12500,15000,25000,30000]
    for load in loads:
       main(load,'Torch')
