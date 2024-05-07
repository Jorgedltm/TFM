import time
import cupy as cp
from cupyx.scipy.signal import stft as stft_cp
from scipy.signal import stft as stft_sp
from librosa import stft as stft_lb
import matplotlib.pyplot as plt
import numpy as np
import librosa
import torch
from scipy.io import wavfile
import scipy

def RIR_format(RIR,num_samples,original_length):
  max_index = np.argmax(RIR)
  min_index = np.argmin(RIR)
  if max_index < min_index:
    direct_sound_index = max_index
  else:
    direct_sound_index = min_index

  RIR = np.concatenate((np.zeros(num_samples), RIR[direct_sound_index:,0]), dtype=np.float32)
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
  fs = 16000
  RIR = np.load('recorded/lastRecording/RIR.npy')
  RIR = RIR[:].astype(np.float32)

  # Benchmarking variables
  timmings = []
  reps = 10

  distance = 300 # distance from mic to speaker in cm
  num_samples = int((distance / 34300) * fs)
  original_length = int(0.2*fs)
  RIR = RIR_format(RIR,num_samples,original_length)
  RIR_cp,RIR_tr = RIR_to_gpu(RIR)
  
  start = torch.cuda.Event(enable_timing=True)
  end = torch.cuda.Event(enable_timing=True)
  window = torch.hann_window(128)
  window = window.cuda()
  for _ in range(reps):
    start.record()
    RIR_fft_tr = torch.stft(RIR_tr,n_fft=256,win_length=128,window=window,hop_length=64,return_complex=True)
    end.record()
    torch.cuda.synchronize()
    timmings.append(start.elapsed_time(end))
  execution_time_torch = np.mean(timmings[1:])
  RIR_fft_tr = RIR_fft_tr.cpu()
  RIR_fft_tr = RIR_fft_tr.numpy()

  timmings = []
  for _ in range(reps):
    start_time = time.perf_counter()
    f_scipy,t_scipy,RIR_fft = stft_sp(RIR,fs=fs,nperseg=128,noverlap=64,nfft=256)
    end_time = time.perf_counter()
    timmings.append(end_time - start_time)
  execution_time_scipy = np.mean(timmings)

  timmings = []
  for _ in range(reps):
    start.record()
    f_cp,t_cp,RIR_fft_cp = stft_cp(RIR_cp,fs=fs,nperseg=128,noverlap=64,nfft=256)
    end.record()
    torch.cuda.synchronize()
    timmings.append(start.elapsed_time(end))
  execution_time_cp = np.mean(timmings[1:])

  timmings = []
  for _ in range(reps):
    start_time = time.perf_counter()
    RIR_fft_lib = stft_lb(RIR,n_fft=256,hop_length=64,win_length=128)
    end_time = time.perf_counter()
    timmings.append(end_time - start_time)
  execution_time_librosa = np.mean(timmings)
  # Librosa doesn´t normalize the fft output, so we are going to do it so results between libraries match.
  hamm_win = scipy.signal.get_window('hann', 128)
  scale = np.sqrt(hamm_win.sum()**2)
  RIR_fft_lib = RIR_fft_lib / scale
  RIR_fft_tr = RIR_fft_tr / scale

  plt.figure(figsize = (60, 5))
  maxval = np.max(RIR)
  minval = np.min(RIR)
  time_RIR = np.arange(np.size(RIR))/fs
  plt.plot(time_RIR,RIR)
  plt.ylim((minval+0.05*minval,maxval+0.05*maxval))
  plt.xlim((0,np.size(RIR)/fs))
  plt.figure(figsize = (20, 10))

  # Torch plot
  plt.subplot(1, 4, 1,title="PyTorch spectrogram")
  librosa.display.specshow(librosa.amplitude_to_db(np.abs(RIR_fft_tr)), sr=fs, hop_length=64, x_axis="time", y_axis="linear",cmap='jet')
  # Scipy plot
  plt.subplot(1, 4, 2,title="SciPy spectrogram")
  librosa.display.specshow(librosa.amplitude_to_db(np.abs(RIR_fft)), sr=fs, hop_length=64, x_axis="time", y_axis="linear",cmap='jet')
  # Librosa plot
  plt.subplot(1, 4, 3,title="Librosa spectrogram")
  librosa.display.specshow(librosa.amplitude_to_db(np.abs(RIR_fft_lib)), sr=fs, hop_length=64, x_axis="time", y_axis="linear",cmap='jet')
  # CuPy plot
  plt.subplot(1, 4, 4,title="CuPy spectrogram")
  librosa.display.specshow(librosa.amplitude_to_db(np.abs(cp.asnumpy(RIR_fft_cp))), sr=fs, hop_length=64, x_axis="time", y_axis="linear", cmap='jet')

  # Execution times
  print("GPU PyTorch computing:",execution_time_torch,"ms")
  print("CPU SciPy computing:",execution_time_scipy*1e3,"ms")
  print("CPU Librosa computing:",execution_time_librosa*1e3,"ms")
  print("GPU CuPy computing:",execution_time_cp,"ms")

  # Display results
  #plt.show()
  
if __name__ == "__main__":
  main()