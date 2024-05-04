import time
import cupy as cp
from cupyx.scipy.signal import stft as stft_cp
import jax.numpy as jnp
from jax.scipy.signal import stft as stft_jax
from scipy.signal import stft as stft_sp
from librosa import stft as stft_lb
from jax import device_put
import matplotlib.pyplot as plt
import numpy as np
import librosa
import sys
import _parseargs as parse
import scipy
import torch
import tensorflow as tf

args = parse._parse()

if args.stftsel==True:
   RIR_jax = jnp.load('../recorded/newrir_mics' + str(args.mics) + '_spk' + str(args.spk) + '/RIR.npy')
   RIR = np.load('../recorded/newrir_mics' + str(args.mics) + '_spk' + str(args.spk) + '/RIR.npy')
else:
   RIR = np.load('../recorded/lastRecording/RIR.npy')
   RIR_jax = jnp.load('../recorded/lastRecording/RIR.npy')

RIR = RIR[:,0]
RIR_jax = RIR_jax[:,0]
RIR_cp = cp.asarray(RIR)
RIR_tr = torch.tensor(RIR, dtype=torch.float32).cuda()

# Benchmarking variables
timmings = []
reps = 50

# Events for GPU time measuring
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

# Computation time with Torch library
print("-------TORCH CALCULATION-------")
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

# Computation time with Tensorflow library
print("-------TENSORFLOW CALCULATION-------")
with tf.device('/device:GPU:0'):
  timmings = []
  for _ in range(reps):
    start.record()
    RIR_fft_tf = tf.signal.stft(RIR,fft_length=256,frame_length=128,window_fn=tf.signal.hann_window,frame_step=64)
    end.record()
    torch.cuda.synchronize()
    timmings.append(start.elapsed_time(end))
  execution_time_tensorflow = np.mean(timmings[1:])
RIR_fft_tf = RIR_fft_tf.numpy()

# Computation time with JAX library
print("-------JAX CALCULATION-------")
timmings = []
for _ in range(reps):
   start_time = time.perf_counter()
   f_jax,t_jax,RIR_fft_jax = stft_jax(RIR,fs=16000,nperseg=128,noverlap=64,nfft=256)
   end_time = time.perf_counter()
   timmings.append(end_time - start_time)
execution_time_jax = np.mean(timmings)

# Computation time with CuPy library
print("-------CUPY CALCULATION-------")
timmings = []
for _ in range(reps):
   start.record()
   f_cp,t_cp,RIR_fft_cp = stft_cp(RIR_cp,fs=16000,nperseg=128,noverlap=64,nfft=256)
   end.record()
   torch.cuda.synchronize()
   timmings.append(start.elapsed_time(end))
execution_time_cp = np.mean(timmings[1:])

# Computation time with SciPy library
print("-------SCIPY CALCULATION-------")
timmings = []
for _ in range(reps):
   start_time = time.perf_counter()
   f_scipy,t_scipy,RIR_fft = stft_sp(RIR,fs=16000,nperseg=128,noverlap=64,nfft=256)
   end_time = time.perf_counter()
   timmings.append(end_time - start_time)
execution_time_scipy = np.mean(timmings)

# Computation time with Librosa library
print("-------LIBROSA CALCULATION-------")
timmings = []
for _ in range(reps):
   start_time = time.perf_counter()
   RIR_fft_lib = stft_lb(RIR,n_fft=256,hop_length=64,win_length=128)
   end_time = time.perf_counter()
   timmings.append(end_time - start_time)
execution_time_librosa = np.mean(timmings)

# Librosa and Torch don´t normalize the fft output, so we are going to do it so results between libraries match.
hamm_win = scipy.signal.get_window('hann', 128)
scale = np.sqrt(hamm_win.sum()**2)
RIR_fft_lib = RIR_fft_lib / scale
RIR_fft_tr = RIR_fft_tr / scale
RIR_fft_tf = RIR_fft_tf / scale

# Plotting RIR
plt.figure(figsize = (9, 3))
maxval = np.max(RIR)
minval = np.min(RIR)
time_RIR = np.arange(len(RIR))/16000
plt.plot(time_RIR,RIR)
plt.ylim((minval+0.05*minval,maxval+0.05*maxval))
plt.xlim((0,np.size(RIR)/16000))
plt.savefig("Figures/RIR.png")


plt.figure(figsize = (28, 6.4))
# Torch plot
plt.subplot(1, 6, 1,title="Torch spectrogram")
librosa.display.specshow(librosa.amplitude_to_db(np.abs(RIR_fft_tr)), sr=16000, hop_length=64, x_axis="time", y_axis="linear", cmap='jet')
# Jax plot
plt.subplot(1, 6, 2,title="JAX spectrogram")
librosa.display.specshow(librosa.amplitude_to_db(np.abs(RIR_fft_jax)), sr=16000, hop_length=64, x_axis="time", y_axis="linear", cmap='jet')
# Scipy ploy
plt.subplot(1, 6, 3,title="SciPy spectrogram")
librosa.display.specshow(librosa.amplitude_to_db(np.abs(RIR_fft)), sr=16000, hop_length=64, x_axis="time", y_axis="linear", cmap='jet')
# Librosa plot
plt.subplot(1, 6, 4,title="Librosa spectrogram")
librosa.display.specshow(librosa.amplitude_to_db(np.abs(RIR_fft_lib)), sr=16000, hop_length=64, x_axis="time", y_axis="linear", cmap='jet')
# CuPy plot
plt.subplot(1, 6, 5,title="CuPy spectrogram")
librosa.display.specshow(librosa.amplitude_to_db(np.abs(cp.asnumpy(RIR_fft_cp))), sr=16000, hop_length=64, x_axis="time", y_axis="linear", cmap='jet')
# CuPy plot
plt.subplot(1, 6, 6,title="Tensorflow spectrogram")
librosa.display.specshow(librosa.amplitude_to_db(np.abs(np.transpose(RIR_fft_tf))), sr=16000, hop_length=64, x_axis="time", y_axis="linear", cmap='jet')
plt.savefig("Figures/STFT_RIR.png")

print("GPU Torch computing time:",execution_time_torch,"ms")
print("CPU JAX computing time:",execution_time_jax*1e3,"ms")
print("CPU SciPy computing time:",execution_time_scipy*1e3,"ms")
print("CPU Librosa computing time:",execution_time_librosa*1e3,"ms")
print("GPU CuPy computing time:",execution_time_cp,"ms")
print("GPU Tensorflow computing time:",execution_time_tensorflow,"ms")

#plt.show()

