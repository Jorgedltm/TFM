import math
import soundfile as sf
from visualize import *
from scipy.signal import resample_poly
import tensorflow as tf
import librosa
import torch

class FeatureExtractor:
    def __init__(self, n_fft, win_length, hop_length):
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length

    def extract(self, waveform):
        # Convert the waveform to a spectrogram via a STFT.
        waveform = torch.tensor(waveform, dtype=torch.float32).cuda()
        window = torch.hann_window(self.win_length)
        window = window.cuda()
        
        spectrogram = torch.stft(waveform,n_fft=self.n_fft,win_length=self.win_length,window=window,hop_length=self.hop_length,return_complex=True)
        spectrogram = spectrogram.cpu()
        #spectrogram = tf.signal.stft(waveform, fft_length=self.n_fft, frame_length=self.win_length, window_fn=tf.signal.hann_window,frame_step=self.hop_length)
        spectrogram = spectrogram.numpy()
        #spectrogram = spectrogram.T
        
        #spectrogram = librosa.stft(waveform, n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_length)
        amp = np.abs(spectrogram)
        phase = np.angle(spectrogram)
        return amp, phase


class Normalizer:
    def __init__(self):
        self.md = 100
        self.ep = 10 ** (-1 * self.md / 20)

    def normalize(self, amp, phase):
        amp_norm = 20 * np.log10(amp / (128) + self.ep)
        amp_norm = (amp_norm + self.md) / self.md

        phase_norm = (phase + math.pi) / (2 * math.pi)

        return amp_norm, phase_norm

    def denormalize(self, amp_norm, phase_norm):
        amp = (amp_norm * self.md) - self.md
        amp = (10 ** (amp / 20) - self.ep) * (128)

        phase = (phase_norm * 2 * math.pi) - math.pi
        phase = (phase + math.pi) % (2 * math.pi) - math.pi

        return amp, phase

    def denormalize_embedding(self, emb, min_max_vector):
        emb[..., 0:4] = (emb[..., 0:4] * (min_max_vector[1] - min_max_vector[0])) + min_max_vector[0] # Room dimensions
        emb[..., 4:8] = (emb[..., 4:8] * (min_max_vector[3] - min_max_vector[2])) + min_max_vector[2] # Room angles
        emb[..., 8:14] = (emb[..., 8:14] * (min_max_vector[5] - min_max_vector[4])) + min_max_vector[4] # Mic and spk positions
        emb[..., 14:15] = (emb[..., 14:15] * (min_max_vector[7] - min_max_vector[6])) + min_max_vector[6] # Height
        emb[..., 15:16] = (emb[..., 15:16] * (min_max_vector[9] - min_max_vector[8])) + min_max_vector[8] # RT60
        return emb

class Loader:

    def __init__(self, sample_rate, duration, mono):
        self.sample_rate = sample_rate
        self.duration = duration
        self.mono = mono

    def load(self, file_path):
        signal = np.load(file_path)[:,0] # Loading RIR
        signal = resample_poly(signal,int(3),1) # Upsampling to match 48 kHz
        signal -= np.mean(signal)
        return signal
    
class TensorPadder:

    def __init__(self, desired_shape):
        self.current_shape = None
        self.desired_shape = desired_shape
        self.c_rows = None
        self.c_columns = None
        self.n_rows = None
        self.n_columns = None

    def pad_amp_phase(self, amp, phase):
        padded_amp = self.transform(amp)
        padded_phase = self.transform(phase)
        return padded_amp, padded_phase

    def transform(self, tensor):
        if self.get_needed_transform(tensor):
            rp_tensor = self.row_transform(tensor)
            padded_tensor = self.col_transform(rp_tensor)
            return padded_tensor
        else:
            return tensor

    def get_needed_transform(self, tensor):
        self.current_shape = tensor.shape
        self.c_rows = self.current_shape[0]
        self.c_columns = self.current_shape[1]

        conditions = self.current_shape[0] > self.desired_shape[0] or self.current_shape[1] > self.desired_shape[1]
        if not conditions:
            self.n_rows = self.desired_shape[0] - self.current_shape[0]
            self.n_columns = self.desired_shape[1] - self.current_shape[1]
            return True
        else:
            return False

    def row_transform(self, tensor):
        rt_tensor = np.r_[tensor, np.zeros((self.n_rows, self.c_columns))]
        return rt_tensor

    def col_transform(self, tensor):
        temp_tensor = tensor
        for i in range(self.n_columns):
            temp_tensor = np.c_[temp_tensor, np.zeros(self.desired_shape[0])]
        ct_tensor = temp_tensor
        return ct_tensor

    @staticmethod
    def un_pad(amp, phase, desired_shape):
        amp_d = np.delete(amp, slice(desired_shape[0], amp.shape[0]), 0)
        amp_d = np.delete(amp_d, slice(desired_shape[1], amp.shape[1]), 1)
        phase_d = np.delete(phase, slice(desired_shape[0], phase.shape[0]), 0)
        phase_d = np.delete(phase_d, slice(desired_shape[1], phase.shape[1]), 1)
        return amp_d, phase_d

def sigmoid(beta, dimensions):
    x = np.linspace(-10, 10, dimensions[1])
    z = 1 / (1 + np.exp(-(x+5) * beta))
    z = np.flip(z)
    sig = np.tile(z, (dimensions[0], 1))
    return sig


if __name__ == '__main__':
    N_FFT = 256
    WINDOW_LENGTH = 128
    HOP_LENGTH = 64
    DURATION = 0.2  # in seconds
    SAMPLE_RATE = 48000 # Real 16 kHz but upsampling applied
    MONO = True
    DESIRED_SHAPE = (144, 160)

    loader = Loader(SAMPLE_RATE, DURATION, MONO)
    normalizer = Normalizer()
    extractor = FeatureExtractor(n_fft=N_FFT, win_length=WINDOW_LENGTH, hop_length=HOP_LENGTH)
    padder = TensorPadder(DESIRED_SHAPE)

    features = []
    room = 'ShoeBox'
    mic_pos=[208, 70, 78]
    spk_pos=[64, 318, 90]

    wav = loader.load(f'/home/jetson1/Documents/jorge/pyrirtool/recorded_RIR/newrir_mic{mic_pos}_spk{spk_pos}/RIR.npy')
    plot_wav(wav)

    sig = sigmoid(0.5, (144, 160))
    sig = np.repeat(np.expand_dims(sig, 0), 16, axis=0)

    amp, phase = extractor.extract(wav)

    print(amp.shape, phase.shape)

    amp_norm, phase_norm = normalizer.normalize(amp, phase)

    print(amp.max(), amp.min(), phase.max(), phase.min())
    print(amp_norm.max(), amp_norm.min(), phase_norm.max(), phase_norm.min())

    print(amp_norm.shape, phase_norm.shape)
    amp_pad, phase_pad = padder.pad_amp_phase(amp_norm, phase_norm)
    print(amp_pad.shape, phase_pad.shape)

    plot_spec(amp)
    plot_spec(amp_norm)
    plot_spec(amp_pad)

    # sigmoid_spec = amp_pad * sig[:,:,0]
    # plot_spec(sigmoid_spec)

    plot_spec(phase)
    plot_spec(phase_norm)
    plot_spec(phase_pad)

    # sigmoid_phase = phase_pad * sig[:,:,0]
    # plot_spec(sigmoid_phase)

    amp_unpad, phase_unpad = padder.un_pad(amp_pad, phase_pad, (129, 151))
    print(amp_unpad.shape, phase_unpad.shape)

    spectrogram = tf.stack([amp_pad, phase_pad], axis=-1)

    print(spectrogram.shape)

    sigmoid_phase = phase_pad * sig[1,:,:]
    sigmoid_amp = amp_pad * sig[1,:,:]

    print(phase_pad.shape)

    plot_spec(sigmoid_phase)
    plot_spec(sigmoid_amp)

    amp_denorm, phase_denorm = normalizer.denormalize(amp_unpad, phase_unpad)

    print(np.max(amp), np.max(amp_denorm))

    conv_stft = amp_denorm * (np.cos(phase_denorm) + 1j * np.sin(phase_denorm))
    conv_stft = conv_stft.T # To match tf istft implementation
    
    waveform = tf.signal.inverse_stft(conv_stft, fft_length=N_FFT, frame_length=WINDOW_LENGTH, window_fn=tf.signal.hann_window,frame_step=HOP_LENGTH)
    print(wav.shape, wav.dtype)
    print(waveform.shape, waveform.dtype)
    plot_wav(waveform)

    num = tf.norm((waveform[:len(wav)] - wav), ord=2)
    den = tf.norm(wav, ord=2)
    loss_missa_wav = 20 * math.log10(num / den)

    print(loss_missa_wav)

    plot_wav(waveform)