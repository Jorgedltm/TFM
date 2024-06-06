import math
import pandas as pd
from visualize import *
from datageneratorv2 import DataGenerator
from dataset import Dataset
from unet_vae import UNetVAE
import numpy as np
from numpy.fft import fft, ifft
from postprocess import PostProcess
from preprocess import Loader
import time
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras.utils import Progbar

def amplitude_loss(y_true, y_pred):
    return tf.keras.losses.mse(y_true, y_pred)


def phase_loss(y_true, y_pred):
    y_true_adj = y_true * 2 * math.pi - math.pi
    y_pred_adj = y_pred * 2 * math.pi - math.pi
    return tf.keras.backend.mean(1 - tf.math.cos(y_true_adj - y_pred_adj))


def sdr(pred, true):
    pred_adj = pred - np.mean(pred)
    true_adj = true - np.mean(true)
    origin_power = np.sum(true_adj ** 2) + 1e-8
    scale = np.sum(true_adj * pred_adj) / origin_power
    est_true = scale * true_adj
    est_res = pred_adj - est_true
    true_power = np.sum(est_true ** 2)
    res_power = np.sum(est_res ** 2)
    return 10 * np.log10(true_power) - 10 * np.log10(res_power)


def calculate_similarity(signal_a, signal_b, weights=None):
    """
    Compares two signals using various similarity measures and combines them into a single metric.

    Parameters:
    - signal_a: np.array, the first signal.
    - signal_b: np.array, the second signal.
    - weights: dict, optional; weights for each similarity measure.

    Returns:
    - metric: float, a composite metric evaluating the similarity between the two signals.
    """
    if weights is None:
        weights = {
            'time_static': 1.0,
            'time_shift': 1.0,
            'freq_static': 1.0,
            'freq_shift': 1.0,
            'energy': 1.0,
        }

    def time_domain_similarity_static():
        return np.sum(signal_a * signal_b)

    def time_domain_similarity_shift():
        fft_a = fft(signal_a)
        fft_b = fft(signal_b)
        product = fft_a * np.conj(fft_b)
        return np.sum(np.abs(ifft(product)))

    def frequency_domain_similarity_static():
        fft_a = fft(signal_a)
        fft_b = fft(signal_b)
        return np.sum(np.abs(fft_a * np.conj(fft_b)))

    def frequency_domain_similarity_shift():
        product = signal_a * signal_b
        fft_product = fft(product)
        return np.sum(np.abs(fft_product))

    def energy_similarity():
        power_a = np.sum(np.square(signal_a)) / len(signal_a)
        power_b = np.sum(np.square(signal_b)) / len(signal_b)
        return np.abs(power_a - power_b)

    # Compute similarities
    similarities = {
        'time_static': time_domain_similarity_static(),
        'time_shift': time_domain_similarity_shift(),
        'freq_static': frequency_domain_similarity_static(),
        'freq_shift': frequency_domain_similarity_shift(),
        'energy': energy_similarity(),
    }

    # Normalize and weight similarities
    max_vals = {key: max(1, np.abs(val)) for key, val in similarities.items()}  # Avoid division by zero
    weighted_sum = sum(weights[key] * (similarities[key] / max_vals[key]) for key in similarities)
    total_weight = sum(weights.values())

    # Compute final metric
    metric = weighted_sum / total_weight

    return metric

if __name__ == '__main__':
    
    models_folder = "models/"
    model_name = "unet-vae-64-mse-diff"
    latent_space_dim = 128
    loss = "mae"
    diff = True

    batch_size = 16
    debug = False

    rooms = None
    arrays = ["PlanarMicrophoneArray"]
    zones = None

    algorithms = ['ph', 'gl_ph', 'gl_mag']  # ['gl_ph', 'gl_mag', 'ph']
    n_iters = 64
    momentum = 0.99

    target_size = (144, 160, 2)
    mode = 3

    if diff:
        diff_str = "-diff"
        diff_gen = True
    else:
        diff_str = ""
        diff_gen = False

    modifier = f"-{latent_space_dim}-{loss}{diff_str}"

    if model_name in ["unet-vae", "unet-n"]:
        normalize_vector = True
    else:
        normalize_vector = False

    dataset_dir = '../../../datasets'
    models_folder = '../results/'
    saving_path = '../generated_rir/' + model_name + modifier

    print("Generating with UNET-VAE")
    trained_model = UNetVAE(input_shape=target_size,
                            inf_vector_shape=(2, 16),
                            mode=mode,
                            number_filters_0=32,
                            kernels=3,
                            latent_space_dim=latent_space_dim / 2,
                            name=model_name + modifier
                            )
    
    physical_devices = tf.config.list_physical_devices('GPU')
    print(f"\n Device(s) : {tf.config.experimental.get_device_details(physical_devices[0])['device_name']} \n")
    
    optimizer = tf.keras.optimizers.legacy.Adam()
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=trained_model.model)
    manager = tf.train.CheckpointManager(checkpoint, directory=models_folder + model_name, max_to_keep=1)
    checkpoint.restore(manager.latest_checkpoint)

    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")
        
    trained_model.summary()
        
    loader = Loader(sample_rate=48000, mono=True, duration=0.2) # Initializing class for loading librosa real ouput RIR
    
    dataset = Dataset(dataset_dir, 'room_impulse', normalization=True, debugging=debug, extract=False,
                      room_characteristics=True, room=rooms, array=arrays, zone=zones,
                      normalize_vector=normalize_vector)
    test_generator = DataGenerator(dataset, batch_size=batch_size, partition='test', shuffle=False,
                                   characteristics=True)
    
    if normalize_vector:
        min_max_vector = dataset.return_min_max_vector()
    else:
        min_max_vector = None
    
    for algorithm in algorithms:

        postprocessor = PostProcess(folder=saving_path + "/" + model_name + modifier, algorithm=algorithm,
                                    momentum=momentum, n_iters=n_iters, normalize_vector=normalize_vector)
        
        time.sleep(1)
        print(f'Generating wavs and obtaining loss | {algorithm}')
        numUpdates = test_generator.__len__()
        time_inference, time_postprocessing, time_loss = [], [], []
        (total_loss, amp_loss, pha_loss, wav_loss, wav_loss_50ms,
         missa_amp_loss, missa_wav_loss, sdr_metric, similarity_metric) = [], [], [], [], [], [], [], [], []

        hemi_total_loss, large_total_loss, medium_total_loss, shoe_total_loss, small_total_loss = [], [], [], [], []
        hemi_amp_loss, large_amp_loss, medium_amp_loss, shoe_amp_loss, small_amp_loss = [], [], [], [], []
        hemi_pha_loss, large_pha_loss, medium_pha_loss, shoe_pha_loss, small_pha_loss = [], [], [], [], []
        hemi_wav_loss, large_wav_loss, medium_wav_loss, shoe_wav_loss, small_wav_loss = [], [], [], [], []
        hemi_wav_loss_50ms, large_wav_loss_50ms, medium_wav_loss_50ms, shoe_wav_loss_50ms, small_wav_loss_50ms = [], [], [], [], []
        hemi_missa_amp_loss, large_missa_amp_loss, medium_missa_amp_loss, shoe_missa_amp_loss, small_missa_amp_loss = [], [], [], [], []
        hemi_missa_wav_loss, large_missa_wav_loss, medium_missa_wav_loss, shoe_missa_wav_loss, small_missa_wav_loss = [], [], [], [], []
        hemi_sdr_metric, large_sdr_metric, medium_sdr_metric, shoe_sdr_metric, small_sdr_metric = [], [], [], [], []
        hemi_similarity_metric, large_similarity_metric, medium_similarity_metric, shoe_similarity_metric, small_similarity_metric = [], [], [], [], []

        hemi_count, large_count, medium_count, shoe_count, small_count = 0, 0, 0, 0, 0

        plot_countdown = 0
        plot_count = 0

        progBar = Progbar(numUpdates)

        start = time.time()
    