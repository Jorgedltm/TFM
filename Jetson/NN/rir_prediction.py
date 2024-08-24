import math
import re
import pandas as pd
import gc
from visualize import *
from datagenv2 import DataGenerator
from genembedding import EmbedGen
from unet_vae import UNetVAE
import numpy as np
from numpy.fft import fft, ifft
from postprocessv2 import PostProcess
from preprocessv2 import Loader
import time
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras.utils import Progbar
from tensorflow.keras.models import load_model
from time import time as tm
from time import ctime

def amplitude_loss(y_true, y_pred):
    return tf.keras.losses.MSE(y_true, y_pred)


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

    print(ctime(tm()))

    model_name = "unet-vae"
    latent_space_dim = 64
    loss = "mse"
    diff = True

    batch_size = 1
    debug = False

    rooms = None
    arrays = ["PlanarMicrophoneArray"]
    zones = None

    algorithms = ['gl_ph', 'gl_mag', 'ph']  # ['gl_ph', 'gl_mag', 'ph']
    n_iters = 64
    momentum = 0.99

    target_size = (144, 160, 2)
    mode = 3
    
    time_dataframes = []
    loss_dataframes = []

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

    dataset_dir = '../../pyrirtool/recorded_RIR'
    models_folder = 'models/unet-vae-64-mse-diff/'
    saving_path = 'generated_rir/' + model_name + modifier
    
    plot = False
    models = os.listdir(models_folder)
    #models = ["saved_model"]
        
    for model in models:
    
        model_load = os.path.join(models_folder + model)
        
        print(f"Generating with UNET-VAE, and initializing from {model}")
        
        physical_devices = tf.config.list_physical_devices('GPU')
            
        if physical_devices:
            
            print(f"\n Device(s) : {tf.config.experimental.get_device_details(physical_devices[0])['device_name']} \n")
            tf.config.set_logical_device_configuration(physical_devices[0],[tf.config.LogicalDeviceConfiguration(memory_limit=1024)])
            
        else:
            
            print("No GPU devices found.")
        
        loader = Loader(sample_rate=48000, mono=True, duration=0.2) # Initializing class for loading librosa real ouput RIR
    
        dataset = EmbedGen(dataset_dir, 'room_impulse', debugging=False, normalization=True, normalize_vector=True, room_characteristics=True)
    
        test_generator = DataGenerator(dataset, batch_size=batch_size, shuffle=False, normalize_vector=True, characteristics=True)
        
        if normalize_vector:
            min_max_vector = dataset.return_min_max_vector()
        else:
            min_max_vector = None
        
        if 'saved_model' in model:
            
            patron = r'saved_model_(\w+)'
            trt = re.search(patron,model)
            
            if trt:
                infering_mode = trt.group(1)
            else:
                infering_mode = 'TF_FP32'
            
            print(infering_mode)
            trained_model = load_model(model_load)
            infer = trained_model.signatures["serving_default"]
            
        else:
            
            infering_mode = 'Checkpoint'
            
            trained_model = UNetVAE(input_shape=target_size,
                                    inf_vector_shape=(2, 16),
                                    mode=mode,
                                    number_filters_0=32,
                                    kernels=3,
                                    latent_space_dim=latent_space_dim / 2,
                                    name=model_name + modifier
                                    )
            
            optimizer = tf.keras.optimizers.legacy.Adam()
            checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=trained_model.model)
            manager = tf.train.CheckpointManager(checkpoint, directory=model_load, max_to_keep=1)
            checkpoint.restore(manager.latest_checkpoint)

            if manager.latest_checkpoint:
                print("Restored from {}".format(manager.latest_checkpoint))
            else:
                print("Initializing from scratch.")
                
        # Mostrar el resumen del modelo cargado
        #trained_model.summary()
        
        for algorithm in algorithms:
    
            postprocessor = PostProcess(folder=saving_path + "/" + model_name + modifier, algorithm=algorithm,
                                        momentum=momentum, n_iters=n_iters, normalize_vector=normalize_vector)
            
            time.sleep(10)
            print(f'Generating wavs and obtaining loss | {algorithm}')
            numUpdates = test_generator.__len__()
            time_inference, time_postprocessing, time_loss = [], [], []
            (total_loss, amp_loss, pha_loss, wav_loss, wav_loss_50ms,
            missa_amp_loss, missa_wav_loss, sdr_metric, similarity_metric) = [], [], [], [], [], [], [], [], []
    
            small_total_loss = []
            small_amp_loss = []
            small_pha_loss = []
            small_wav_loss = []
            small_wav_loss_50ms = []
            small_missa_amp_loss = []
            small_missa_wav_loss = []
            small_sdr_metric = []
            small_similarity_metric = []
    
            hemi_count, large_count, medium_count, shoe_count, small_count = 0, 0, 0, 0, 0
    
            plot_countdown = 0
            plot_count = 0
    
            progBar = Progbar(numUpdates)
    
            start = time.time()
            
            for i in range(0, numUpdates):
    
                spec_in, emb, spec_out, characteristic = test_generator.__getitem__(i)
                #print("Time spent: ",time.time()-start)
                start_inf = time.time()
    
                if model_name == "unet-vae" or model_name == 'unet-vae-emb': 
                    
                    if 'saved_model' in model:
                    
                        inputs = {
                        'input_1': spec_in,
                        'input_2': emb
                        }
                        spec_generated = infer(**inputs)
                        spec_generated = spec_generated['linear_layer']
                    
                    else:

                        spec_generated, _, _ = trained_model.model([spec_in, emb], training=False)
                
                else:

                    spec_generated = trained_model.model([spec_in, emb], training=False)
                    
                end_inf = time.time()
    
                time_inference.append(end_inf - start_inf)
    
                for j in range(0, emb.shape[0]):
                    
                    start_gen = time.time()
    
                    if diff_gen:
                        diff_phase_generated = (spec_generated[j, :, :, 1] + spec_in[j, :, :, 1]).numpy()
                        diff_spec_generated = np.stack((spec_generated[j, :, :, 0], diff_phase_generated), axis=-1)
                        wav_pred = postprocessor.post_process(diff_spec_generated, emb[j, 1, :], min_max_vector)
                    else:
                        wav_pred = postprocessor.post_process(spec_generated[j], emb[j, 1, :], min_max_vector)
    
                    end_gen = time.time()
                    time_postprocessing.append(end_gen - start_gen)
    
                    start_loss = time.time()
    
                    stft_true = spec_out[j, :, :, 0]
                    phase_true = spec_out[j, :, :, 1]
    
                    stft_pred = spec_generated[j, :, :, 0]
    
                    if diff_gen:
                        phase_pred = diff_phase_generated
                    else:
                        phase_pred = spec_generated[j, :, :, 1]
    
                    loss_stft = np.mean(amplitude_loss(stft_true, stft_pred))
                    loss_phase = np.mean(phase_loss(phase_true, phase_pred))
                    loss = np.mean(amplitude_loss(spec_out[j], spec_generated[j]))
    
                    total_loss.append(loss)
                    amp_loss.append(loss_stft)
                    pha_loss.append(loss_phase)
    
                    num = tf.norm((stft_pred - stft_true), ord=2)
                    den = tf.norm(stft_true, ord=2)
                    loss_missa_amp = 20 * math.log10(num / den)
    
                    missa_amp_loss.append(loss_missa_amp)
    
                    characteristic_out = characteristic[j, :, 1]
                    wav_true = loader.load(characteristic_out[1] + '/RIR.npy')
    
                    loss_wav = np.mean(amplitude_loss(wav_true, wav_pred[:len(wav_true)]))
                    wav_loss.append(loss_wav)
    
                    loss_wav_50ms = np.mean(amplitude_loss(wav_true[:2400], wav_pred[:2400]))
                    wav_loss_50ms.append(loss_wav_50ms)
    
                    num = tf.norm((wav_pred[:len(wav_true)] - wav_true), ord=2)
                    den = tf.norm(wav_true, ord=2)
                    loss_missa_wav = 20 * math.log10(num / den)
    
                    missa_wav_loss.append(loss_missa_wav)
    
                    sdr_metric_wav = sdr(wav_pred[:len(wav_true)], wav_true)
                    sdr_metric.append(sdr_metric_wav)
    
                    similarity = calculate_similarity(wav_true, wav_pred[:len(wav_true)])
                    similarity_metric.append(similarity)
    
                    if characteristic_out[0] == 'SmallMeetingRoom':
                        small_count += 1
    
                        small_total_loss.append(loss)
                        small_amp_loss.append(loss_stft)
                        small_pha_loss.append(loss_phase)
    
                        small_wav_loss.append(loss_wav)
                        small_wav_loss_50ms.append(loss_wav_50ms)
    
                        small_missa_amp_loss.append(loss_missa_amp)
                        small_missa_wav_loss.append(loss_missa_wav)
    
                        small_sdr_metric.append(sdr_metric_wav)
                        small_similarity_metric.append(similarity)
    
                    end_loss = time.time()
                    time_loss.append(end_loss - start_loss)
    
                    if plot_countdown == 1 and plot:
                        
                        create_directory_if_none(f'{saving_path}/{model_name + modifier}_{algorithm}/png/{model}/')
                        plot_feature_vs_wav(stft_pred, wav_pred, model_name + modifier, characteristic_out,
                                            f'{saving_path}/{model_name + modifier}_{algorithm}/png/{model}/spec_vs_wav_{plot_count}.png')
                        plot_feature_vs_feature_wav(wav_true, stft_true, stft_pred, model_name + modifier,
                                                    characteristic_out,
                                                    f'{saving_path}/{model_name + modifier}_{algorithm}/png/{model}/spec_vs_spec_{plot_count}.png')
                        plot_phase_vs_phase(phase_true, phase_pred, model_name + modifier, characteristic_out,
                                            f'{saving_path}/{model_name + modifier}_{algorithm}/png/{model}/phase_vs_phase_{plot_count}.png')
                        plot_wav_vs_wav(wav_true, wav_pred, model_name + modifier, characteristic_out,
                                        f'{saving_path}/{model_name + modifier}_{algorithm}/png/{model}/wav_vs_wav_{plot_count}.png')
                        plot_count += 1
                        plot_countdown = 0
                    else:
                        plot_countdown += 1
                        
                    
    
                progBar.update(i)
    
            progBar.update(test_generator.__len__(), finalize=True)
    
            end = time.time()
            total_loss = np.mean(total_loss)
            amp_loss = np.mean(amp_loss)
            pha_loss = np.mean(pha_loss)
            wav_loss = np.mean(wav_loss)
            wav_loss_50ms = np.mean(wav_loss_50ms)
            missa_amp_loss = np.mean(missa_amp_loss)
            missa_wav_loss = np.mean(missa_wav_loss)
            sdr_metric = np.mean(sdr_metric)
            similarity_metric = np.mean(similarity_metric)
    
            small_total_loss = np.mean(small_total_loss)
            small_amp_loss = np.mean(small_amp_loss)
            small_pha_loss = np.mean(small_pha_loss)
            small_wav_loss = np.mean(small_wav_loss)
            small_wav_loss_50ms = np.mean(small_wav_loss_50ms)
            small_missa_amp_loss = np.mean(small_missa_amp_loss)
            small_missa_wav_loss = np.mean(small_missa_wav_loss)
            small_sdr_metric = np.mean(small_sdr_metric)
            small_similarity_metric = np.mean(small_similarity_metric)
    
            time_inference = np.mean(time_inference[2:])
            time_postprocessing = np.mean(time_postprocessing[2:])
            time_loss = np.mean(time_loss[2:])
    
            time_data = {
                "mode": [infering_mode],
                "n_samples": [numUpdates * emb.shape[0]],
                "t_model_inference_avg": [float(np.format_float_positional(time_inference, precision=5))],
                "batch_size": [emb.shape[0]],
                "t_postprocess": [float(np.format_float_positional(time_postprocessing, precision=5))],
                "t_loss_calc": [float(np.format_float_positional(time_loss, precision=5))],
                "t_global": [float(np.format_float_positional((end - start), precision=5))]
            }
    
            loss_data = {
                "mode": [infering_mode],
                "room": ['Small'],
                "n samples": [numUpdates * emb.shape[0]],
                "MSE spectrogram": [float(np.format_float_positional(small_total_loss, precision=4))],
                "MSE magnitude": [float(np.format_float_positional(small_amp_loss, precision=4))],
                "1-cos(y-y_) phase": [float(np.format_float_positional(small_pha_loss, precision=4))],
                "MSE waveform": [float(np.format_float_scientific(small_wav_loss, precision=4))],
                "MSE waveform 50ms": [float(np.format_float_scientific(small_wav_loss_50ms, precision=4))],
                "Misalignment magnitude": [float(np.format_float_scientific(small_missa_amp_loss, precision=4))],
                "Misalignment waveform": [float(np.format_float_scientific(small_missa_wav_loss, precision=4))],
                "SDR": [float(np.format_float_scientific(small_wav_loss, precision=4))],
                "Similarity": [float(np.format_float_scientific(small_similarity_metric, precision=4))]
            }
    
            time_dataframe = pd.DataFrame(time_data)
            loss_dataframe = pd.DataFrame(loss_data)
            
            time_dataframes.append(time_dataframe)
            loss_dataframes.append(loss_dataframe)

            final_time_dataframe = pd.concat(time_dataframes, ignore_index=True)
            final_loss_dataframe = pd.concat(loss_dataframes, ignore_index=True)
            
            print(final_time_dataframe)
        
        print('Done! Clearing cache and allocated memory')
        del trained_model
        K.clear_session()
        del dataset
        del test_generator
        gc.collect()
    
    n = 0
    slice = len(algorithms)
    final_time_dataframe.to_csv(f'{saving_path}/infer_time.csv', index=False, decimal=',', sep=';')
    final_loss_dataframe.to_csv(f'{saving_path}/losses_time.csv', index=False, decimal=',', sep=';')
    for algorithm in algorithms:
        
        final_time_dataframe.iloc[n::slice,:].to_csv(f'{saving_path}/{model_name + modifier}_{algorithm}/{model_name + modifier}_infer_time.csv', index=False, decimal=',', sep=';')
        final_loss_dataframe.iloc[n::slice,:].to_csv(f'{saving_path}/{model_name + modifier}_{algorithm}/{model_name + modifier}_losses.csv', index=False, decimal=',', sep=';')
        n += 1
      
    print(ctime(tm()))