"""
DOA Algorithms
==============

- MUSIC [1]_
- SRP-PHAT [2]_
- CSSM [3]_
- WAVES [4]_
- TOPS [5]_


.. [1] R. Schmidt, *Multiple emitter location and signal parameter estimation*, 
    IEEE Trans. Antennas Propag., Vol. 34, Num. 3, pp 276--280, 1986

.. [2] J. H. DiBiase, J H, *A high-accuracy, low-latency technique for talker localization 
    in reverberant environments using microphone arrays*, PHD Thesis, Brown University, 2000

.. [3] H. Wang, M. Kaveh, *Coherent signal-subspace processing for the detection and 
    estimation of angles of arrival of multiple wide-band sources*, IEEE Trans. Acoust., 
    Speech, Signal Process., Vol. 33, Num. 4, pp 823--831, 1985

.. [4] E. D. di Claudio, R. Parisi, *WAVES: Weighted average of signal subspaces for 
    robust wideband direction finding*, IEEE Trans. Signal Process., Vol. 49, Num. 10, 
    2179--2191, 2001

.. [5] Y. Yeo-Sun, L. M. Kaplan, J. H. McClellan, *TOPS: New DOA estimator for wideband 
    signals*, IEEE Trans. Signal Process., Vol. 54, Num 6., pp 1977--1989, 2006

.. [6] H. Pan, R. Scheibler, E. Bezzam, I. Dokmanić, and M. Vetterli, *FRIDA:
    FRI-based DOA estimation for arbitrary array layouts*, Proc. ICASSP,
    pp 3186-3190, 2017

"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import fftconvolve
import sounddevice as sd
import os
import pandas as pd
from scipy.io.wavfile import write as wavwrite

from tuning import Tuning
import usb.core
import usb.util

dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)

import pyroomacoustics as pra
import librosa
from pyroomacoustics.doa import circ_dist
import _parseargs as parse

flag_defaultsInitialized = parse._checkdefaults()
args = parse._parse()
parse._defaults(args)

def record(dur,fs,inputChannels):
    
    sd.default.samplerate = fs
    sd.default.dtype = 'float32'
    print("Input channels:",  inputChannels)
    
    # Start the recording
    recorded = sd.rec(int(dur*fs),samplerate=fs, mapping = inputChannels)
    sd.wait()

    for i in range (recorded.shape[1]):
        plt.plot(recorded[:,i])
    #plt.show()
    return recorded

def saverecordings(rec_mics,fs):
    
    dirflag = False
    counter = 1
    dirname = 'recorded_mics/rec' + str(counter)
    while dirflag == False:
        if os.path.exists(dirname):
            counter = counter + 1
            dirname = 'recorded_mics/rec' + str(counter)
        else:
            os.mkdir(dirname)
            dirflag = True
            counter = 1


    for idx in range(rec_mics.shape[1]):
        
        wavwrite(dirname + '/mic' + str(idx+1) + '.wav',fs,rec_mics[:,idx])

    print('Success! Recording saved in directory ' + dirname)
    
def main():

    algos = []
    azimuth_recov = []
    error = []
    
    if flag_defaultsInitialized == True:
        
        if args.listdev == True:

            print(sd.query_devices())
            sd.check_input_settings()
            sd.check_output_settings()
            print("Default input and output device: ", sd.default.device )
            return None,None

        elif args.defaults == True:
            aa = np.load('_data/defaults.npy', allow_pickle = True).item()
            for i in aa:
                print (i + " => " + str(aa[i]))
            return None,None

        elif args.setdev == True:

            sd.default.device[0] = args.inputdevice
            sd.default.device[1] = args.outputdevice
            sd.check_input_settings()
            sd.check_output_settings()
            print(sd.query_devices())
            print("Default input and output device: ", sd.default.device )
            print("Sucessfully selected audio devices. Ready to record.")
            parse._defaults(args)
            return None,None

        else:
        
            sd.default.device[0] = args.inputdevice
            sd.default.device[1] = args.outputdevice
            # Record
            recorded = record(args.duration,args.fs,args.inputChannelMap)
            
            if args.record:
                saverecordings(recorded, args.fs)

            #######################
            # Algorithms parameters
            c = 343.0  # speed of sound
            fs = args.fs  # sampling frequency
            nfft = 256  # FFT size
            freq_bins = np.arange(2, 60)  # FFT bins to use for estimationn
            
            # We use a circular array with radius 3.2 cm # and 4 microphones
            R = pra.circular_2D_array([0,0], 4, np.pi/4 , 0.032)

            ################################
            # Compute the STFT frames needed
            X = np.array(
                [
                    pra.transform.stft.analysis(signal, nfft, nfft // 2).T
                    for signal in recorded.T
                ]
            )

            plt.figure()
            for i in range(X.shape[0]):
                plt.subplot(1, X.shape[0], i+1,title="Spectrogram")
                librosa.display.specshow(librosa.amplitude_to_db(np.abs(X[i,:,:])), sr=fs, hop_length=nfft//2, x_axis="time", y_axis="linear", cmap='jet')
            #plt.show()
            
            ##############################################
            # Now we can test all the algorithms available
            algo_names = sorted(pra.doa.algorithms.keys())
            algo_names.remove("FRIDA")
            
            if dev:
                Mic_tuning = Tuning(dev)
                azimuth = Mic_tuning.direction
                print("Real DOA: ",azimuth)
                azimuth = np.deg2rad(azimuth)
            
            for algo_name in algo_names:

                # The max_four parameter is necessary for FRIDA only
                doa = pra.doa.algorithms[algo_name](R, fs, nfft, c=c, max_four=4)
        
                # This call here perform localization on the frames in X
                doa.locate_sources(X, freq_bins=freq_bins)
                
                plt.subplot(1,len(algo_names),algo_names.index(algo_name)+1, title=algo_name)
                plt.plot(doa.grid.values)
                
                algos.append(algo_name)
                azimuth_recov.append(np.degrees(doa.azimuth_recon).astype(np.int16))
                error.append(np.degrees(circ_dist(azimuth, doa.azimuth_recon)).astype(np.int16))
                # DOA.azimuth_recon contains the reconstructed location of the source
            plt.show()
            azimuth_recov_flat = [item for sublist in azimuth_recov for item in sublist]
            error_flat = [item for sublist in error for item in sublist]
            
            for idx in range(len(algos)):
                print(algos[idx])
                print("  Recovered azimuth:", azimuth_recov[idx][0], "degrees")
                print("  Error:", error[idx][0], "degrees")
            
            return algos, error_flat
        

if __name__=="__main__":
    distances = [0.25,0.5,1,2,4]
    reps = 5
    
    filename = "DOA_algos.csv"
        
    for distance in distances:
        
        print("Computing distance ", distance)
        errors = []
        dirname = "CSV_files/distance_" + str(distance)
        
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        
        for i in range(reps):
            print("Rep: ", i+1, " Change sound source position.")
            algos, error = main()
            
            if algos == None:
                quit()
            
            # Asks if the measured DOA should be taken into account
            if (input("Keep? ")=="y"):
                errors.append(np.array(error))
        
        errors = np.vstack(errors)
        mean_errors = np.mean(errors,axis=0)
        
        data = {
                "Algorithm" : algos,
                "Error" : mean_errors
            }
            
        data = pd.DataFrame(data)
        
        data.to_csv(os.path.join(dirname,filename), index=False, mode='w', decimal=',', sep=';')