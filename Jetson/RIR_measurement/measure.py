# ================================================================
# Room impulse response measurement with an exponential sine sweep
# ----------------------------------------------------------------
# Author:                    Maja Taseska, ESAT-STADIUS, KU LEUVEN
# ================================================================
import os
import sys
import sounddevice as sd
import numpy as np
from matplotlib import pyplot as plt
from scipy.io.wavfile import read as wavread
from scipy.signal import resample
from matplotlib import pyplot as plt
import librosa

# modules from this software
import stimulus as stim
import _parseargs as parse
import utils as utils
import peak_finding
import re

# --- Parse command line arguments and check defaults
flag_defaultsInitialized = parse._checkdefaults()
args = parse._parse()
parse._defaults(args)
rec = True
# -------------------------------
if flag_defaultsInitialized == True:

    if args.listdev == True:

        print(sd.query_devices())
        sd.check_input_settings()
        sd.check_output_settings()
        print("Default input and output device: ", sd.default.device )

    elif args.defaults == True:
        aa = np.load('_data/defaults.npy', allow_pickle = True).item()
        for i in aa:
            print (i + " => " + str(aa[i]))

    elif args.setdev == True:

        sd.default.device[0] = args.inputdevice
        sd.default.device[1] = args.outputdevice
        sd.check_input_settings()
        sd.check_output_settings()
        print(sd.query_devices())
        print("Default input and output device: ", sd.default.device )
        print("Sucessfully selected audio devices. Ready to record.")
        parse._defaults(args)

    elif args.test == True:

        deltapeak = stim.test_deconvolution(args)
        plt.plot(deltapeak)
        plt.show()

    else:
        
        if rec:
            dataset_path = 'recorded_RIR/'
            rir_folders = os.listdir(dataset_path)
            rir_folder = rir_folders[0]
            print(rir_folder)
            dir_path = os.path.join(dataset_path + f"/{rir_folder}")
            pattern = r'mic\[(.*?)\]_spk\[(.*?)\]'
            match = re.search(pattern, rir_folder)
            args.mics = list(map(int, match.group(1).split(',')))
            args.spk = list(map(int, match.group(2).split(',')))
            lstfs = 16000
            args.fs = 48000
            
        # Create a test signal object, and generate the excitation
        testStimulus = stim.stimulus('sinesweep', args.fs);
        testStimulus.generate(args.fs, args.duration, args.amplitude,args.reps,args.startsilence, args.endsilence, args.sweeprange)
        
        if rec:
            recorded_path = os.path.join(dir_path + "/sigrec1.wav")
            fs,recorded = wavread(recorded_path)
            print(fs,recorded.shape)
            plt.plot(np.linspace(0,fs/len(recorded),len(recorded)),recorded)
            recorded = resample(recorded,int(len(recorded)*args.fs/lstfs))
            recorded = recorded.reshape(-1,1)
            plt.plot(np.linspace(0,args.fs/len(recorded),len(recorded)),recorded)
            plt.show()
            print(recorded.shape)
        else:
            # Record
            recorded = utils.record(testStimulus.signal,args.fs,args.inputChannelMap,args.outputChannelMap)

        # Deconvolve
        RIR = testStimulus.deconvolve(recorded)

        # Truncate
        lenRIR = 0.2; # RIR length in seconds
        distance = np.sqrt(np.sum((np.array(args.mics) - np.array(args.spk)) ** 2)); # Distamce from speaker to mic in cm
        #startId = int(testStimulus.signal.shape[0]/args.reps) + int(latency*args.fs) - args.endsilence*args.fs -1
        #endId = startId + int(lenRIR*args.fs)
        RIR = peak_finding.wav_timeset(RIR,lenRIR,distance,args.fs)
        RIR_true = np.load(dir_path + "/RIR.npy")
        RIR_fft = librosa.stft(RIR[:,0],n_fft=256,hop_length=64,win_length=128)
        plt.plot(RIR)
        plt.show()
        plt.plot(RIR_true)
        plt.show()
        librosa.display.specshow(librosa.amplitude_to_db(np.abs(RIR_fft),ref=np.max), sr=args.fs, hop_length=64, x_axis="time", y_axis="linear",cmap='jet')
        plt.show()
        sys.exit()
        # save some more samples before linear part to check for nonlinearities
        #startIdToSave = startId - int(args.fs/2)
        #RIRtoSave = RIR[startIdToSave:endId,:]
        #RIR = RIR[startId:endId,:]
        RIRtoSave = RIR
        RIR = RIR
        # Save recordings and RIRs
        utils.saverecording(RIR, RIRtoSave, testStimulus.signal, recorded, args.fs)
