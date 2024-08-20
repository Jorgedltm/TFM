# ================================================================
# Parsing command line arguments
# ----------------------------------------------------------------
# Author:                    Maja Taseska, ESAT-STADIUS, KU LEUVEN
# ================================================================
import argparse
import numpy as np
import os


# === FUNCTION: Parsing command line arguments
def _parse():

    # Load the defaults
    defaults = np.load('_data/defaults.npy', allow_pickle = True).item()

    parser = argparse.ArgumentParser(description='Setting the parameters for DOA estimation \n ----------------------------------------------------------------------')
    #---
    parser.add_argument("-f", "--fs", type = int, help=" The sampling rate (make sure it matches that of your audio interface). Default: 44100 Hz.", default = defaults['fs'])
    #---
    parser.add_argument("-dur", "--duration", type = float, help=" The duration of recording.", default = defaults['duration'])
    #---
    parser.add_argument("-m","--mics", nargs=3, type = int, help = "Cartesian coordiantes of the microphone array, units cm.", default = [0,0,0])
    #---
    parser.add_argument("-s","--spk", nargs=3, type = int, help = "Cartesian coordiantes of the speaker, units cm.", default = [0,0,0])
    #---
    parser.add_argument("-chin", "--inputChannelMap", nargs='+', type=int, help = "Input channel mapping", default = defaults['inputChannelMap'])

    parser.add_argument("-chou", "--outputChannelMap", nargs='+', type=int, help = "Output channel mapping", default = defaults['outputChannelMap'])

    #--- arguments for checking and selecting audio interface
    parser.add_argument("-rec", "--record", type = bool, help = "Save recordings ", default = False)

    parser.add_argument("-outdev", "--outputdevice", type = int, help = "Output device ID.", default = defaults['outputdevice'])

    parser.add_argument("-indev", "--inputdevice", type = int, help = "Input device ID.",default = defaults['inputdevice'])

    parser.add_argument('--listdev', help='List the available devices, indicating the default one',action='store_true')

    parser.add_argument('--defaults', help = 'List the default measurement parameters (devices, channels, and signal properties)', action = 'store_true')

    parser.add_argument('--setdev', help='Use this keyword in order to change the default audio interface.',action='store_true')


    args = parser.parse_args()

    return args


#------------------------------------------------
# === FUNCTION: Update defaults

def _defaults(args):

    if (args.listdev == False and  args.defaults == False):
        defaults = {
            "duration" : args.duration,
            "fs" : args.fs,
            "record": args.record,
            "inputChannelMap" : args.inputChannelMap,
            "outputChannelMap": args.outputChannelMap,
            "inputdevice": args.inputdevice,
            "outputdevice": args.outputdevice,
        }
        np.save('_data/defaults.npy', defaults)


#-------------------------------------------------------------
# === FUNCTION: Check if a file with defaults exists. If not, make one

def _checkdefaults():

    flag_defaultsInitialized = True

    if not os.path.exists('_data'):
        os.makedirs('_data')

    if not os.path.exists('_data/defaults.npy'):
        print("Default settings not detected. Creating a defaults file in _data")
        defaults = {
            "duration" : 0.5,
            "fs" : 16000,
            "record" : False,
            "inputChannelMap" : [1],
            "outputChannelMap": [1],
            "inputdevice": 0,
            "outputdevice": 1,
        }
        np.save('_data/defaults.npy', defaults)
        flag_defaultsInitialized = False

    return flag_defaultsInitialized
