import numpy as np

def find_direct_sound_index(waveform):
     max_index = np.argmax(waveform)
     min_index = np.argmin(waveform)
     if max_index < min_index:
        return max_index
     else:
        return min_index

def wav_timeset(waveform,length_wav,distance,fs):
     num_samples = int((distance / 34300) * fs)
     final_length = int(length_wav*fs)
     
     for idx in range(waveform.shape[1]):
        waveform_unit = waveform[:,idx]
        direct_sound_index = find_direct_sound_index(waveform_unit)
        waveform_unit = np.concatenate((np.zeros(num_samples), waveform_unit[direct_sound_index:]), dtype=np.float32)
        if len(waveform_unit) > final_length:
           waveform_unit = waveform_unit[:final_length]
        elif len(waveform_unit) < final_length:
           waveform_unit = np.concatenate((waveform_unit, np.zeros(final_length - len(waveform_unit))), dtype=np.float32)
        RIRs = np.zeros((waveform_unit.shape[0], waveform.shape[1]))
        RIRs[:,idx] = waveform_unit

     return RIRs
