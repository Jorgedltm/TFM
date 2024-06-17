import os
import random
import librosa
from matplotlib import pyplot as plt
import zipfile
import numpy as np
from tqdm import tqdm
from roomsv2 import UTSRoom
from preprocessv2 import Normalizer, FeatureExtractor, Loader, TensorPadder

class EmbedGen():
    
    def __init__(self, dir_dataset, dataset_name, extract=False, normalization=True, debugging=False,
                 room_characteristics=False, room=None, array=None, zone=None, normalize_vector=False):
        
        'Initialization'

        if room is None or room == ['All']:
            self.rooms = ['SmallMeetingRoom']
        else:
            self.rooms = room
        if array is None:
            self.array = ['PlanarMicrophoneArray', 'CircularMicrophoneArray'] # Modify
        else:
            self.array = array
        if zone is None:
            self.zones = ['ZoneA', 'ZoneB', 'ZoneC', 'ZoneD', 'ZoneE'] # Modify
        else:
            self.zones = zone

        self.Small_Room = None

        self.normalizer = None
        self.extractor = None
        self.loader = None
        self.padder = None

        self.Spectrograms_Amp = []
        self.Spectrograms_Pha = []
        self.Embeddings_list = []
        self.Embeddings = []
        self.characteristics = []

        self.index_small = []

        self.index_in = []
        self.index_out = []

        self.dir_dataset = dir_dataset
        self.dataset_name = dataset_name

        'Constants'
        self.n_fft = 256
        self.win_length = 128
        self.hop_length = 64

        self.duration = 0.2  # in seconds
        self.sr = 48000
        self.mono = True

        self.input_shape = (144, 160)

        self.normalization = normalization
        self.debugging = debugging
        self.room_characteristics = room_characteristics

        self.normalize_vector = normalize_vector

        self.min_dim = 10e5
        self.max_dim = 0

        self.min_angle = 10e5
        self.max_angle = 0

        self.min_pos = 10e5
        self.max_pos = 0

        self.min_height = 10e5
        self.max_height = 0

        self.min_t60 = 10e5
        self.max_t60 = 0

        self.seed = 500  # Seed for consistency at selecting training / validation and test datasets

        self.set_rooms()
        self.set_preprocessers()
        self.load_data()
        
    def set_rooms(self):
        self.Small_Room = UTSRoom(310, 450, 310, 450, 90, 90, 90, 90, 300, [155, 225], 45) # Mi room

    def set_preprocessers(self):
        self.normalizer = Normalizer()
        self.extractor = FeatureExtractor(n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_length)
        self.loader = Loader(sample_rate=self.sr, duration=self.duration, mono=self.mono)
        self.padder = TensorPadder(desired_shape=self.input_shape)

    def load_data(self):
        dataset_path = self.dir_dataset + '/'
        rir_folders = os.listdir(dataset_path)
        index = 0
        print('Loading and preprocessing data...')
        for rir_folder in rir_folders:
            
            if rir_folder != "lastRecording":
                dir_path = os.path.join(dataset_path + f"/{rir_folder}")

                print(f'Accessing {rir_folder}')

                characteristics = ['SmallMeetingRoom',dir_path]

                rir_path = os.path.join(dir_path + "/RIR.npy")

                embedding = self.get_embedding(rir_folder, characteristics, index)
                amp, phase = self.preprocess(rir_path)
                #librosa.display.specshow(librosa.amplitude_to_db(amp,ref=np.max), sr=48000, hop_length=64, x_axis="time", y_axis="linear",cmap='jet')
                #plt.show()
                self.Spectrograms_Amp.append(amp)
                self.Spectrograms_Pha.append(phase)

                self.Embeddings_list.append(embedding)

                if self.room_characteristics:
                    self.characteristics.append(characteristics)

                index += 1

        self.Embeddings = np.array(self.Embeddings_list).astype(np.float32)

        if self.normalize_vector:
            self.normalize_vector_embedding()

        # Initial shuffle
        
        self.index_in = self.index_small

        random.Random(self.seed).shuffle(self.index_small)
        
        self.index_out = self.index_small


    def get_embedding(self, rir_folder, characteristics, index):

        if characteristics[0] == "SmallMeetingRoom":
            embedding = self.Small_Room.return_embedding(rir_folder)
            self.index_small.append(index)

        if self.normalize_vector:
            self.obtain_min_max_vector(embedding)

        return embedding

    def normalize_vector_embedding(self):
        
        self.Embeddings[..., 0:4] = (self.Embeddings[..., 0:4] - self.min_dim) / (self.max_dim - self.min_dim)
        self.Embeddings[..., 4:8] = (self.Embeddings[..., 4:8] - self.min_angle) / (self.max_angle - self.min_angle)
        self.Embeddings[..., 8:14] = (self.Embeddings[..., 8:14] - self.min_pos) / (self.max_pos - self.min_pos)
        if self.max_height == self.min_height:
            self.Embeddings[..., 14:15] = 0.5
        else:
            self.Embeddings[..., 14:15] = (self.Embeddings[..., 14:15] - self.min_height) / (self.max_height - self.min_height)
        self.Embeddings[..., 15:16] = (self.Embeddings[..., 15:16] - self.min_t60) / (self.max_t60 - self.min_t60)

    def obtain_min_max_vector(self, embedding):
        
        # Values obtained from training
        
        min_dim = 355
        max_dim = 1175

        min_angle = 81
        max_angle = 105

        min_pos = 26
        max_pos = 1031

        min_height = 300
        max_height = 529

        min_t60 = 45
        max_t60 = 1281

        if min_dim < self.min_dim:
            self.min_dim = min_dim
        if max_dim > self.max_dim:
            self.max_dim = max_dim

        if min_angle < self.min_angle:
            self.min_angle = min_angle
        if max_angle > self.max_angle:
            self.max_angle = max_angle

        if min_pos < self.min_pos:
            self.min_pos = min_pos
        if max_pos > self.max_pos:
            self.max_pos = max_pos

        if min_height < self.min_height:
            self.min_height = min_height
        if max_height > self.max_height:
            self.max_height = max_height

        if min_t60 < self.min_t60:
            self.min_t60 = min_t60
        if max_t60 > self.max_t60:
            self.max_t60 = max_t60

    def preprocess(self, rir_path):
        wav = self.loader.load(rir_path)
        amp, phase = self.extractor.extract(wav)
        if self.normalization:
            norm_amp, norm_phase = self.normalizer.normalize(amp, phase)
            padded_amp, padded_phase = self.padder.pad_amp_phase(norm_amp, norm_phase)
        else:
            padded_amp, padded_phase = self.padder.pad_amp_phase(amp, phase)

        return padded_amp, padded_phase

    def return_characteristics(self):
        if self.room_characteristics:
            return self.characteristics
        else:
            return None

    def return_min_max_vector(self):
        return np.array((self.min_dim, self.max_dim, self.min_angle, self.max_angle, self.min_pos, self.max_pos, self.min_height,
                self.max_height, self.min_t60, self.max_t60))

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.Spectrograms_Amp)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        idx = index

        # Get features from dataset
        amp = self.Spectrograms_Amp[idx]
        phase = self.Spectrograms_Pha[idx]
        emb = self.Embeddings[idx]

        return amp, phase, emb


if __name__ == "__main__":
    dataset = EmbedGen('../../pyrirtool/recorded_RIR', 'room_impulse',
                      debugging=False,
                      normalization=True,
                      normalize_vector=True)