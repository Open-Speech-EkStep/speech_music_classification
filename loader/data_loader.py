from pathlib import Path

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from configs.train_config import SpectConfig

def load_audio(audio_path, samp_rate=16000):
    """Loads the audio signal stored at the given path."""

    audio, sr = librosa.load(audio_path, sr=samp_rate)
    return audio  

class SpectrogramParser():
    def __init__(self, 
                 audio_conf: SpectConfig):
        """Parses the audio file to a spectrogram"""

        self.sr = audio_conf.samp_rate
        self.n_fft = audio_conf.n_fft
        self.win_length = int(self.sr * audio_conf.win_dur)
        self.hop_length = int(self.sr * audio_conf.win_stride) 

    def parse_audio(self, audio_path):
        """ Computes and returns the spectrogram of the given audio file."""

        y = load_audio(audio_path, self.sr)
        D = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length,
                         win_length=self.win_length)
        spect, phase = librosa.magphase(D)
        spect = np.log1p(spect)
        spect = torch.FloatTensor(spect)
        
        # Global normalization
        mean = spect.mean() 
        std = spect.std()
        spect.add_(-mean)
        spect.div_(std)

        return spect

class SpectrogramDataset(Dataset, SpectrogramParser):
    def __init__(self, 
                 input_path: str,
                 label: int,
                 audio_conf: SpectConfig):
        """
        The spectrogram dataset for speech/music classifcation.
        Loads tensors using the audio files in the given dir path.
        """

        self.label = label
        self.audio_paths = list(Path(input_path).glob("**/*.wav"))
        self.size = len(self.audio_paths)
        super(SpectrogramDataset, self).__init__(audio_conf)

    def __getitem__(self, index):
        audio_path = self.audio_paths[index]
        spect = self.parse_audio(audio_path)
        return spect, self.label

    def __len__(self):
        return self.size
