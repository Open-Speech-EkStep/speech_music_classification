from dataclasses import dataclass

@dataclass
class SpectConfig:
    samp_rate : int = 16000 # Sampling rate for extracting spectrogram features
    n_fft : int = 512 # Length of the windowed signal after padding
    win_dur : float = 0.025 # Window size in seconds
    win_stride : float = 0.01 # Window stride in seconds

@dataclass
class TrainConfig:
    speech_folder_path : str = ''
    songs_folder_path : str = ''
    lr : float = 0.0001
    batch_size : int = 16
    num_epochs : int = 50