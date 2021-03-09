from dataclasses import dataclass

@dataclass
class TestConfig:
    speech_folder_path : str = ''
    songs_folder_path : str = ''
    batch_size : int = 16