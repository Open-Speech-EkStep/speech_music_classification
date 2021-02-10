import glob
from pathlib import Path

import librosa
from tqdm import tqdm
from joblib import Parallel, delayed


def _get_duration(audio_path):
    """Returns the duration of the given audio file."""
    return librosa.get_duration(filename=audio_path)

def calc_total_duration(folder_path, num_workers=1):
    """ Calculates the total duration of all the wav files present inside the given folder and returns it."""

    audio_paths = list(Path(folder_path).glob("**/*.wav"))
    audio_durs = Parallel(n_jobs=num_workers)(delayed(_get_duration)(audio_path) for audio_path in tqdm(audio_paths))
    total_dur = sum(audio_durs) / 3600 # Total duration in hours

    return total_dur

def get_total_duration():
    songs_dur = calc_total_duration('/home/soma/song_speech/songs', -1)
    speech_dur = calc_total_duration('/home/soma/song_speech/speech', -1)

    print(f'Total Songs Duration: {songs_dur:.3f} hrs Total Speech Duration: {speech_dur:.3f} hrs')

if __name__=="__main__":
    get_total_duration()
