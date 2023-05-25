import matplotlib.pyplot as plt
from typing import Iterable
import numpy as np
import librosa



def draw_histogram(sample: Iterable, density=True, bins=None):
    if bins is None:
        bins = _choose_bin_number(sample)
    plt.hist(sample, density=density, bins=bins)


def _choose_bin_number(sample):
    q25, q75 = np.percentile(sample, [25, 75])
    bin_width = 2 * (q75 - q25) * len(sample) ** (-1/3)
    bin_num = round(
        (sample.max() - sample.min()) / bin_width
    )
    return bin_num


def _db_to_power(waveform, ref=1.0):
    return np.power(10, waveform / 10) * ref

def mel2wave(melspec, sr=22050, n_fft=2048):
    return librosa.feature.inverse.mel_to_audio(
        melspec,
        sr=sr, n_fft=n_fft
    )

def display_mel_as_wave(melspec, sr=22050, n_fft=2048, use_log=False):
    if not isinstance(melspec, list):
        melspec = [melspec]
    if use_log:
        melspec = [_db_to_power(w) for w in melspec]
        
    waveforms = [mel2wave(w) for w in melspec]
    for w in waveforms:
        librosa.display.waveshow(w)



