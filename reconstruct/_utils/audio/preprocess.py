import librosa
import numpy as np
from typing import Any, Iterable
import os
from pathlib import Path
'''
    main reference:
    https://github.com/musikalkemist/generating-sound-with-neural-networks 
'''

def zero_pad(waveform, sample_rate=22050, sec=0.74):
    # zero pad input waveform at both ends according to duaration specified by sec
    wave_len = waveform.shape[-1]
    target_len = int(sample_rate * sec)
    if wave_len >= target_len:
        return waveform[:target_len]
    else:
        pad = target_len - wave_len
        left_pad = pad // 2; right_pad = pad - left_pad
        padded_waveform = np.concatenate(
            [
                np.zeros(*waveform.shape[:-1], left_pad),
                waveform,
                np.zeros(*waveform.shape[:-1], right_pad)
            ], 
            axis=-1
        )
        return padded_waveform
    
class Loader:
    def __init__(self, sample_rate=22050, mono=True):
        self.sample_rate = sample_rate
        self.mono = mono
    def load(self, file_path):
        waveform, _ = librosa.load(
            file_path,
            sr=self.sample_rate,
            mono=self.mono
        )
        return waveform
    def __call__(self, file_path):
        return self.load(file_path)
        

class MinMaxNormalizer:
    def __init__(self, min_val, max_val):
        '''
            invertible operation which maps value in array to another interval [min_val, max_val]        
        '''
        self.min_val = min_val
        self.max_val = max_val
        
        self.orignal_min_val = None
        self.original_max_val = None
        
    def normalize(self, array):
        self.orignal_min_val = array.min()
        self.original_max_val = array.max()
        
        array = (array - array.min()) / (array.max() - array.min())
        mapped_array = array * (self.max_val - self.min_val) + self.min_val
        return mapped_array
    
    def __call__(self, array):
        return self.normalize(array)
    
    def denormalize(self, array):
        assert self.original_max_val is not None and self.orignal_min_val is not None
        array = (array - array.min()) / (array.max() - array.min())
        original_array = \
            array * (self.original_max_val - self.orignal_min_val) + self.orignal_min_val
        return original_array


class LogSpectrogramExtractor:
    def __init__(self, frame_size, hop_length):
        self.frame_size = frame_size
        self.hop_length = hop_length
    
    def extract(self, waveform):
        stft = librosa.stft(
            waveform,
            n_fft=self.frame_size,
            hop_length=self.hop_length
        )[:-1]
        spectrogram = np.abs(stft).astype(np.float32)
        log_spectrogram = librosa.amplitude_to_db(spectrogram)
        return log_spectrogram

    def __call__(self, waveform):
        return self.extract(waveform)
    
    
class MelSpectrogramExtractor:
    def __init__(self, sr=22050):
        self.sr = sr
    def extract(self, waveform):
        melspec = librosa.feature.melspectrogram(waveform, sr=self.sr)
        return melspec
        

class Padder:
    def __init__(self, sample_rate=22050, duration=0.74, mode='lr'):
        self.sample_rate = sample_rate
        self.duartion = duration
        self.mode = mode
    def pad(self, waveform):
        return zero_pad(waveform, sample_rate=self.sample_rate, sec=self.duartion)
    def __call__(self, waveform):
        return self.pad(waveform)
    
    
class Pipeline:
    def __init__(
        self, 
        loader=None,
        extractor=None, 
        padder=None,
        normalizer=None,
    ):
        self._loader = loader
        self._padder = padder
        self._extractor = extractor
        self._normalizer = normalizer
    
    @property
    def loader(self):
        if self._loader is None:
            return Loader(sample_rate=22050, mono=True)
        else:
            return self._loader
    @property
    def extractor(self):
        if self._extractor is None:
            return LogSpectrogramExtractor(
                frame_size=512, hop_length=256
            )
        else:
            return self._extractor
    @property
    def padder(self):
        if self._padder is None:
            return Padder(sample_rate=22050, duration=0.74)
        else:
            return self._padder
    
    @property
    def normalizer(self):
        if self._normalizer is None:
            return MinMaxNormalizer(0, 1)
        else:
            return self._normalizer

    def process(
        self,
        file_iterator: Iterable,
        expand_dim=True,
        verbose=False,
        on_the_fly=False,
        save_path=None,
    ):
        processed = []
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        for file in file_iterator:
            waveform = self._process_file(file)
            
            if expand_dim == True:
                waveform = waveform[np.newaxis, ...]
                
            if on_the_fly:
                processed.append(waveform)
            if save_path is not None:
                _filename = Path(os.path.basename(file)).with_suffix('.npy')
                _save_path = os.path.join(
                    save_path, _filename
                )
                np.save(_save_path, waveform)
            if verbose:
                print(f'{file} processed')
            
        if on_the_fly:
            return np.array(processed)
        
    
    def __iter__(self):
        _pipeline = iter([
            self.loader, self.padder,
            self.extractor, self.normalizer
        ])
        return _pipeline
    
    def _process_file(self, filepath):
        x = filepath
        for f in self:
            x = f(x)
        return x
        
        
    
    
        