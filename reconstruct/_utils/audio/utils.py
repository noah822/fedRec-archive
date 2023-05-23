import librosa
import soundfile
import pathlib

def save_mel_as_wav(melspec, save_path, sr=22050):
    _suffix = pathlib.Path(save_path).suffix
    assert _suffix == '.wav'
    
    spectrogram = librosa.feature.inverse.mel_to_audio(
        melspec, sr=sr
    )
    _save_sp_as_wav(spectrogram, save_path, sr=sr)


def _save_sp_as_wav(data, save_path, sr=22050):
    soundfile.write(save_path, data, sr)