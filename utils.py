import numpy as np
import librosa
from scipy.io import wavfile


def files_to_list(filename):
    with open(filename, encoding="utf-8") as f:
        files = f.readlines()

    files = [file.rstrip() for file in files]

    return files


def load_wav(path, sampling_rate):
    wav = librosa.core.load(path, sr=sampling_rate)[0]

    return wav


def trim_silence(wav, top_db=40, fft_size=2048, hop_size=512):
    return librosa.effects.trim(wav, top_db=top_db, frame_length=fft_size, hop_length=hop_size)[0]


def normalize(wav):
    return librosa.util.normalize(wav)


def mulaw(x, mu=255):
    return np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)


def mulaw_quantize(x, mu=255):
    x = mulaw(x)
    x = (x + 1) / 2 * mu

    return x.astype(np.int)


def inv_mulaw(x, mu=255):
    return np.sign(x) * (1.0 / mu) * ((1.0 + mu) ** np.abs(x) - 1.0)


def inv_mulaw_quantize(x, mu=255):
    x = 2 * x.astype(np.float32) / mu - 1

    return inv_mulaw(x, mu)


def save_wav(wav, path, sr):
    wav *= 32767 / max(0.0001, np.max(np.abs(wav)))
    wavfile.write(path, sr, wav.astype(np.int16))


def melspectrogram(wav, sampling_rate, num_mels, n_fft, hop_size, win_size):
    d = librosa.stft(y=wav, n_fft=n_fft, hop_length=hop_size,
                     win_length=win_size, pad_mode='constant')
    mel_filter = librosa.filters.mel(sampling_rate, n_fft,
                                     n_mels=num_mels)
    s = np.dot(mel_filter, np.abs(d))

    return np.log10(np.maximum(s, 1e-5))
