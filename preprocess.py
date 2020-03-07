import tensorflow as tf
import numpy as np
import os

from utils import *
import hparams


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _int64_array_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float32_array_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def audio_preprocess(wav_path):
    wav = load_wav(wav_path, hparams.sampling_rate)
    wav = trim_silence(wav, top_db=40, fft_size=2048, hop_size=512)
    wav = normalize(wav) * 0.95

    mel_sp = melspectrogram(wav, hparams.sampling_rate, hparams.num_mels,
                            n_fft=hparams.n_fft, hop_size=hparams.hop_size, win_size=hparams.win_size)

    pad = (wav.shape[0] // hparams.hop_size + 1) * hparams.hop_size - len(wav)
    wav = np.pad(wav, (0, pad), mode='constant', constant_values=0.0)
    assert len(wav) % hparams.hop_size == 0

    wav = mulaw_quantize(wav, 255)

    mel_sp_channels, mel_sp_frames = mel_sp.shape
    mel_sp = mel_sp.flatten()
    record = tf.train.Example(features=tf.train.Features(feature={
        'wav': _int64_array_feature(wav),
        'mel_sp': _float32_array_feature(mel_sp),
        'mel_sp_frames': _int64_feature(mel_sp_frames),
    }))

    return record


def createTFRecord():
    os.makedirs(hparams.result_dir, exist_ok=True)

    train_files = files_to_list(hparams.train_files)
    with tf.io.TFRecordWriter(hparams.result_dir + "train_data.tfrecord") as writer:
        for wav_path in train_files:
            record = audio_preprocess(wav_path)
            writer.write(record.SerializeToString())


if __name__ == '__main__':
    createTFRecord()
