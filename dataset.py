import tensorflow as tf

import hparams


def parse_function(example_proto):
    features = {
        'wav': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        'mel_sp': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'mel_sp_frames': tf.io.FixedLenFeature([], tf.int64),
    }
    parsed_features = tf.io.parse_single_example(example_proto, features)

    mel_sp = tf.reshape(parsed_features['mel_sp'],
                        [hparams.num_mels, parsed_features['mel_sp_frames']])

    return parsed_features['wav'], mel_sp


def adjust_time_resolution(wav, mel_sp):
    if hparams.seq_len % hparams.hop_size == 0:
        max_steps = hparams.seq_len
    else:
        max_steps = hparams.seq_len - hparams.seq_len % hparams.hop_size

    max_time_frames = max_steps // hparams.hop_size

    mel_offset = tf.random.uniform([1], minval=0, maxval=tf.shape(mel_sp)[1] - max_time_frames,
                              dtype=tf.int32)[0]
    wav_offset = mel_offset * hparams.hop_size

    mel_sp = mel_sp[:, mel_offset:mel_offset + max_time_frames]
    x = wav[wav_offset:wav_offset + max_steps]
    x = tf.one_hot(x, 256, axis=-1, dtype=tf.float32)
    y = wav[wav_offset + 1:wav_offset + max_steps + 1]

    return x, mel_sp, y


def get_train_data():
    train_data = tf.data.TFRecordDataset(filenames=hparams.result_dir + "train_data.tfrecord")\
        .shuffle(300)\
        .map(parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
        .map(adjust_time_resolution, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
        .batch(hparams.batch_size)\
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return train_data

