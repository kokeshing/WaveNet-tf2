import tensorflow as tf

import numpy as np
import os

from model.wavenet import WaveNet
from model.module import CrossEntropyLoss
from dataset import get_train_data
from utils import files_to_list, load_wav, melspectrogram, inv_mulaw_quantize
import hparams


@tf.function
def train_step(model, x, mel_sp, y, loss_fn, optimizer):
    with tf.GradientTape() as tape:
        y_hat = model(x, mel_sp)
        loss = loss_fn(y, y_hat)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss


def train():
    os.makedirs(hparams.result_dir + "weights/", exist_ok=True)

    summary_writer = tf.summary.create_file_writer(hparams.result_dir)

    wavenet = WaveNet(hparams.num_mels, hparams.upsample_scales)

    loss_fn = CrossEntropyLoss(num_classes=256)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(hparams.learning_rate,
                                                                 decay_steps=hparams.exponential_decay_steps,
                                                                 decay_rate=hparams.exponential_decay_rate)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule,
                                         beta_1=hparams.beta_1)

    if hparams.load_path is not None:
        wavenet.load_weights(hparams.load_path)
        step = np.load(hparams.result_dir + "weights/step.npy")
        step = step
        print(f"weights load: {hparams.load_path}")
    else:
        step = 0

    test_files = files_to_list(hparams.test_files)
    test_mels = []
    for i in range(hparams.n_test_samples):
        wav = load_wav(test_files[i], hparams.sampling_rate)
        wav_len = wav.shape[0]
        if wav_len > hparams.sampling_rate * 2:
            sample_num = hparams.sampling_rate * 2 - hparams.sampling_rate * 2 % hparams.hop_size
        else:
            sample_num = wav_len - wav_len % hparams.hop_size

        wav = wav[:sample_num]
        mel_sp = melspectrogram(wav, hparams.sampling_rate, hparams.num_mels,
                                n_fft=hparams.n_fft, hop_size=hparams.hop_size, win_size=hparams.win_size)
        test_mels.append(mel_sp)

        wav = tf.convert_to_tensor(wav, dtype=tf.float32)
        with summary_writer.as_default():
            tf.summary.audio(f'origin_{i}', tf.reshape(wav, [1, -1, 1]), hparams.sampling_rate, step=0)

    for epoch in range(hparams.epoch):
        train_data = get_train_data()
        for x, mel_sp, y in train_data:
            loss = train_step(wavenet, x, mel_sp, y, loss_fn, optimizer)
            with summary_writer.as_default():
                tf.summary.scalar('train/loss', loss, step=step)

            step += 1

        if step % hparams.gen_interval == 0:
            for i, mel_sp in enumerate(test_mels):
                mel_sp = tf.expand_dims(mel_sp, axis=0)
                wavenet.init_queue()
                outputs = wavenet.synthesis(mel_sp)
                outputs = np.squeeze(outputs)
                outputs = inv_mulaw_quantize(outputs)
                outputs = tf.convert_to_tensor(outputs, dtype=tf.float32)
                with summary_writer.as_default():
                    tf.summary.audio(f'generate_{i}', tf.reshape(outputs, [1, -1, 1]),
                                     hparams.sampling_rate, step=step)

        if step % hparams.save_interval == 0:
            print(f'Step {step}, Loss: {loss}')
            np.save(hparams.result_dir + f"weights/step.npy", np.array(step))
            wavenet.save_weights(hparams.result_dir + f"weights/wavenet_{epoch:04}")

    np.save(hparams.result_dir + f"weights/step.npy", np.array(step))
    wavenet.save_weights(hparams.result_dir + f"weights/wavenet_{epoch:04}")

if __name__ == '__main__':
    train()
