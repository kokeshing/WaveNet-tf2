import tensorflow as tf
import numpy as np
import argparse


from model.wavenet import WaveNet
from utils import load_wav, normalize, melspectrogram, inv_mulaw_quantize, save_wav
import hparams

parser = argparse.ArgumentParser()
parser.add_argument('input_path', help="Path of input audio")
parser.add_argument('output_path', help="Path of synthesized audio")
parser.add_argument('weight_path', help="Path of checkpoint (ex:./result/weights/wavenet_0800)")
args = parser.parse_args()


def synthesize(mel_sp, save_path, weight_path):
    wavenet = WaveNet(hparams.num_mels, hparams.upsample_scales)
    wavenet.load_weights(weight_path)
    mel_sp = tf.expand_dims(mel_sp, axis=0)

    outputs = wavenet.synthesis(mel_sp)
    outputs = np.squeeze(outputs)
    outputs = inv_mulaw_quantize(outputs)

    save_wav(outputs, save_path, hparams.sampling_rate)


if __name__ == '__main__':
    wav = load_wav(args.input_path, hparams.sampling_rate)
    wav = normalize(wav) * 0.95

    mel_sp = melspectrogram(wav, hparams.sampling_rate, hparams.num_mels,
                            n_fft=hparams.n_fft, hop_size=hparams.hop_size, win_size=hparams.win_size)

    synthesize(mel_sp, args.output_path, args.weight_path)
