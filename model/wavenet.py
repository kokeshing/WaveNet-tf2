import tensorflow as tf
import numpy as np

from .module import Conv1D, ReLU, ResidualConv1DGLU
from .upsample import UpsampleNetwork
from utils import mulaw_quantize


class WaveNet(tf.keras.Model):
    def __init__(self, num_mels, upsample_scales):
        super().__init__()

        self.upsample_network = UpsampleNetwork(upsample_scales)

        self.first_layer = Conv1D(128,
                                  kernel_size=1,
                                  padding='causal')

        self.residual_blocks = []
        for _ in range(2):
            for i in range(10):
                self.residual_blocks.append(
                    ResidualConv1DGLU(128,
                                      256,
                                      kernel_size=3,
                                      skip_out_channels=128,
                                      dilation_rate=2 ** i)
                )

        self.final_layers = [
            ReLU(),
            Conv1D(128,
                   kernel_size=1,
                   padding='causal'),
            ReLU(),
            Conv1D(256,
                   kernel_size=1,
                   padding='causal')
        ]

    @tf.function
    def call(self, inputs, c):
        c = tf.expand_dims(c, axis=-1)
        c = self.upsample_network(c)
        c = tf.transpose(tf.squeeze(c, axis=-1), perm=[0, 2, 1])

        x = self.first_layer(inputs)
        skips = None
        for block in self.residual_blocks:
            x, h = block(x, c)
            if skips is None:
                skips = h
            else:
                skips = skips + h

        x = skips
        for layer in self.final_layers:
            x = layer(x)

        return x

    def init_queue(self):
        for block in self.residual_blocks:
            block.init_queue()

    def synthesis(self, c):
        c = tf.expand_dims(c, axis=-1)
        c = self.upsample_network(c)
        c = tf.transpose(tf.squeeze(c, axis=-1), perm=[0, 2, 1])

        batch_size, time_len, _ = c.shape
        initial_value = mulaw_quantize(0, 256)
        inputs = tf.one_hot(indices=initial_value,
                            depth=256, dtype=tf.float32)
        inputs = tf.tile(tf.reshape(inputs, [1, 1, 256]),
                         [batch_size, 1, 1])

        outputs = []
        for i in range(time_len):
            c_t = tf.expand_dims(c[:, i, :], axis=1)

            x = self.first_layer(inputs, is_synthesis=True)

            skips = None
            for block in self.residual_blocks:
                x, h = block.synthesis_feed(x, c_t)

                if skips is not None:
                    skips = skips + h
                else:
                    skips = h

            x = skips
            for layer in self.final_layers:
                x = layer(x, is_synthesis=True)

            x = tf.argmax(tf.squeeze(x, axis=1), axis=-1)
            x = tf.one_hot(x, depth=256)
            inputs = x

            outputs.append(tf.argmax(x, axis=1).numpy())

        outputs = np.array(outputs)

        return np.transpose(outputs, [1, 0])
