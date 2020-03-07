import tensorflow as tf


class ReLU(tf.keras.layers.ReLU):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def call(self, inputs, is_synthesis=False):
        return super().call(inputs)


class Conv1D(tf.keras.layers.Conv1D):
    def __init__(self, filters, kernel_size, strides=1, padding='causal',
                 dilation_rate=1, residual_channels=None, *args, **kwargs):
        super().__init__(filters, kernel_size, strides=strides, padding=padding,
                         dilation_rate=dilation_rate)

        self.k = kernel_size
        self.d = dilation_rate

        if kernel_size > 1:
            self.queue_len = kernel_size + (kernel_size - 1) * (dilation_rate - 1)
            self.queue_dim = residual_channels
            self.init_queue()

    def build(self, input_shape):
        super().build(input_shape)

        self.linearized_weights = tf.cast(tf.reshape(self.kernel, [-1, self.filters]), dtype=tf.float32)

    def call(self, inputs, is_synthesis=False):
        if not is_synthesis:
            return super().call(inputs)

        if self.k > 1:
            self.queue = self.queue[:, 1:, :]
            self.queue = tf.concat([self.queue, tf.expand_dims(inputs[:, -1, :], axis=1)], axis=1)

            if self.d > 1:
                inputs = self.queue[:, 0::self.d, :]
            else:
                inputs = self.queue

        outputs = tf.matmul(tf.reshape(inputs, [1, -1]), self.linearized_weights)
        outputs = tf.nn.bias_add(outputs, self.bias)

        # [batch_size, 1(time_len), channels]
        return tf.reshape(outputs, [-1, 1, self.filters])

    def init_queue(self):
        self.queue = tf.zeros([1, self.queue_len, self.queue_dim], dtype=tf.float32)


class ResidualConv1DGLU(tf.keras.Model):
    """
        conv1d + GLU => add condition => residual add + skip connection
    """

    def __init__(self, residual_channels, gate_channels, kernel_size,
                 skip_out_channels=None, dilation_rate=1, **kwargs):
        super().__init__()

        self.residual_channels = residual_channels

        if skip_out_channels is None:
            skip_out_channels = residual_channels

        self.dilated_conv = Conv1D(gate_channels,
                                   kernel_size=kernel_size,
                                   padding='causal',
                                   dilation_rate=dilation_rate,
                                   residual_channels=residual_channels)

        self.conv_c = Conv1D(gate_channels,
                             kernel_size=1,
                             padding='causal')

        self.conv_skip = Conv1D(skip_out_channels,
                                kernel_size=1,
                                padding='causal')
        self.conv_out = Conv1D(residual_channels,
                               kernel_size=1,
                               padding='causal')

    @tf.function
    def call(self, inputs, c):
        x = self.dilated_conv(inputs)
        x_tanh, x_sigmoid = tf.split(x, num_or_size_splits=2, axis=2)

        c = self.conv_c(c)
        c_tanh, c_sigmoid = tf.split(c, num_or_size_splits=2, axis=2)

        x_tanh, x_sigmoid = x_tanh + c_tanh, x_sigmoid + c_sigmoid
        x = tf.nn.tanh(x_tanh) * tf.nn.sigmoid(x_sigmoid)

        s = self.conv_skip(x)
        x = self.conv_out(x)

        x = x + inputs

        return x, s

    def init_queue(self):
        self.dilated_conv.init_queue()

    def synthesis_feed(self, inputs, c):
        x = self.dilated_conv(inputs, is_synthesis=True)
        x_tanh, x_sigmoid = tf.split(x, num_or_size_splits=2, axis=2)

        c = self.conv_c(c, is_synthesis=True)
        c_tanh, c_sigmoid = tf.split(c, num_or_size_splits=2, axis=2)

        x_tanh, x_sigmoid = x_tanh + c_tanh, x_sigmoid + c_sigmoid
        x = tf.nn.tanh(x_tanh) * tf.nn.sigmoid(x_sigmoid)

        s = self.conv_skip(x, is_synthesis=True)
        x = self.conv_out(x, is_synthesis=True)

        x = x + inputs

        return x, s


class CrossEntropyLoss(tf.keras.losses.Loss):
    def __init__(self, num_classes=256, name=None):
        super().__init__()
        self.num_classes = num_classes

    def call(self, targets, outputs):
        targets_ = tf.one_hot(targets, depth=self.num_classes)
        losses = tf.nn.softmax_cross_entropy_with_logits(labels=targets_, logits=outputs)

        return tf.reduce_mean(losses)
