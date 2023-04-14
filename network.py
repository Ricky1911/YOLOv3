import tensorflow as tf
import numpy as np
layers = tf.keras.layers
from utils import yolo_head

class Convolutional(layers.Layer):
    '''conv bn leakyReLu'''
    def __init__(self,
        filters,
        kernel_size,
        strides=(1, 1),
        padding='same',
        data_format=None,
        dilation_rate=(1, 1),
        groups=1,
        activation=None,
        use_bias=False,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None):
        super().__init__()
        self.Conv = layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            groups=groups,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint)

        self.BN = layers.BatchNormalization()
        self.LeakyRelU = layers.LeakyReLU()

    def call(self, inputs, training=None):
        out = self.Conv(inputs)
        out = self.BN(out, training=training)
        out = self.LeakyRelU(out)
        return out

class Residual(layers.Layer):

    def __init__(self, filter_num1, filter_num2):
        super().__init__()
        self.Conv1 = Convolutional(filters=filter_num1, kernel_size=(1,1))
        self.Conv2 = Convolutional(filters=filter_num2, kernel_size=(3,3))

    def call(self, inputs, training=None):
        out = self.Conv1(inputs, training=training)
        out = self.Conv2(out, training=training)
        output = layers.add([inputs, out])
        return output

class Head(layers.Layer):

    def __init__(self, out_channel):
        super().__init__()
        self.out_channel = out_channel
        self.conv2 = layers.Conv2D(out_channel, 1, padding="same")

    def build(self, input_shape):
        in_channel = input_shape[-1]
        self.conv1 = Convolutional(in_channel, 3)

    def call(self, inputs):
        out = self.conv1(inputs)
        out = self.conv2(out)
        return out

class UpSample(layers.Layer):

    def __init__(self):
        super().__init__()
        self.upsample = layers.UpSampling2D(size=(2,2))

    def build(self, input_shape):
        in_channel = input_shape[-1]
        self.conv = Convolutional(in_channel, 3)

    def call(self, inputs):
        out = self.conv(inputs)
        out = self.upsample(out)
        return out

class Convolutional5(layers.Layer):

    def __init__(self):
        super().__init__()
        self.convs = []

    def build(self, input_shape):
        in_channel = input_shape[-1]
        for _ in range(5):
            self.convs.append(Convolutional(in_channel, 3))

    def call(self, inputs):
        out = inputs
        for layer in self.convs:
            out = layer(out)
        return out

class BackBone(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.net = tf.keras.Sequential(layers=
            (layers.Conv2D(filters=32, kernel_size=3),
            Convolutional(filters=64, kernel_size=3, strides=(2,2)),
            Residual(32, 64),
            Convolutional(filters=128, kernel_size=3, strides=(2,2)),
            Residual(64, 128),
            Residual(64, 128),
            Convolutional(filters=256, kernel_size=3, strides=(2,2)))
        )
        for _ in range(8):
            self.net.add(Residual(128, 256))

    def call(self, inputs, training=None):
        out = self.net(inputs, training)
        return out

class Yolov3(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.BackBone = BackBone()
        self.sub1 = [Convolutional(filters=512, kernel_size=3, strides=(2,2))]
        self.sub2 = []
        for _ in range(8):
            self.sub1.append(Residual(256, 512))
        self.sub2.append(Convolutional(filters=1024, kernel_size=3, strides=(2,2)))
        for _ in range(4):
            self.sub2.append(Residual(512, 1024))
        self.sub2.append(Convolutional5())
        self.upsample_1, self.upsample_2 = UpSample(), UpSample()
        self.conv5_1, self.conv5_2 = Convolutional5(), Convolutional5()
        self.Head_1, self.Head_2, self.Head_3 = Head(255), Head(255), Head(255)

    def call(self, inputs):
        f8 = self.BackBone(inputs)
        f16 = f8
        for i in self.sub1:
            f16 = i(f16)
        f32 = f16
        for i in self.sub2:
            f32 = i(f32)
        y1 = self.Head_1(f32)
        r1 = self.upsample_1(f32)
        r1 = layers.concatenate([r1, f16])
        r1 = self.conv5_1(r1)
        y2 = self.Head_2(r1)
        r2 = self.upsample_2(r1)
        r2 = layers.concatenate([r2, f8])
        r2 = self.conv5_2(r2)
        y3 = self.Head_3(r2)
        return y1, y2, y3
    

if __name__ == "__main__":
    array = np.random.random((1,416,416,3))
    model = Yolov3()
    predict = model.predict(array)
    anchors = np.array([[1,1],[2,2],[3,3]])