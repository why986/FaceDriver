import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, BatchNormalization, ReLU, Layer

class SampleLayer(Layer):
    def __init__(self, num_out, kernel_size, stride):
        super(SampleLayer, self).__init__()
        self.n_out = num_out
        self.k_size = kernel_size
        self.stride = stride

        self.conv1 = Conv1D(self.n_out, 3, strides=self.stride, padding='same')
        self.batch_norm = BatchNormalization()
        self.relu = ReLU()
        self.conv2 = Conv1D(self.n_out, 3, strides=self.stride, padding='same')
    
    def call(self, inputs):
        out = self.conv1(inputs)
        out = self.batch_norm(out)
        out = self.relu(out)
        out = self.conv2(out)
        return out


class AudioLayer(Layer):
    def __init__(self, audio_out=48, energy_out=16):
        super(AudioLayer, self).__init__()
        self.audio_out = audio_out
        self.energy_out = energy_out

        self.audio_sample = SampleLayer(audio_out, 3, 1)
        self.energy_sample = SampleLayer(energy_out, 3, 1)
    
    def call(self, audio, energy):
        out_audio = self.audio_sample(audio)
        out_energy = self.energy_sample(energy)
        return tf.concat([out_audio, out_energy], axis=-1)