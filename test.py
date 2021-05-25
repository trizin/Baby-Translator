# %%
import os

from IPython import display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import tensorflow_io as tfio

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu_devices[0], True)

FILENAME = "./test.wav"  # File to read


@tf.function
def load_wav_16k_mono(filename):
    """ read in a waveform file and convert to 16 kHz mono """
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(
        file_contents,
        desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav


my_classes = ['Belly Pain', 'Burping', 'Tired', 'Hungry', 'Discomfort']

saved_model_path = './model'  # Trained model path

reloaded_model = tf.saved_model.load(saved_model_path)

reloaded_results = reloaded_model(load_wav_16k_mono(FILENAME))
result = my_classes[tf.argmax(reloaded_results)]
probs = tf.nn.softmax(reloaded_results)*100
probs = tf.nn.softmax(reloaded_results)
plt.bar(my_classes, probs)
plt.show()
print(f'Probabilites: {reloaded_results}')
print(f'The main sound is: {result}')

# %%
