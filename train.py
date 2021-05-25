# %% Import dependencies
import os

from IPython import display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import tensorflow_io as tfio

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu_devices[0], True)
max_padding = 1000
# %% Load yamnet

yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
yamnet_model = hub.load(yamnet_model_handle)

# %% Helper functions


@tf.function
def load_wav_16k_mono(filename, test=False):
    """ read in a waveform file and convert to 16 kHz mono """
    global max_padding
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(
        file_contents,
        desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    if test == False:
        zero_padding = tf.zeros([max_padding] - tf.shape(wav), dtype=tf.float32)
        waveform = tf.cast(wav, tf.float32)
        wav = tf.concat([waveform, zero_padding], 0)
    return wav


def get_label(file_path):
    """ get label of file """
    parts = file_path.split(os.path.sep)
    return parts[-2]


def loadData(data_dir):
    commands = np.array(tf.io.gfile.listdir(str(data_dir)))
    commands = commands[commands != 'README.md']
    filenames = tf.io.gfile.glob(str(data_dir) + '/*/*')
    filenames = tf.random.shuffle(filenames)
    return filenames, commands


def loadTrainData(filenames):
    num_samples = len(filenames)
    print('Number of total examples:', num_samples)

    train_data_size = int(num_samples*0.9)
    if (num_samples-train_data_size) % 2 != 0:
        train_data_size += 1
    testandvalidsize = int((num_samples-train_data_size)/2)

    print(f"Using {train_data_size} files for training and {testandvalidsize} for testing/validation")

    train_files = filenames[:train_data_size]
    val_files = filenames[train_data_size:testandvalidsize+train_data_size]
    test_files = filenames[:(num_samples-train_data_size-testandvalidsize)]

    return train_files, val_files, test_files


def getLabels(file):
    return file, namedict[get_label(file)]


def load_wav_for_map(filename, label):
    return load_wav_16k_mono(filename), label


def extract_embedding(wav_data, label):
    ''' run YAMNet to extract embedding from the wav data '''
    scores, embeddings, spectrogram = yamnet_model(wav_data)
    num_embeddings = tf.shape(embeddings)[0]
    return (embeddings,
            tf.repeat(label, num_embeddings))


def get_waveform_and_label(file_path):
    waveform = load_wav_16k_mono(file_path, test=True)
    return waveform


def preprocessCell(files):

    filenames = []
    labels = []
    for i in files:
        filenames.append(i)
        labels.append(namedict[get_label(str(i.numpy()))])

    files_ds = tf.data.Dataset.from_tensor_slices((filenames, labels))
    files_ds = files_ds.map(load_wav_for_map)
    files_ds = files_ds.map(extract_embedding)
    return files_ds


def setMaxPadding(train_files):
    global max_padding
    files_ds = tf.data.Dataset.from_tensor_slices(train_files)
    waveform_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=tf.data.AUTOTUNE)
    for waveform in waveform_ds:
        if waveform.shape[0] > max_padding:
            max_padding = waveform.shape[0]

# %% Data preprocessing


data_dir = "./crydata"  # Load data from crydata directory https://github.com/gveres/donateacry-corpus
filenames, commands = loadData(data_dir)
namedict = {i: n for n, i in enumerate(commands)}
dictoname = {n: i for n, i in enumerate(commands)}

setMaxPadding(filenames)
# %% test
train_files, val_files, test_files = loadTrainData(filenames)

train_ds = preprocessCell(train_files)
val_ds = preprocessCell(val_files)
test_ds = preprocessCell(test_files)

# %% Cache datas
train_ds = train_ds.cache().shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.cache().batch(32).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.cache().batch(32).prefetch(tf.data.AUTOTUNE)


# %% Create model
my_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1024), dtype=tf.float32,
                          name='input_embedding'),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(len(commands))
], name='my_model')

my_model.summary()

my_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                 optimizer="adam",
                 metrics=['accuracy'])

callback = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                            patience=9,
                                            restore_best_weights=True)
# %% Train the model

history = my_model.fit(train_ds,
                       epochs=150,
                       validation_data=val_ds,
                       callbacks=callback)
# %% Export and save the model


class ReduceMeanLayer(tf.keras.layers.Layer):
    def __init__(self, axis=0, **kwargs):
        super(ReduceMeanLayer, self).__init__(**kwargs)
        self.axis = axis

    def get_config(self):
        config = super().get_config().copy()
        return config

    def call(self, input):
        return tf.math.reduce_mean(input, axis=self.axis)


saved_model_path = './model'  # .h5  # save the model

input_segment = tf.keras.layers.Input(shape=(), dtype=tf.float32, name='audio')
embedding_extraction_layer = hub.KerasLayer(yamnet_model_handle,
                                            trainable=False, name='yamnet')
_, embeddings_output, _ = embedding_extraction_layer(input_segment)
serving_outputs = my_model(embeddings_output)
serving_outputs = ReduceMeanLayer(axis=0, name='classifier')(serving_outputs)
serving_model = tf.keras.Model(input_segment, serving_outputs)
serving_model.save(saved_model_path, include_optimizer=False)

# %%
