# %% Constants and imports
import os
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from pydub import AudioSegment
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow import keras
from scipy.io import wavfile
import scipy
import tensorflow_io as tfio

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu_devices[0], True)

EXPECTED_SAMPLE_RATE = 16000
AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 64
EPOCHS = 50
max_padding = 1000


# %% Helper functions


def decode_audio(audio_binary):
    audio, sample = tf.audio.decode_wav(audio_binary, desired_channels=1)
    audio = tf.squeeze(audio, axis=-1)
    sample = tf.cast(sample, dtype=tf.int64)
    audio = tfio.audio.resample(audio, rate_in=sample, rate_out=16000)
    return audio


def get_label(file_path):
    """ get label of file """
    parts = tf.strings.split(file_path, os.path.sep)
    return parts[-2]


def get_spectrogram(waveform):
    zero_padding = tf.zeros([max_padding] - tf.shape(waveform), dtype=tf.float32)
    waveform = tf.cast(waveform, tf.float32)
    equal_length = tf.concat([waveform, zero_padding], 0)
    spectrogram = tf.signal.stft(equal_length, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)
    return spectrogram


def get_waveform_and_label(file_path):
    label = get_label(file_path)
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    return waveform, label


def get_spectrogram_and_label_id(audio, label):
    spectrogram = get_spectrogram(audio)
    spectrogram = tf.expand_dims(spectrogram, -1)
    label_id = tf.argmax(label == commands)
    return spectrogram, label_id


def preprocess_dataset(files):
    files_ds = tf.data.Dataset.from_tensor_slices(files)
    output_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)
    output_ds = output_ds.map(get_spectrogram_and_label_id,  num_parallel_calls=AUTOTUNE)
    return output_ds


def loadData(data_dir):
    commands = np.array(tf.io.gfile.listdir(str(data_dir)))
    commands = commands[commands != 'README.md']
    filenames = tf.io.gfile.glob(str(data_dir) + '/*/*')
    filenames = tf.random.shuffle(filenames)
    return filenames, commands


def loadTrainData(filenames, commands):
    num_samples = len(filenames)
    print('Number of total examples:', num_samples)

    train_data_size = int(num_samples*0.8)
    if (num_samples-train_data_size) % 2 != 0:
        train_data_size += 1
    testandvalidsize = int((num_samples-train_data_size)/2)

    print(f"Using {train_data_size} files for training and {testandvalidsize} for testing/validation")

    train_files = filenames[:train_data_size]
    val_files = filenames[train_data_size:testandvalidsize+train_data_size]
    test_files = filenames[:(num_samples-train_data_size-testandvalidsize)]

    return train_files, val_files, test_files

# %% Data preprocessing


data_dir = "./crydata"
filenames, commands = loadData(data_dir)
train_files, val_files, test_files = loadTrainData(filenames, commands)

files_ds = tf.data.Dataset.from_tensor_slices(train_files)
waveform_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)
for waveform, _ in waveform_ds:
    if waveform.shape[0] > max_padding:
        max_padding = waveform.shape[0]


train_data = preprocess_dataset(train_files)
val_data = preprocess_dataset(val_files)
test_data = preprocess_dataset(test_files)

train_ds = train_data.batch(BATCH_SIZE)
val_ds = val_data.batch(BATCH_SIZE)
spectrogram_ds = waveform_ds.map(get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)

# %% Training

for spectrogram, _ in spectrogram_ds.take(1):
    input_shape = spectrogram.shape
print('Input shape:', input_shape)

num_labels = len(commands)
norm_layer = preprocessing.Normalization()
norm_layer.adapt(spectrogram_ds.map(lambda x, _: x))

model = models.Sequential([
    layers.Input(shape=input_shape),
    preprocessing.Resizing(32, 32),
    norm_layer,
    layers.Conv2D(32, 3, activation='relu'),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_labels),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
)

metrics = history.history
plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.show()

# %% Testing

test_audio = []
test_labels = []

for audio, label in test_data:
    test_audio.append(audio.numpy())
    test_labels.append(label.numpy())

test_audio = np.array(test_audio)
test_labels = np.array(test_labels)

y_pred = np.argmax(model.predict(test_audio), axis=1)
y_true = test_labels

test_acc = sum(y_pred == y_true) / len(y_true)
print(f'Test set accuracy: {test_acc:.0%}')

confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx, xticklabels=commands, yticklabels=commands,
            annot=True, fmt='g')
plt.xlabel('Prediction')
plt.ylabel('Label')
plt.show()

# %%
