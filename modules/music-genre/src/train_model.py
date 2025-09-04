#-----------------------------------------------------------------------------------------------------------------------
# train the model
# display example MFCC
# display accuracy and loss
# save the .keras model
# convert and save to .tflite model
# Evaluate the accuracy of the quantized TensorFlow Lite model
#
# to run:
# python train_model.py
#
# for now it trains only for: ['disco', 'jazz', 'metal']
#
# dataset GTZAN downloaded from:
# https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification
# it contains the folder with 10 types of music. Each contains 100 of samples ~30s each. copy to: ../data
#-----------------------------------------------------------------------------------------------------------------------


import librosa
import numpy as np
import soundfile as sf
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split

from tensorflow.keras import activations
from tensorflow.keras import layers

# Audio sample rate
SAMPLE_RATE = 22050

# MFFCs constants
FRAME_LENGTH = 2048
FRAME_STEP = 1024
FFT_LENGTH = 2048
FMIN_HZ = 20
FMAX_HZ = SAMPLE_RATE / 2
NUM_MEL_FREQS = 40
NUM_MFCCS = 18

# Define data generators
train_dir = '../data'


def extract_mfccs_tf(
        ad_src,
        ad_sample_rate,
        num_mfccs,
        frame_length,
        frame_step,
        fft_length,
        fmin_hz,
        fmax_hz,
        num_mel_freqs):
    n = ad_src.shape[0]
    num_frames = int(((n - frame_length) / frame_step) + 1)

    output = np.zeros(shape=(num_frames, num_mfccs))

    # Iterate over each frame to get the MFCC coefficients
    for i in range(num_frames):
        idx_s = i * frame_step
        idx_e = idx_s + frame_length
        src = ad_src[idx_s:idx_e]

        # Apply the Hann Window in-place
        hann_coef = tf.signal.hann_window(frame_length)
        hann = src * hann_coef

        # Apply the RFFT
        fft_spect = tf.signal.rfft(hann)

        # Calculate the magnitude of the FFT
        fft_mag_spect = tf.math.abs(fft_spect)

        # Calculate the coefficients of Mel-weights for converting the spectrum from Hz to Mel
        num_fft_freqs = fft_mag_spect.shape[0]
        mel_wei_mtx = tf.signal.linear_to_mel_weight_matrix(
            num_mel_freqs,
            num_fft_freqs,
            ad_sample_rate,
            fmin_hz,
            fmax_hz)

        # Convert the spectrum to Mel
        mel_spect = np.matmul(fft_mag_spect, mel_wei_mtx)

        # Perform the log function
        log_mel_spect = np.log(mel_spect + 1e-6)

        dct = tf.signal.mfccs_from_log_mel_spectrograms(
            log_mel_spect)

        # Extract the MFFC coefficients
        output[i] = dct[0:num_mfccs]

    return output

#test
filepath = train_dir + "/disco/disco.00002.wav"
ad, sr = sf.read(filepath)

print('Sample rate:', sr)
print('Sample shape:', ad.shape)
print('Song duration:', ad.shape[0] / sr)

# # Take one second of audio
test_ad = ad[0:SAMPLE_RATE]

import IPython.display as ipd

ipd.Audio(test_ad, rate=sr)


mfccs_tf = extract_mfccs_tf(
    test_ad,
    SAMPLE_RATE,
    NUM_MFCCS,
    FRAME_LENGTH,
    FRAME_STEP,
    FFT_LENGTH,
    FMIN_HZ,
    FMAX_HZ,
    NUM_MEL_FREQS)

import matplotlib.pyplot as plt
from matplotlib import cm


def display_mfccs(mfcc_src):
    fig, ax = plt.subplots()
    cax = ax.imshow(mfcc_src, interpolation='nearest', cmap=cm.gray, origin='lower')
    ax.set_title('MFCCs')
    plt.xlabel('Frame index - Time')
    plt.ylabel('Coefficient index - Frequency')
    plt.colorbar(cax)
    plt.show()


display_mfccs(mfccs_tf.T)


def extract_mfccs_librosa(
        ad_src,
        ad_sample_rate,
        num_mfccs,
        frame_length,
        frame_step,
        fft_length,
        fmin_hz,
        fmax_hz,
        num_mel_freqs):
    return librosa.feature.mfcc(
        y=ad_src,
        sr=ad_sample_rate,
        n_mfcc=num_mfccs,
        n_fft=fft_length,
        hop_length=frame_step,
        win_length=frame_length,
        center=False,
        n_mels=num_mel_freqs,
        fmin=fmin_hz,
        fmax=fmax_hz)


mfccs_librosa = extract_mfccs_librosa(
    test_ad,
    SAMPLE_RATE,
    NUM_MFCCS,
    FRAME_LENGTH,
    FRAME_STEP,
    FFT_LENGTH,
    FMIN_HZ,
    FMAX_HZ,
    NUM_MEL_FREQS)

display_mfccs(mfccs_librosa)

# Music genres. just to try, can select more and different
LIST_GENRES = ['disco', 'jazz', 'metal']

# Training audio length in seconds
TRAIN_AUDIO_LENGTH_SEC = 1

# Training audio length in number of samples
TRAIN_AUDIO_LENGTH_SAMPLES = SAMPLE_RATE * TRAIN_AUDIO_LENGTH_SEC

# TensorFlow model name
TF_MODEL = 'music_genre.keras'

# TensorFlow lite model name
TFL_MODEL_FILE = 'model_music_int8.tflite'




x = []
y = []

for genre in LIST_GENRES:
  folder = train_dir + "/" + genre

  list_files = os.listdir(folder)

  for song in list_files:
    filepath = folder + "/" + song

    try:
      ad, sr = sf.read(filepath)

      # Number of splits
      num_it = int(len(ad) / TRAIN_AUDIO_LENGTH_SAMPLES)

      for i in range(num_it):
        s0 = TRAIN_AUDIO_LENGTH_SAMPLES * i
        s1 = s0 + TRAIN_AUDIO_LENGTH_SAMPLES
        src_audio = ad[s0 : s1]

        mfccs = extract_mfccs_librosa(
            src_audio,
            SAMPLE_RATE,
            NUM_MFCCS,
            FRAME_LENGTH,
            FRAME_STEP,
            FFT_LENGTH,
            FMIN_HZ,
            FMAX_HZ,
            NUM_MEL_FREQS)

        x.append(mfccs.tolist())
        y.append(LIST_GENRES.index(genre))

    except Exception as e:
      continue

# Convert the x and y lists to NumPy arrays
x, y = np.array(x), np.array(y)

# Split the dataset into train(60 %), validation(20 %), and test(20 %) datasets
# Split 1 (60% vs 40%)
x_train, x0, y_train, y0 = train_test_split(x, y, test_size=0.40, random_state=1)
# Split 2 (50% vs 50%)
x_test, x_validate, y_test, y_validate = train_test_split(x0, y0, test_size=0.50, random_state=3)



# Design a many - to - one LSTM model
input_shape = (x_train.shape[1], x_train.shape[2])

norm_layer = layers.Normalization(axis=-1)

# Learn mean and standard deviation from dataset
norm_layer.adapt(x_train)

input = layers.Input(shape=input_shape)
x = norm_layer(input)
x = layers.LSTM(32, return_sequences=True, unroll=True)(x)
x = layers.LSTM(32, unroll=True)(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(len(LIST_GENRES),
                 activation='softmax')(x)

model = tf.keras.Model(input, x)

# Visualize model summary
model.summary()

optimiser = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)

model.compile(optimizer=optimiser,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

NUM_EPOCHS = 30
BATCH_SIZE = 50

history = model.fit(x_train, y_train, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE,
                    validation_data=(x_validate, y_validate))

### Plot training and validation accuracy and loss over time
# Extract accuracy and loss values (in list form) from the history
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

# Create a list of epoch numbers
epochs = range(1, len(acc) + 1)

# Plot training and validation loss values over time
plt.figure()
plt.plot(epochs, loss, color='blue', marker='.', label='Training loss')
plt.plot(epochs, val_loss, color='orange', marker='.', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

# Plot training and validation accuracies over time
plt.figure()
plt.plot(epochs, acc, color='blue', marker='.', label='Training acc')
# plt.plot(epochs, val_acc, color='orange', marker='.', label='Validation acc')
# plt.title('Training and validation accuracy')
plt.title('Training accuracy')
plt.legend()

# Plot training and validation accuracies over time
plt.figure()
plt.plot(epochs, acc, color='blue', marker='.', label='Training acc')
plt.plot(epochs, val_acc, color='orange', marker='.', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.show()


model.save('../models/music_genre.keras') # i save it here but there is some issue to load such model from file to convert to tflite


def representative_data_gen():
    data = tf.data.Dataset.from_tensor_slices(x_test)
    for i_value in data.batch(1).take(100):
        i_value_f32 = tf.dtypes.cast(i_value, tf.float32)
        yield [i_value_f32]


# Quantize the TensorFlow model with the TensorFlow Lite converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = tf.lite.RepresentativeDataset(representative_data_gen)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# converter.inference_input_type = tf.int8 #do i need int 8? it run well on raspberry PI, so not necessary
# converter.inference_output_type = tf.int8

tfl_model = converter.convert()
with open("../models/music_genre.tflite", "wb") as f:
    f.write(tfl_model)



# Initialize the TensorFlow Lite interpreter
interp = tf.lite.Interpreter(model_content=tfl_model)

# Allocate the tensor and get the input and output details
interp.allocate_tensors()

# Get input/output layer information
i_details = interp.get_input_details()[0]
o_details = interp.get_output_details()[0]


def classify(i_value):
    # Add an extra dimension to the test sample
    # to match the expected 3D tensor shape
    i_value_f32 = np.expand_dims(i_value, axis=0)

    # Quantize (float -> 8-bit) the input
    i_value_f32 = tf.cast(i_value_f32, dtype=tf.float32)
    interp.set_tensor(i_details["index"], i_value_f32)

    interp.invoke()

    # TfLite fused Lstm kernel is stateful.
    # Therefore, we need to reset the states before the next inference
    interp.reset_all_variables()

    return interp.get_tensor(o_details["index"])[0]


# Evaluate the accuracy of the quantized TensorFlow Lite model

num_correct_samples = 0

for i_value, o_value in zip(x_test, y_test):
    o_pred_f32 = classify(i_value)
    o_res = np.argmax(o_pred_f32)
    if o_res == o_value:
        num_correct_samples += 1

num_total_samples = len(x_test)
print("Accuracy:", num_correct_samples / num_total_samples)

