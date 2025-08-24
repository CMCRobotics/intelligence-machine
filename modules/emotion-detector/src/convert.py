import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
import cv2

#
# # 1) Rebuild the exact architecture to use the saved h5 file
# def build_model():
#     model = Sequential()
#     model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
#     model.add(Conv2D(64, (3, 3), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.25))
#
#     model.add(Conv2D(128, (3, 3), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Conv2D(128, (3, 3), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.25))
#
#     model.add(Flatten())
#     model.add(Dense(1024, activation='relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(7, activation='softmax'))
#     return model
#
# model = build_model()
#
# # Load weights (weights-only file). original model
# model.load_weights("model.h5")
# optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, decay=1e-6)
# model.compile(
#     optimizer=optimizer,
#     loss="categorical_crossentropy",
#     metrics=["accuracy"]
# )
# model.save("detect_emotions.keras")

# #Load the full Keras model - trained by me
model = tf.keras.models.load_model("detect_emotions.keras")


#----------------------
#int 8 quantization

rep_datagen = ImageDataGenerator(rescale=1./255)
rep_gen = rep_datagen.flow_from_directory(
    "data/train", target_size=(48,48), color_mode="grayscale",
    class_mode=None, batch_size=1, shuffle=True)

def representative_dataset():
    for i in range(200):          # ~100–300 samples is plenty
        x = next(rep_gen)         # (1,48,48,1) already float32 in [0,1]
        yield [x.astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type  = tf.int8
converter.inference_output_type = tf.int8
tflite_int8 = converter.convert()
open("model_int8.tflite","wb").write(tflite_int8)

