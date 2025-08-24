#---------------------------------------------------------------------------------
# train the model or use (tf model .keras:
# acquire the image from camera
# convert image: to grayscale, find faces
# perform inference
# show image with drawn rectangle and detected emotion
#
# to run:
# python emotions.py --mode [train/display]
#
# dataset FER-2013 downloaded from:
# https://www.kaggle.com/datasets/msambare/fer2013?resource=download
# it contains the folder with train and test. copy both to: ../data
#---------------------------------------------------------------------------------


import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# command line argument
ap = argparse.ArgumentParser()
ap.add_argument("--mode",help="train/display")
mode = ap.parse_args().mode


# Define data generators
train_dir = '../data/train'
# val_dir = 'data/test'
# num_train = 28709
# num_val = 7178
batch_size = 64

num_epoch = 2 #50

# train_datagen = ImageDataGenerator(rescale=1./255)
# val_datagen = ImageDataGenerator(rescale=1./255)
#
# train_generator = train_datagen.flow_from_directory(
#         train_dir,
#         target_size=(48,48),
#         batch_size=batch_size,
#         color_mode="grayscale",
#         class_mode='categorical')
#
# validation_generator = val_datagen.flow_from_directory(
#         val_dir,
#         target_size=(48,48),
#         batch_size=batch_size,
#         color_mode="grayscale",
#         class_mode='categorical')

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2   # 20% goes to validation
)

##consider adding augmentation above
# validation_split = 0.2,
# rotation_range = 15,
# width_shift_range = 0.1,
# height_shift_range = 0.1,
# zoom_range = 0.1,
# horizontal_flip = True,
# fill_mode = 'nearest'

# Training generator
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48,48),
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode='categorical',
    subset='training'      # specify training split
)

# Validation generator
validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48,48),
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode='categorical',
    subset='validation'    # specify validation split
)


# Create the model
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

# If you want to train the same model or try other models, go for this
if mode == "train":
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, decay=1e-6)
    model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])
    history = model.fit(
            train_generator,
            steps_per_epoch=train_generator.n // batch_size,
            epochs=num_epoch,
            validation_data=validation_generator,
            validation_steps=validation_generator.n // batch_size)

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


    # model.save_weights('model.weights.h5') #to keep it consistent with original

    # Save model to file
    model.save('../models/detect_emotions.keras') #new way of saving - not only the weights

# emotions will be displayed on your face from the webcam feed
elif mode == "display":
    # model.load_weights('model.h5')
    #model = tf.keras.models.load_model("../models/detect_emotions_personalized.keras")
    model = tf.keras.models.load_model("../models/detect_emotions.keras")

    # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)

    # dictionary which assigns each label an emotion (alphabetical order)
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

    # start the webcam feed
    cap = cv2.VideoCapture(0)
    while True:
        # Find haar cascade to draw bounding box around face
        ret, frame = cap.read()
        if not ret:
            break
        facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Video', cv2.resize(frame,(1600,960),interpolation = cv2.INTER_CUBIC))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()