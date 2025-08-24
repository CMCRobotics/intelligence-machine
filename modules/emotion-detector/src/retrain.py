#-----------------------------------------------------------------------------------------------------------
# Retrain existing model with new images
#
# workflow:
# load base model - which one to choose and how?
# evaluate the model with test data, show accuracy, confusion matrix
#
# read the images from new_images directory <-  IT NEEDS TO CONTAIN IMAGES FOR EACH CLASS!
# retrain
# evaluate the model with test data, show accuracy, confusion matrix
#
# prompt to save and convert new model?
# save new model as keras
# convert to tf lite
# save as tflite
# delete those images from new_images directory
#
# TODO: IMAGE_WIDTH and IMAGE_HEIGHT to ext module?
#
# i fail to send the new model to pi from this script, we may need to just run this command at the end:
# $ scp /home/kasik/Documents/Books/ai/tinyML/projects/emotion_detection/Emotion-detection/src/model_int8_personalized.tflite pi@192.168.3.231:/home/pi/projects/tinyml/emotions
#------------------------------------------------------------------------------------------------------------


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import shutil, os
import numpy as np

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
# import subprocess
# import paramiko
# from scp import SCPClient

IMAGE_WIDTH = 48
IMAGE_HEIGHT = 48
test_dir = '../data/test'
new_data_dir='../data/processed' #it needs to contain all classes

# Labels
emotion_dict = {
    0:"Angry", 1:"Disgusted", 2:"Fearful",
    3:"Happy", 4:"Neutral", 5:"Sad", 6:"Surprised"
}

def representative_dataset():
    for i in range(200):  # ~100–300 samples is plenty
        x = next(rep_gen)  # (1,48,48,1) already float32 in [0,1]
        yield [x.astype(np.float32)]


test_datagen = ImageDataGenerator(
    rescale=1./255,              # normalize pixel values to [0,1]

)

test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(48,48),
        batch_size=32,
        color_mode="grayscale",
        class_mode='categorical')

model = tf.keras.models.load_model("../models/detect_emotions.keras")

# Evaluate
loss, acc = model.evaluate(test_generator)
print(f"Test loss: {loss:.4f}, Test accuracy: {acc:.4f}")


# Compute confusion matrix
y_true = test_generator.classes
y_pred = np.argmax(model.predict(test_generator), axis=1)
cm = confusion_matrix(y_true, y_pred)

labels = [emotion_dict[i] for i in range(len(emotion_dict))]

sns.heatmap(cm,
            annot=True,
            fmt='g',
            xticklabels=labels,
            yticklabels=labels)
plt.ylabel('Prediction', fontsize=13)
plt.xlabel('Actual', fontsize=13)
plt.title('Confusion Matrix', fontsize=17)
plt.show()


#prepare for retrain
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,      # 20% of TRAIN used for validation
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_gen = train_datagen.flow_from_directory(
    new_data_dir,
    target_size=(48,48),
    batch_size=32,
    color_mode="grayscale",
    class_mode="categorical",
    subset="training"
)

val_gen = train_datagen.flow_from_directory(
    new_data_dir,
    target_size=(48,48),
    batch_size=32,
    color_mode="grayscale",
    class_mode="categorical",
    subset="validation",
    shuffle=False
)


# # Freeze all layers except the final Dense
# for layer in model.layers[:-1]:
#     layer.trainable = False


# Freeze layers
for layer in model.layers[-5:]:
    layer.trainable = True

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, decay=1e-6)
model.compile(
    # optimizer=tf.keras.optimizers.Adam(3e-3),
    optimizer = optimizer,
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor="val_accuracy")
]

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=50,
    callbacks=callbacks
)

# Evaluate
loss, acc = model.evaluate(test_generator)
print(f"Test loss: {loss:.4f}, Test accuracy: {acc:.4f}")


# Compute confusion matrix
y_true = test_generator.classes
y_pred = np.argmax(model.predict(test_generator), axis=1)
cm = confusion_matrix(y_true, y_pred)

sns.heatmap(cm,
            annot=True,
            fmt='g',
            xticklabels=labels,
            yticklabels=labels)
plt.ylabel('Prediction', fontsize=13)
plt.xlabel('Actual', fontsize=13)
plt.title('Confusion Matrix', fontsize=17)
plt.show()

# Save the model
print("save?")
s = input()
if s == 'y':
    #save new model as. keras
    model.save("../models/detect_emotions_personalized.keras")

    #prepare to convert
    rep_datagen = ImageDataGenerator(rescale=1. / 255)
    rep_gen = rep_datagen.flow_from_directory(
        "../data/train", target_size=(IMAGE_WIDTH, IMAGE_HEIGHT), color_mode="grayscale",
        class_mode=None, batch_size=1, shuffle=True)

    #convert to tflite and save
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_int8 = converter.convert()
    open("../models/model_int8_personalized.tflite", "wb").write(tflite_int8)

    #delete the newly used images
    # if os.path.exists(new_data_dir):
    #     shutil.rmtree(new_data_dir)
    # os.makedirs(new_data_dir)

    # #send to PI
    # src = "/home/kasik/Documents/Books/ai/tinyML/projects/emotion_detection/Emotion-detection/src/model_int8_personalized.tflite"
    # dst = "/home/pi/projects/tinyml/emotions/model_int8_personalized.tflite"
    #
    # ssh = paramiko.SSHClient()
    # ssh.load_system_host_keys()
    # ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    #
    # ssh.connect("192.168.3.231", username="pi", password="") #the IP may be different in the venue?
    #
    # with SCPClient(ssh.get_transport()) as scp:
    #     scp.put(src, dst)
    #
    # ssh.close()
