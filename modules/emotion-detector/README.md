pi_stream.pi for raspbery pi - takes images, searches for faces, performs inference - detects one of 7 emotions and sends it to the webserver (hosted by pi) 
pi_detect_emotions.py - for raspberry pi - takes images, searches for faces, performs inference - detects one of 7 emotions and displays it.

for both above --model [base/personalized] will specify whether use a basic model or newly retrained. 
both use pi camera module v2

take_photo.py is a tool to run on the laptop - takes images - performs processing (searches for a face, converts to graycale, crops/scales etc) and saves to  folder with requested emotion. 

retrain.py is a tool that loads the basic model, performs a test with test dataset - shows accuracy and confusion matrix. then freezes most of the model layers and performs retrainig with the new images (from the previous script- but it needs images for every class) - retraining with 10 images (i include augmentation though) for each class took a few seconds - then tests with a test dataset, shows accuracy and confusion matrix. asks whether to save new model. deletes the used images. 

emotions.py is the main scrip that creates a model and trains it, you can also just load the model and use it - meaning perform live inference with the images from laptop camera.


files expected on pi location /home/pi/projects/tinyml/emotions/:
pi_stream.py
pi_detect_emotions.py
haarcascade_frontalface_default.xml
model_int8.tflite
