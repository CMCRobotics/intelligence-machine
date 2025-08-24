#--------------------------------------------------------------------------------
# takes specified number of images
# creates the folder if necessary
# converts to grayscale, searches for face/s, resizes to 48x 48
# shows and saves the image to the folder for emotion specified from command line
#
# to run:
# python take_photo.py --emotion [happy] --number [10]
#
# TODO: consider using random number to save the images
#---------------------------------------------------------------------------------

import argparse
import cv2
import os


# command line argument
ap = argparse.ArgumentParser()
ap.add_argument("--emotion",help="angry/disgust/fear/happy/neutral/sad/surprise")
ap.add_argument("--number",help="integer number of photos to take")

emotion = ap.parse_args().emotion
image_number = int(ap.parse_args().number)


DIR = "../data/new_images"
IMAGE_WIDTH = 48
IMAGE_HEIGHT = 48


# Create the directory
try:
    os.makedirs(os.path.join(DIR, emotion))
except PermissionError:
    print(f"Permission denied: Unable to create '{os.path.join(DIR, emotion)}'.")
except Exception as e:
    print(f"An error occurred: {e}")


# prevents openCL usage and unnecessary logging messages
cv2.ocl.setUseOpenCL(False)

# start the webcam feed
cap = cv2.VideoCapture(0)


while image_number:
    # Find haar cascade to draw bounding box around face
    ret, frame = cap.read()
    if not ret:
        break
    facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        res = cv2.resize(roi_gray, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)
        cv2.imshow('Video', cv2.resize(res, (1600, 960), interpolation=cv2.INTER_CUBIC))
        image_number = image_number - 1
        cv2.imwrite(os.path.join(DIR, emotion, f"{emotion}_{image_number}.jpg"), res)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()