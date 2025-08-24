import cv2
import numpy as np
import os

from PIL import Image
from matplotlib import pyplot as plt



# --- OpenCV setup ---
cv2.ocl.setUseOpenCL(False)
facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # load once

data_dir = "data/new_data" # input: data/new_data/<class>/.jpg
DST_DIR_OUT = "data/processed"   # output: data/processed/<class>/.jpg
SIZE = (48, 48)

subfolders = [x for x in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, x))]

images = []
labels = []
for subfolder in subfolders:

    exists = os.path.exists(os.path.join(DST_DIR_OUT, subfolder))
    if not exists:
        # Create a new directory because it does not exist
        os.makedirs(os.path.join(DST_DIR_OUT, subfolder))

    for file in os.listdir(os.path.join(data_dir, subfolder)):
        labels.append(subfolder)
        image = cv2.imread(os.path.join(data_dir, subfolder, file))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            res= cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            images.append(res)
            # plt.imshow(res, cmap='gray', vmin=0, vmax=255)
            # plt.show()

            # Save
            # out_name = file
            out_path = os.path.join(DST_DIR_OUT, subfolder, file)
            print (out_path)
            cv2.imwrite(out_path, res)

