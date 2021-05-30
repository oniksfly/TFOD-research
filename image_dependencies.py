from posixpath import join
import cv2
import uuid
import os
import time
from pathlib import Path

labels = ['thumbsup', 'thumbsdown', 'thankyou', 'livelong']
number_images = 5

IMAGES_PATH = os.path.join("Tensorflow", "workspace", "images", "collectedimages")
IMAGES_TEST_PATH = os.path.join("Tensorflow", "workspace", "images", "test")
IMAGES_TTRAIN_PATH = os.path.join("Tensorflow", "workspace", "images", "train")
LABEL_PATH = os.path.join("Tensorflow", "labels")

Path(IMAGES_PATH).mkdir(parents=True, exist_ok=True)
Path(IMAGES_TEST_PATH).mkdir(parents=True, exist_ok=True)
Path(IMAGES_TTRAIN_PATH).mkdir(parents=True, exist_ok=True)
Path(LABEL_PATH).mkdir(parents=True, exist_ok=True)

for label in labels:
    sub_folder = os.path.join(IMAGES_PATH, label)
    Path(sub_folder).mkdir(parents=True, exist_ok=True)

if False: 
    for label in labels:
        cap = cv2.VideoCapture(0)
        print("Collecting images for {}".format(label))
        time.sleep(5)

        for index in range(number_images):
            print("\tCollecting image {}".format(index))

            ret, frame = cap.read()

            image_path = os.path.join(IMAGES_PATH, label, "{}-{}.jpg".format(label, str(uuid.uuid1())))
            cv2.imwrite(image_path, frame)
            cv2.imshow('frame', frame)
            time.sleep(2)
        
        cap.release()
        cv2.destroyAllWindows()

        