from posixpath import join
import cv2
import uuid
import os
import time
import camera
from pathlib import Path

labels = ['thumbsup', 'thumbsdown', 'thankyou', 'livelong']
labels = ['thumbsup']
number_images = 7

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
    photos = camera.Camera()
    for label in labels:
        print("Collecting images for {}".format(label))
        for index in range(number_images):
            print("\tCollecting image {}".format(index))

            image_path = os.path.join(IMAGES_PATH, label)
            photo = photos.capture_compressed_file("{}-{}".format(label, str(uuid.uuid1())), path=image_path, show_photo=True)        