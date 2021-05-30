import cv2
import uuid
import os
import time
from pathlib import Path

labels = ['thumbsup', 'thumbsdown', 'thankyou', 'livelong']
number_images = 5

IMAGES_PATH = os.path.join("Tensorflow", "workspace", "images", "collectedimages")


Path(IMAGES_PATH).mkdir(parents=True, exist_ok=True)

for label in labels:
    sub_folder = os.path.join(IMAGES_PATH, label)
    Path(sub_folder).mkdir(parents=True, exist_ok=True)