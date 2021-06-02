import training
import cv2
import numpy as np
from object_detection.utils import label_map_util

model = training.load_train_model_from_checkpoint()

categories_index = label_map_util.create_category_index_from_labelmap(training.files['LABELMAP'])

capture = cv2.VideoCapture(0)
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

while capture.isOpened():
    ret, frame = capture.read()

    detected = training.detect_image(frame, model, categories_index)
    cv2.imshow('object detection',  np.array(detected))

    if cv2.waitKey(10) & 0xFF == ord('q'):
        capture.release()
        capture.destroyAllWindows()
        break