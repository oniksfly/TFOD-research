import os
import object_detection
import wget
import tarfile
from git.repo.base import Repo
from pathlib import Path

CUSTOM_MODEL_NAME = "my_ssd_mobnet"
PRETRAINED_MODEL_NAME = "ssd_mobilenet_v2_320x320_coco17_tpu-8"
PRETRAINED_MODEL_URL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz"
LABELMAP_NAME = 'label_map.pbtxt'

root_path = "Tensorflow"
paths = {
    "APIMODEL_PATH": os.path.join(root_path, "models"),
    "ANNOTATION_PATH": os.path.join(root_path, "workspace", 'annotations'),
    "PRETRAINED_MODEL_PATH": os.path.join(root_path, "workspace", "pre-trained-models")
}

files = {
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABELMAP_NAME)
}

for path in paths.values():
    Path(path).mkdir(parents=True, exist_ok=True)

if not os.path.exists(os.path.join(paths["APIMODEL_PATH"], "research", "object_detection")):
    print("Cloning TF models to {}".format(paths["APIMODEL_PATH"]))
    Repo.clone_from("https://github.com/tensorflow/models.git", paths["APIMODEL_PATH"])

if not os.path.exists(os.path.join(paths["PRETRAINED_MODEL_PATH"], PRETRAINED_MODEL_NAME)):
    print("Download pretrained TF model {}".format(PRETRAINED_MODEL_NAME))
    filename = wget.download(PRETRAINED_MODEL_URL, out=paths["PRETRAINED_MODEL_PATH"])
    archive = tarfile.open(filename)
    archive.extractall(paths["PRETRAINED_MODEL_PATH"])
    archive.close()
    os.remove(filename)

labels = [
    {
        'name': 'ThumbsUp',
        'id': 1
    },
    {
        'name': 'ThumbsDown',
        'id': 2
    },
    {
        'name': 'ThankYou',
        'id': 3
    },
    {
        'name': 'LiveLong',
        'id': 4
    },
]
if not os.path.exists(files['LABELMAP']):
    print("Creating label's map at {}".format(files['LABELMAP']))
    file = open(files['LABELMAP'], 'w')

    for label in labels:
        file.write('item {\n')
        file.write("\tname: '{}'\n".format(label['name']))
        file.write("\tid: '{}'\n".format(label['id']))
        file.write('}\n')
    file.close()