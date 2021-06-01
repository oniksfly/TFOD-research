import os
import tensorflow as tf
import wget
import tarfile
import sys
import subprocess
from git.repo.base import Repo
from pathlib import Path
from shutil import copyfile, rmtree
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

CUSTOM_MODEL_NAME = "my_ssd_mobnet"
PRETRAINED_MODEL_NAME = "ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8"
PRETRAINED_MODEL_URL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz"
LABELMAP_NAME = 'label_map.pbtxt'
TF_RECORD_SCRIPT_NAME = "generate_tfrecord.py"

retrain = False
if len(sys.argv) > 0 and 'retrain' in sys.argv:
    print("Model will be retrained")
    retrain = True


root_path = "Tensorflow"
paths = {
    "APIMODEL_PATH": os.path.join(root_path, "models"),
    "ANNOTATION_PATH": os.path.join(root_path, "workspace", 'annotations'),
    "PRETRAINED_MODEL_PATH": os.path.join(root_path, "workspace", "pre-trained-models"),
    "IMAGE_PATH": os.path.join(root_path, 'workspace','images'),
    "CHECKPOINT_PATH": os.path.join(root_path, 'workspace', 'models', CUSTOM_MODEL_NAME)
}

files = {
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABELMAP_NAME),
    'TF_RECORD_SCRIPT': TF_RECORD_SCRIPT_NAME,
    'PIPELINE_CONFIG': os.path.join(paths['CHECKPOINT_PATH'], 'pipeline.config'),
    'TRAIN_RECORD': os.path.join(paths['ANNOTATION_PATH'], 'train.record'),
    'TEST_RECORD': os.path.join(paths['ANNOTATION_PATH'], 'test.record'),
    'TRAINING_SCRIPT': os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'model_main_tf2.py')
}

def create_tf_record(record_path: str, images_path: str, override=False) -> None:
    if os.path.exists(record_path):
        if override:
            os.remove(record_path)
        else:
            return

    print("Generating TF record at " + record_path)

    subprocess.call([
        sys.executable, 
        files['TF_RECORD_SCRIPT'], 
        "-x" + images_path,
        "-l" + files['LABELMAP'],
        "-o" + record_path
    ])

def train_tf_custom_model(train_steps=2000):
    if not os.path.exists(files['TRAINING_SCRIPT']):
        print("Error! Base model doesn't exists")
        return

    subprocess.call([
        sys.executable, 
        files['TRAINING_SCRIPT'], 
        "--model_dir=" + paths['CHECKPOINT_PATH'],
        "--pipeline_config_path=" + files['PIPELINE_CONFIG'],
        "--num_train_steps=" + str(train_steps)
    ])

def eval_tf_custom_model():
    if not os.path.exists(files['TRAINING_SCRIPT']):
        print("Error! Base model doesn't exists")
        return

    subprocess.call([
        sys.executable, 
        files['TRAINING_SCRIPT'], 
        "--model_dir=" + paths['CHECKPOINT_PATH'],
        "--pipeline_config_path=" + files['PIPELINE_CONFIG'],
        "--checkpoint_dir=" + paths['CHECKPOINT_PATH']
    ])

# Remove previous trained model
if retrain and os.path.exists(paths['CHECKPOINT_PATH']):
    print("Remove previously trained model at path " + paths['CHECKPOINT_PATH'])
    rmtree(paths['CHECKPOINT_PATH'])

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
        file.write("\tid: {}\n".format(label['id']))
        file.write('}\n')
    file.close()

# Create TF records from labels and images
create_tf_record(files['TRAIN_RECORD'], os.path.join(paths['IMAGE_PATH'], 'train'), retrain)
create_tf_record(files['TEST_RECORD'], os.path.join(paths['IMAGE_PATH'], 'test'), retrain)

# Copy default pipeline
if not os.path.exists(files['PIPELINE_CONFIG']):
    print('Copying pretrained model pipeline')
    copyfile(
        os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME, 'pipeline.config'),
        os.path.join(files['PIPELINE_CONFIG'])
    )

    # Customize default pipeline
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], "r") as f:
        proto_str = f.read()                                                                                                                                                                                                                                          
        text_format.Merge(proto_str, pipeline_config)

    pipeline_config.model.ssd.num_classes = len(labels)
    pipeline_config.train_config.batch_size = 4
    pipeline_config.train_config.fine_tune_checkpoint = os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME, 'checkpoint', 'ckpt-0')
    pipeline_config.train_config.fine_tune_checkpoint_type = 'detection'
    pipeline_config.train_input_reader.label_map_path= files['LABELMAP']
    pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [os.path.join(paths['ANNOTATION_PATH'], 'train.record')]
    pipeline_config.eval_input_reader[0].label_map_path = files['LABELMAP']
    pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [os.path.join(paths['ANNOTATION_PATH'], 'test.record')]

    config_text = text_format.MessageToString(pipeline_config)
    with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], "wb") as f:                                                                                                                                                                                                                     
        f.write(config_text)


# Generating model training command
train_tf_custom_model()

# Generating model metrics command
eval_tf_custom_model()