import os
import tensorflow as tf
from tensorflow.python.training.checkpoint_management import latest_checkpoint
import wget
import tarfile
import sys
import re
import subprocess
import numpy as np
from git.repo.base import Repo
from pathlib import Path
from shutil import copyfile, rmtree
from object_detection.protos import pipeline_pb2
from object_detection.utils import config_util
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from google.protobuf import text_format
from PIL import Image
from matplotlib import pyplot as plt

CUSTOM_MODEL_NAME = "my_ssd_mobnet"
PRETRAINED_MODEL_NAME = "ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8"
PRETRAINED_MODEL_URL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz"
LABELMAP_NAME = 'label_map.pbtxt'
TF_RECORD_SCRIPT_NAME = "generate_tfrecord.py"

retrain, export = False, False
if len(sys.argv) > 0:
    if 'retrain' in sys.argv:
        print("Model will be retrained")
        retrain = True
    if 'export' in sys.argv:
        print("Model will be export")
        export = True


root_path = "Tensorflow"
paths = {
    "APIMODEL_PATH": os.path.join(root_path, "models"),
    "ANNOTATION_PATH": os.path.join(root_path, "workspace", 'annotations'),
    "PRETRAINED_MODEL_PATH": os.path.join(root_path, "workspace", "pre-trained-models"),
    "IMAGE_PATH": os.path.join(root_path, 'workspace','images'),
    "CHECKPOINT_PATH": os.path.join(root_path, 'workspace', 'models', CUSTOM_MODEL_NAME),
    "OUTPUT_PATH": os.path.join(root_path, 'workspace', 'models', CUSTOM_MODEL_NAME, 'export'),
    "OUTPUT_LITE_PATH": os.path.join(root_path, 'workspace', 'models', CUSTOM_MODEL_NAME, 'export_lite'),
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

def train_tf_custom_model(train_steps=2000, override=False):
    if not os.path.exists(files['TRAINING_SCRIPT']):
        print("Error! Base model doesn't exists")
        return

    if override or not os.path.exists(os.path.join(paths['CHECKPOINT_PATH'], "checkpoint")):
        subprocess.call([
            sys.executable, 
            files['TRAINING_SCRIPT'], 
            "--model_dir=" + paths['CHECKPOINT_PATH'],
            "--pipeline_config_path=" + files['PIPELINE_CONFIG'],
            "--num_train_steps=" + str(train_steps)
        ])

def eval_tf_custom_model(override=False):
    if not os.path.exists(files['TRAINING_SCRIPT']):
        print("Error! Base model doesn't exists")
        return

    if override or not os.path.exists(os.path.join(paths['CHECKPOINT_PATH'], "eval")):
        subprocess.call([
            sys.executable, 
            files['TRAINING_SCRIPT'], 
            "--model_dir=" + paths['CHECKPOINT_PATH'],
            "--pipeline_config_path=" + files['PIPELINE_CONFIG'],
            "--checkpoint_dir=" + paths['CHECKPOINT_PATH']
        ])


def load_train_model_from_checkpoint():
    if not os.path.exists(os.path.join(paths['CHECKPOINT_PATH'], "checkpoint")):
        print("No checkpoints found for custom model")
        return

    configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
    detection_model = model_builder.build(model_config=configs['model'], is_training=False)

    checkpoint = tf.compat.v2.train.Checkpoint(model = detection_model)
    latest_checkpoint = None
    pattern = re.compile("^(ckpt-([\d+]))\.index")
    for number in [ pattern.search(x)[2] for x in os.listdir(paths['CHECKPOINT_PATH']) if pattern.search(x) ]:
        cuurent_checkpoint = latest_checkpoint if latest_checkpoint else 0
        checkpoint_number = int(number)
        if checkpoint_number > cuurent_checkpoint:
            latest_checkpoint = checkpoint_number

    if latest_checkpoint:
        checkpoint.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-' + str(latest_checkpoint))).expect_partial()
    else:
        print("No checkpoints found")

    return detection_model

def detect_object(model, image_tenzor):
    image, shapes = model.preprocess(image_tenzor)
    prediction = model.predict(image, shapes)
    detections = model.postprocess(prediction, shapes)

    return detections

def detect_image(image: Image, model, categories_index) -> Image:
    image_array = np.array(image)

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_array, 0), dtype=tf.float32)
    detections = detect_object(model, input_tensor)

    num_detections = int(detections.pop('num_detections'))

    detections = { key: value[0, :num_detections].numpy() for key, value in detections.items() }

    detections['num_detections'] = num_detections

    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1

    image_array_with_detections = image_array.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_array_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'] + label_id_offset,
        detections['detection_scores'],
        categories_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=5,
        min_score_thresh=.8,
        agnostic_mode=False
    )

    return Image.fromarray(image_array_with_detections)

def model_export():
    """
    Export TF model
    """

    # Check if custom congig present
    if not os.path.exists(files['PIPELINE_CONFIG']):
        print("Custom model's piplene config missing")
        return

    export_script = os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'exporter_main_v2.py')
    # Check if custom congig present
    if not os.path.exists(export_script):
        print("Export script missing")
        return

    subprocess.call([
        sys.executable, 
        export_script, 
        "--input_type=image_tensor",
        "--pipeline_config_path=" + files['PIPELINE_CONFIG'],
        "--trained_checkpoint_dir=" + paths['CHECKPOINT_PATH'],
        "--output_directory=" + paths['OUTPUT_PATH'],
    ])

def model_export_lite():
    """
    Export lite TF model
    """

    # Check if custom congig present
    if not os.path.exists(files['PIPELINE_CONFIG']):
        print("Custom model's piplene config missing")
        return

    export_script = os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'export_tflite_graph_tf2.py')
    # Check if custom congig present
    if not os.path.exists(export_script):
        print("Export script missing")
        return

    subprocess.call([
        sys.executable, 
        export_script, 
        "--pipeline_config_path=" + files['PIPELINE_CONFIG'],
        "--trained_checkpoint_dir=" + paths['CHECKPOINT_PATH'],
        "--output_directory=" + paths['OUTPUT_LITE_PATH'],
    ])

    single_file = os.path.join(paths['OUTPUT_LITE_PATH'], 'saved_model', 'detect.tflite')

    subprocess.call([
        'tflite_convert', 
        "--saved_model_dir=" + os.path.join(paths['OUTPUT_LITE_PATH'], 'saved_model'),
        "--output_file=" + single_file,
        "--input_shapes=1,300,300,3",
        "--input_arrays=normalized_input_image_tensor",
        "--output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3'",
        "--inference_type=FLOAT",
        "--allow_custom_ops",
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
train_tf_custom_model(override=retrain)

# Generating model metrics command
eval_tf_custom_model(override=retrain)

# Export TF model
if export:
    model_export()
    model_export_lite()