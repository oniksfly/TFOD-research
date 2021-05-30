# Setup

## Setupp required packages
```bash
pip3 install opencv-python pyqt5 lxml gitpython wget
```

```bash
mkdir Tensorflow/labels
```

```bash
git clone https://github.com/tzutalin/labelImg.git Tensorflow/labels
```

```bash
cd Tensorflow/labels && make qt5py3 && python3 labelImg.py
```

### Setup TF2
```bash
brew install protobuf
```

```bash
cd Tensorflow/models/research && protoc object_detection/protos/*.proto --python_out=. && cp object_detection/packages/tf2/setup.py . &&
```

Verify installation: `python3 Tensorflow/models/research/object_detection/builders/model_builder_tf2_test.py`
## Activate virual env
```bash
source ./bin/activate
```

# Analyze results 
Use Tensorboard to check training and test results

```bash
tensorboard --logdir=Tensorflow/workspace/models/my_ssd_mobnet/train
```

```bash
tensorboard --logdir=Tensorflow/workspace/models/my_ssd_mobnet/eval
```

# Links
* [How to use AMD GPU on Hackintosh](https://kenthinson.com/machine-learning-with-gpu-on-mac-osx/)
* [How to capture DSLR stream to webcam on Linux](https://medium.com/nerdery/dslr-webcam-setup-for-linux-9b6d1b79ae22)
* [How to pass OpenCV images to FFMPEG](https://stackoverflow.com/questions/34167691/pipe-opencv-images-to-ffmpeg-using-python)

