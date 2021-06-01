import xml.etree.ElementTree as ET
import os
import pathlib
from PIL import Image

IMG_EXT = ".jpg"

def resize_to(path: str, max_dimension: int) -> None:
    if max_dimension <= 0:
        raise ValueError("Dimension should be greater 0")

    image = Image.open(path + IMG_EXT)

    image_dimensions = image.size

    scale_factor = (image_dimensions[0] if image_dimensions[0] > image_dimensions[1] else image_dimensions[1]) / max_dimension

    new_image_dimensions = tuple([round(x / scale_factor) for x in image_dimensions])

    xml_path = "{}.xml".format(path)
    resize_label(xml_path, scale_factor)

    image.resize(new_image_dimensions).save(path + IMG_EXT)


def resize_label(path: str, scale: float):
    """
    Resize dataset label for image
    
    Parameters
    ----------
    path : str
        Path to XML

    Returns
    -------
    PIL.Image
        Resized image
    """

    tree = ET.parse(path)
    
    root = tree.getroot()

    image_size = root.find('size')
    image_size.find('width').text = str(round(int(image_size.find('width').text) / scale))
    image_size.find('height').text = str(round(int(image_size.find('height').text) / scale))

    box_size = root.find('object').find('bndbox')
    box_size.find('xmin').text = str(round(int(box_size.find('xmin').text) / scale))
    box_size.find('ymin').text = str(round(int(box_size.find('ymin').text) / scale))
    box_size.find('xmax').text = str(round(int(box_size.find('xmax').text) / scale))
    box_size.find('ymax').text = str(round(int(box_size.find('ymax').text) / scale))

    tree.write(path)


paths = [
    os.path.join(pathlib.Path(__file__).parents[1], 'Tensorflow/workspace/images/test'),
    os.path.join(pathlib.Path(__file__).parents[1], 'Tensorflow/workspace/images/train')
]

for path in paths:
    for filename in os.listdir(path):
        if IMG_EXT in filename:
            fullpath = os.path.join(path, filename.replace(IMG_EXT, ''))
            resize_to(fullpath, 1000)
