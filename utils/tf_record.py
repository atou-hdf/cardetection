import os
import sys
# from PIL import Image
import json
from skimage import io
from PIL import Image
from six import BytesIO
import tensorflow as tf
import contextlib2
from object_detection.dataset_tools import tf_record_creation_util
from object_detection.utils import dataset_util
from tensorflow._api.v2 import train
import configurations as cfg


def progress_bar(length, *, prefix="", message="", file=sys.stdout):
    if message == "":
        message = f"Done processing {prefix}"
    
    def progress_bar_at_index(index):
        if index < length - 1:
            percentage = int((index/length) * 30)
            file.write(f"{prefix}: [{'#'*percentage}{'.'*(30-percentage)}] {index + 1}/{length}\r")
            file.flush()
        elif index == length - 1:
            file.write(f"{message}: {index + 1}/{length}{' '*100}\r")
            file.flush()
        else:
            raise RuntimeError("Index past max length")

    return progress_bar_at_index

def get_annotations(file_path):
    """Get annotations list from json file"""

    annotations = []
    with open(file_path) as file:
        annotations = json.load(file)
    return annotations


def get_image_dict(image: dict, objects) -> dict:
    """Build a dictionary with image metadat ready for tfredcord"""

    width = image['width']
    height = image['height']
    filename = image['file_name']

    path = os.path.join(cfg.DATA_PATH, filename)

    with tf.io.gfile.GFile(path, 'rb') as fid:
        encoded = fid.read()

    # encoded = io.imread(path)
    # encoded = Image.open(path)
    # encoded = None
    iformat = b'jpeg' # hard coded for now

    idict = {
        "height": height,
        "width": width,
        "filename": filename,
        "encoded": encoded,
        "format": iformat,
        "objects": objects,
    }
    return idict

def create_tf_example(image_dict):
    """Generate a tf Example from image dictionnary"""
    width = image_dict["width"]
    height = image_dict["height"]

    xmins = [iobject['bbox'][0]/width for iobject in image_dict["objects"]]
    xmaxs = [iobject['bbox'][2]/width for iobject in image_dict["objects"]]
    ymins = [iobject['bbox'][1]/height for iobject in image_dict["objects"]]
    ymaxs = [iobject['bbox'][3]/height for iobject in image_dict["objects"]]
    classes = [iobject['category_id'] for iobject in image_dict["objects"]]
    classes_text = [cfg.BYTE_NAME_MAPPER[cls] for cls in classes]

    tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(image_dict["height"]),
      'image/width': dataset_util.int64_feature(image_dict["width"]),
      'image/filename': dataset_util.bytes_feature(image_dict["filename"].encode('utf-8')),
      'image/source_id': dataset_util.bytes_feature(image_dict["filename"].encode('utf-8')),
      'image/encoded': dataset_util.bytes_feature(image_dict["encoded"]),
      'image/format': dataset_util.bytes_feature(image_dict["format"]),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def write_tf_record(image_list: list, output_file: str, shards_number: int):
    """Write tf record files to disk"""

    dir_path = output_file.split('\\')
    dir_path = '\\'.join(dir_path[:-1])
    assert os.path.exists(dir_path), f"Directory {dir_path} not found. Check the variable TF_RECORD_OUTPUT_FILES_SHARDS in configs!"

    with contextlib2.ExitStack() as tf_record_close_stack:
        output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(tf_record_close_stack,
                                                                                 output_file,
                                                                                 shards_number)
        for index, image_dict in enumerate(image_list):
            tf_example = create_tf_example(image_dict)
            output_shard_index = index % shards_number
            output_tfrecords[output_shard_index].write(tf_example.SerializeToString())

def create_tf_record():
    # Training set
    print("Creating training records...")
    train_path = os.path.join(cfg.NEW_DATASET_ANNOTATION_PATH, "train.json")
    train_annotations  = get_annotations(train_path)
    if train_annotations:
        pb_train = progress_bar(len(train_annotations['images']),
                                    prefix="Creating training tfrecord",
                                    message="Training tf record creaated")

        with contextlib2.ExitStack() as tf_record_close_stack:
            output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(tf_record_close_stack,
                                                                                    cfg.TFRECORED["train_path"],
                                                                                    cfg.TFRECORED["train_shards"])
            for index, image in enumerate(train_annotations['images'][:10000]):
                objects = [a for a in train_annotations["annotations"] if a['image_id'] == image['id']]
                idict = get_image_dict(image, objects)
                tf_example = create_tf_example(idict)
                output_shard_index = index % cfg.TFRECORED["train_shards"]
                output_tfrecords[output_shard_index].write(tf_example.SerializeToString())
                pb_train(index)
    else:
        raise Exception("Problem importing training annotations!")

    # Validation set
    print("Creating validation records...")
    val_path = os.path.join(cfg.NEW_DATASET_ANNOTATION_PATH, "val.json")
    val_annotations  = get_annotations(val_path)
    if val_annotations:
        pb_val = progress_bar(len(val_annotations['images']),
                                    prefix="Creating validation tfrecord",
                                    message="Validation tf record creaated")

        with contextlib2.ExitStack() as tf_record_close_stack:
            output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(tf_record_close_stack,
                                                                                cfg.TFRECORED["val_path"],
                                                                                cfg.TFRECORED["val_shards"])
            for index, image in enumerate(val_annotations['images'][:2000]):
                objects = [a for a in val_annotations["annotations"] if a['image_id'] == image['id']]
                idict = get_image_dict(image, objects)
                tf_example = create_tf_example(idict)
                output_shard_index = index % cfg.TFRECORED["val_shards"]
                output_tfrecords[output_shard_index].write(tf_example.SerializeToString())
                pb_val(index)
    else:
        raise Exception("Problem importing validation annotations!")

if __name__ == "__main__":
    create_tf_record()