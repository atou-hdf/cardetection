"""For image and video processing before using the model"""
import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from object_detection.utils import visualization_utils as viz_utils
import configurations as cfg
matplotlib.use('Qt5Agg') # for programme called from console

def get_image_numpy(image_path: str):
    """Import image in path, and return it as a numpy array

    Args:
      imag_path: str path to image
    
    Returns:
      image as a numpy array
    """
    # Get image : as numpoy array, and as an encoded
    with tf.io.gfile.GFile(image_path, 'rb') as fid:
        # Image encoded : image as bytes for tfRecord
        try:
            image_encoded = fid.read()
            img = tf.io.decode_image(image_encoded, 3)
            return np.asarray(img)

        except Exception as tf_e:
            raise Exception(f'Tensorflow exception when reading image {image_path}\n{tf_e}')

def get_image_tensor(image_path: str):
  """Import image in path, and return it as a tensor

    Args:
      imag_path: str path to image
    
    Returns:
      image as a tensor
    """
  image_np = get_image_numpy(image_path)
  image_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
  return image_tensor

def _plot_detections(image_np,
                    boxes,
                    classes,
                    scores,
                    category_index,
                    min_score_thresh=0.5,
                    figsize=(12, 16),
                    image_name=None):
    """Wrapper function to visualize detections.

    Args:
        image_np: uint8 numpy array with shape (img_height, img_width, 3)
        boxes: a numpy array of shape [N, 4]
        classes: a numpy array of shape [N]. Note that class indices are 1-based,
          and match the keys in the label map.
        scores: a numpy array of shape [N] or None.  If scores=None, then
          this function assumes that the boxes to be plotted are groundtruth
          boxes and plot all boxes as black with no classes or scores.
        category_index: a dict containing category dictionaries (each holding
          category index `id` and category name `name`) keyed by category indices.
        figsize: size for the figure.
        image_name: a name for the image file.
    """
    image_np_with_annotations = image_np.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_annotations,
          boxes,
          classes,
          scores,
          category_index,
          use_normalized_coordinates=True,
          line_thickness=1,
          min_score_thresh=min_score_thresh)
    if image_name:
        plt.imsave(image_name, image_np_with_annotations)
    else:
        plt.imshow(image_np_with_annotations)
        plt.show()

def plot_detections_on_image(path: str, model):
  predections = model.predict(path)
  thresh = model.threshold

  # offset for background class, aka 0
  label_id_offset = 1

  # plot predections
  _plot_detections(predections['image_np'],
                  predections['detections']['detection_boxes'],
                  predections['detections']['detection_classes'] + label_id_offset,
                  predections['detections']['detection_scores'],
                  cfg.LABELMAP,
                  thresh,
                  image_name=None,
                )