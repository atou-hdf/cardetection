"""For image and video processing before using the model"""
import tensorflow as tf
import numpy as np
import cv2
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

def _draw_predections_on_image(path: str, model):
  # Map classe id to a specific rgb color
  RGB_MAPPER = {
    1: (85, 239, 196),
    2: (116, 185, 255),
    3: (162, 155, 254),
    4: (250, 177, 160),
    5: (255, 118, 117),
    6: (225, 112, 85),
    7: (253, 203, 110),
    8: (237, 76, 103),
    9: (27, 20, 100),
    10: (0, 98, 102),
    11: (111, 30, 81),
    12: (18, 137, 167)
  }
  predections = model.predict(path)
  classes = predections['detections']['detection_classes']
  scores =  predections['detections']['detection_scores']
  boxes = predections['detections']['detection_boxes']
  image = cv2.imread(path,cv2.IMREAD_COLOR)
  label_id_offset = 1

  width, height, _ = image.shape
  for idx, cls in enumerate(classes):
    if scores[idx] > model.threshold:
      print("yep at least somthing")
      color = RGB_MAPPER[cls + label_id_offset]
      object_cls = cfg.NAME_MAPPER[cls + label_id_offset]

      # Two long strings to two small ones
      if object_cls == 'static_object':
        object_cls = 'startic'
      elif object_cls == 'movable_object':
        object_cls = 'movable'

      box = boxes[idx]
      print(f"box: {boxes[idx]}, type: {type(boxes[idx])}")

      y2 = int(box[0].item() * width)
      x2 = int(box[1].item() * height)
      y1 = int(box[2].item() * width)
      x1 = int(box[3].item() * height)

      print(f"{x1}, {x2}, {y1}, {y2}")

      font = cv2.FONT_HERSHEY_DUPLEX
      image = cv2.rectangle(image, (x1,y2), (x2,y2-30), color, cv2.FILLED)
      image = cv2.putText(image, f'{object_cls}',(x2 + 5, y2 - 10), font, 0.50, (0,0,0), 1, cv2.LINE_AA)
      image = cv2.rectangle(image, (x1,y1), (x2,y2), color, 1)

  cv2.imshow("image", image)
  cv2.waitKey(0)
  cv2.destroyWindow("image")
  return image