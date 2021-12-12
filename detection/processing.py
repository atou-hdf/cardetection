r"""For image and video processing"""

import tensorflow as tf
import numpy as np
import cv2
from tensorflow.python.ops.gen_math_ops import imag
import configurations as cfg

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

def _draw_predections_on_image(path: str, model):
  """Draw predections on image.

  Args:
    path: path to image
    model: inference model

  Returns:
    image with predections drawn on it.
  """
  # Map classe id to a specific bgr color
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
  
  # getting predections
  predections = model.predict(path)
  classes = predections['detections']['detection_classes']
  scores =  predections['detections']['detection_scores']
  boxes = predections['detections']['detection_boxes']
  
  # getting image properties
  image = cv2.imread(path,cv2.IMREAD_COLOR)
  height, width, _ = image.shape
  font = cv2.FONT_HERSHEY_DUPLEX
  label_id_offset = 1 # 0 for background

  # drwing predections on image
  for idx, cls in enumerate(classes):
    if scores[idx] > model.threshold:
      color = RGB_MAPPER[cls + label_id_offset]
      object_cls = cfg.NAME_MAPPER[cls + label_id_offset]
      score = round(scores[idx].item() * 100)

      # Two long strings to two small ones
      if object_cls == 'static_object':
        object_cls = 'startic'
      elif object_cls == 'movable_object':
        object_cls = 'movable'

      # Bounding box
      box = boxes[idx]
      y1 = int(box[0].item() * height)
      x1 = int(box[1].item() * width)
      y2 = int(box[2].item() * height)
      x2 = int(box[3].item() * width)

      # Text, ex: human 70%
      text = object_cls + " " + str(score) + "%"
      (text_w, text_h), _ = cv2.getTextSize(text, font, 0.5, 1)
      
      # Drawing bbox with text on image
      image = cv2.rectangle(image, (x1,y1), (x1 + 8 + text_w,y1 - 8 - text_h), color, cv2.FILLED)
      image = cv2.putText(image, text,(x1 + 4, y1 - 4), font, 0.50, (0,0,0), 1, cv2.LINE_AA)
      image = cv2.rectangle(image, (x1,y1), (x2,y2), color, 1)

  return image

def _mouse_callback(event, x, y, flags, param):
  """Check if 'save' is clicked.
  And Change param['save'] to True if so.
  """
  # Assert param is a dict with point1, point2, point3
  REQUIRED_KEYS = ('point1', 'point2', 'save')
  assert isinstance(param, dict), "param is not a dictionnary"
  for k in REQUIRED_KEYS:
    assert k in param.keys(), f'{k} is missing'

  x1, y1 = param['point1']
  x2, y2 = param['point2']

  # Assert cordinates coherence
  assert x2 > x1, "Rectangle cordinates are not coherent, x1 > x2"
  assert y2 > y1, "Rectangle cordinates are not coherent, y1 > y2"
  
  # Change param['save'] to True when save is clicked
  if event == cv2.EVENT_LBUTTONUP:
    if x2 > x > x1 and y2 > y > y1:
      param['save'] = True

def show_image_predections(path: str, model):
  """Show predections on image.

  Args:
    path: path to image
    model: inference model

  Returns:
    boolean indicating if the image was saved.
  """
  image_saved = False
  image = _draw_predections_on_image(path, model)
  show_image = image.copy()

  # Image properties
  font = cv2.FONT_HERSHEY_DUPLEX
  width = show_image.shape[1]

  # Adding save button to image
  save = "Save"
  (save_w, save_h), _ = cv2.getTextSize(save, font, 0.4, 1)
  point1 = (width - save_w - 25, 6)
  point2 = (width - 5, save_h + 20)
  show_image = cv2.rectangle(show_image, point1, point2, (49, 48, 214), cv2.FILLED)
  show_image = cv2.putText(show_image, save, (width - save_w - 15, 20), font, 0.4, (255, 255, 255), 1, cv2.LINE_AA) # (214, 48, 49)

  param = {
    'point1': point1,
    'point2': point2,
    'save': False
  }
  
  cv2.namedWindow('image')
  cv2.setMouseCallback('image',_mouse_callback, param=param)
  cv2.imshow('image', show_image)

  while True:
    # Break if window closed with X
    try:
      cv2.getWindowProperty('image', 0)
    except:
      break

    # Break if q is presed, or 'save' button is clicked
    c = cv2.waitKey(33)
    if c == ord('q') or param['save']:
      break
    if c == ord('s'):
      param['save'] = True
      break

  cv2.destroyAllWindows()

  # Save image if 'save' is clicked
  if param['save']:
    # Getting image path
    new_path = path.split('\\')
    name, extension = new_path[-1].split('.')
    img_name = name + f'_pred_{model.threshold}' + '.' + extension
    new_path = new_path[:-1]
    new_path.append(img_name)
    new_path = '\\'.join(new_path)

    # Saving image
    cv2.imwrite(new_path, image)
    image_saved = True
    print(f"INFO: image saved at {new_path}")

  return image_saved
