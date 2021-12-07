"""For image and video processing before using the model"""
import tensorflow as tf
import numpy as np
from PIL import Image
from six import BytesIO


def load_image_into_numpy_array(path):
  """Load an image from file into a numpy array.

  Puts image into numpy array to feed into tensorflow graph.

  Args:
    path: the file path to the image

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
  """
  img_data = tf.io.gfile.GFile(path, 'rb').read()
  image = Image.open(BytesIO(img_data))
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def get_image_to_numpy_array(image_path: str):

    # Get image : as numpoy array, and as an encoded
    with tf.io.gfile.GFile(image_path, 'rb') as fid:
        # Image encoded : image as bytes for tfRecord
        try:
            image_encoded = fid.read()
            img = tf.io.decode_image(image_encoded, 3)
            return np.asarray(img)

        except Exception as tf_e:
            raise Exception(f'Tensorflow exception when reading image {image_path}\n{tf_e}')