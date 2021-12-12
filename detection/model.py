import tensorflow as tf
import numpy as np
from absl import flags
from absl import app
from object_detection.builders import model_builder
from object_detection.utils import config_util


import configurations as cfg
import detection.processing as processing

_INFERENCE_MODEL = None
_TRAINING_MODEL = None

class Model:
    def __init__(self, is_training):
        # Verify if is_training is a boolean
        assert type(is_training) == bool, "is_training should be boolean"
        self._is_training = is_training

        # Training Model
        if is_training:
            self._pipeline_cfg  = cfg.TRAINING_MODEL['pipeline_config_path']
            self._model_dir = cfg.TRAINING_MODEL['model_dir']
       
        # Inference Model
        else:
            configs = config_util.get_configs_from_pipeline_file(cfg.INFERENCE_MODEL['config'])
            self._cfg = configs['model']
            self._ckpt = cfg.INFERENCE_MODEL['checkpoints']
            self._threshold = cfg.INFERENCE_MODEL['threshold']
            # Build model
            self._detection = model_builder.build(model_config=self.cfg, is_training=False)
            # Restore checkpoint
            ckpt = tf.compat.v2.train.Checkpoint(model = self._detection)
            ckpt.restore(self.ckpt).expect_partial()

    @property
    def is_training(self):
        return self._is_training

    @is_training.setter
    def is_training(self, is_training):
        # Porhibite changing model's training state 
        raise Exception("Model state can't be changed.")

    @property
    def cfg(self):
        return self._cfg
    
    @property
    def ckpt(self):
        return self._ckpt

    @property
    def threshold(self):
        return self._threshold
    
    @property
    def detection(self):
        return self._detection
    
    @property
    def pipeline_cfg(self):
        return self._pipeline_cfg

    @property
    def model_dir(self):
        return self._model_dir

    def train(self):
        import object_detection.model_main_tf2 as model_main_tf2 # imported here to avoid conflicts with exporter_main_v2
        assert self.is_training, Exception('Model is not inialized to be trained !')

        # model_main_tf2 exits with SystemExit exception
        try:
            model_main_tf2.FLAGS.model_dir = self.model_dir
            model_main_tf2.FLAGS.pipeline_config_path = self.pipeline_cfg
            app.run(model_main_tf2.main)
        except SystemExit:
            del model_main_tf2.FLAGS.pipeline_config_path # To avoid conflicts with exporter_main_v2
            print("=== Training done ! Good luck with your predections ;) ===")
    
    def export(self):
        import object_detection.exporter_main_v2 as exporter_main_v2 # imported here to avoid conflicts with model_main_tf2
        assert self.is_training, Exception('Model is not inialized to be exported !')

        # exporter_main_v2 exits with SystemExit exception
        try:
            exporter_main_v2.FLAGS.pipeline_config_path = cfg.EXPORTING_MODEL['config']
            exporter_main_v2.FLAGS.trained_checkpoint_dir = cfg.EXPORTING_MODEL['checkpoints_dir']
            exporter_main_v2.FLAGS.output_directory = cfg.EXPORTING_MODEL['output_dir']
            app.run(exporter_main_v2.main)
        except SystemExit:
            del exporter_main_v2.FLAGS.pipeline_config_path # To avoid conflicts with model_main_tf2
            print("=== Model exported, all the best ===")

    def _get_detection(self):
        """Get a tf.function for detection."""

        @tf.function
        def detect_fn(image):
            """Detect objects in image."""

            image, shapes = self.detection.preprocess(image)
            prediction_dict = self.detection.predict(image, shapes)
            detections = self.detection.postprocess(prediction_dict, shapes)

            return detections, prediction_dict, tf.reshape(shapes, [-1])
        return detect_fn
        

    def predict(self, path):
        """Get image with detections dectionary
        
        Args:
            path: path to image

        Returns:
            dict with two keys:
                image_np: image as numpy array
                detections: objects in the image
        """
        assert not self.is_training, Exception('Model is not inialized to run inference !')

        image_tensor = processing.get_image_tensor(path)
        image_numpy = processing.get_image_numpy(path)

        detect_fn = self._get_detection()
        detections, *_ = detect_fn(image_tensor)

        num_detections = int(detections.pop('num_detections'))
        detections = {
            key: value[0, :num_detections].numpy()
            for key, value in detections.items()
        }
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        image_data = {
            "image_np" : image_numpy,
            "detections" : detections
        }
        return image_data

def get_inference_model():
    """Get inference model
    
    Returns:
        model ready to run inference
    """
    # Make sure that we have only one instance of inference model in our program
    global _INFERENCE_MODEL
    if not _INFERENCE_MODEL:
        _INFERENCE_MODEL = Model(is_training=False)
    return _INFERENCE_MODEL

def get_training_model():
    """Get training model
    
    Returns:
        model ready to be trained
    """
    # Make sure that we have only one instance of training model in our program
    global _TRAINING_MODEL
    if not _TRAINING_MODEL:
        _TRAINING_MODEL = Model(is_training=True)
    return _TRAINING_MODEL