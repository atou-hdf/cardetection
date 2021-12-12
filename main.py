from absl import app
from absl import flags
import utils.dataset as dataset
import utils.tf_record as tf_record
import detection.processing as processing
from detection.model import get_inference_model, get_training_model

FLAGS = flags.FLAGS
flags.DEFINE_string("mode", None, "Mode to use the application. Values: json, tfrecord, train, detect")
flags.DEFINE_string("path", None, "Path to image to run detection on")
flags.DEFINE_bool("export", True, 'Whether to exporte the model after training.')
flags.mark_flag_as_required("mode")

def get_predection_method(path: str):
    """Handle path parsing to decide wich function to use

    Args:
        path: path to image or video

    Returns:
        a detection function from processig module
    """
    path_list = path.split('\\')
    extension = path_list[-1].split('.')[-1]
    if extension in ('jpg', 'png'):
        return processing.show_image_predections
    elif extension == 'mp4':
        return processing.run_inference_on_video
    else:
        raise Exception("File type not supported")

def main(argv):
    del argv # Not used

    if FLAGS.mode == "json":
        print("\n== Generating json files annotations...")
        dataset.create_json_annotations()
        print("== Annotations generated")

    elif FLAGS.mode == "tfrecord":
        print("\n== Generating tensorflow records...")
        tf_record.create_tf_records()
        print("== Tensorflow records generated")
    
    elif FLAGS.mode == "train":
        print("\n== Model training process starting...")
        model = get_training_model()
        model.train()
        if FLAGS.export:
            print("\n== Exporting the model...")
            model.export()
        
    elif FLAGS.mode == "export":
        print("\n== Model exporing process starting...")
        model = get_training_model()
        model.export()

    elif FLAGS.mode == "detect":
        flags.mark_flag_as_required("path")
        print("\n== Detection starting...")
        # TODO: handle if the path is for video or image
        model = get_inference_model()
        detection = get_predection_method(FLAGS.path)
        detection(FLAGS.path, model)

    else:
        raise Exception("Mode note supported")

if __name__ == '__main__':
    app.run(main)