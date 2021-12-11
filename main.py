from absl import app
from absl import flags
import utils.dataset as dataset
import utils.tf_record as tf_record
import utils.processing as processing
from detection.model import get_inference_model, get_training_model

FLAGS = flags.FLAGS
flags.DEFINE_string("mode", None, "Mode to use the application. Values: json, tfrecord, train, detect")
flags.DEFINE_string("path", None, "Path to image to run detection on")
flags.mark_flag_as_required("mode")


def main(argv):
    del argv # Not used

    if FLAGS.mode == "json":
        print("== Generating json files annotations...")
        dataset.create_json_annotations()
        print("== Annotations generated")

    elif FLAGS.mode == "tfrecord":
        print("== Generating tensorflow records...")
        tf_record.create_tf_records()
        print("== Tensorflow records generated")
    
    elif FLAGS.mode == "train":
        print("== Model training process starting...")
        model = get_training_model()
        model.train()
        model.export() # by default the model is exported after training

    elif FLAGS.mode == "export":
        print("== Model exporing process starting...")
        model = get_training_model()
        model.export()


    elif FLAGS.mode == "detect":
        flags.mark_flag_as_required("path")
        print("== Detection starting...")
        # TODO: handle if the path is for video or image
        model = get_inference_model()
        processing.plot_detections_on_image(FLAGS.path, model)

    else:
        raise "Doesn't exist, sorry"


if __name__ == '__main__':
    app.run(main)