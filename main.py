from absl import app
from absl import flags
import utils.dataset as dataset
import utils.tf_record as tf_record
import detection.processing as processing
from detection.model import get_inference_model, get_training_model

FLAGS = flags.FLAGS
flags.DEFINE_string("mode", None, "Mode to use the application. Values: json, tfrecord, train, detect")
flags.DEFINE_string("path", None, "Path to image to run detection on")
flags.DEFINE_bool("export", True, 'Whether the model to be exported.')
flags.mark_flag_as_required("mode")


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
        processing.show_image_predections(FLAGS.path, model)

    else:
        raise Exception("Doesn't exist, sorry")


if __name__ == '__main__':
    app.run(main)