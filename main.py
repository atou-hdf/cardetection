from absl import app
from absl import flags
import utils.dataset as dataset
import utils.tf_record as tf_record

FLAGS = flags.FLAGS
flags.DEFINE_string("mode", None, "Mode of application. Arguments: jsonannot\n tfrecords")
flags.mark_flag_as_required("mode")


def main(argv):
    del argv # Not used

    if FLAGS.mode == "jsonannot":
        print("== Generating json files annotations...")
        dataset.create_json_annotations()
        print("== Annotations generated")

    elif FLAGS.mode == "tfrecords":
        print("== Generating tensorflow records...")
        tf_record.create_tf_records()
        print("== Tensorflow records generated")

if __name__ == '__main__':
    app.run(main)