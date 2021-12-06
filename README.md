# Car detection project

## Abstract
In this project I’ll be building an object detector to detect cars, pedestrians, buses and other objects. The algorithm will be able to detect 12 objects: Animal, human, movable_object, static_object, bicycle, bus, car, construction, emergency, motorcycle, trailer, and truck.

The project is based on Centernet architecture, tensorflow object detection api and nuscenes dataset.

## Resources
* [CenterNet Paper](https://arxiv.org/abs/1904.07850)
* [Nuscenes dataset](https://www.nuscenes.org/download)
* [Tensorflow object detection api](https://github.com/tensorflow/models/tree/master/research/object_detection)

## Quick Description
1. the configuration.py for settings.
2. The dataset.py module is used to build a json file with annotations and images' metadata.
3. The tf_record.py module will use these json files to generate a set of tfrecord files.

## To do
* Running detection on image.
* Running detection on video.
* Controlling the model (for training or for inference).
