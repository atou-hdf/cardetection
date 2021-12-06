# Car detection project

## Abstract
In this project I’ll be building an object detector to detect cars, pedestrians, buses and other objects. The algorithm will be able to detect 12 objects: Animal, human, movable_object, static_object, bicycle, bus, car, construction, emergency, motorcycle, trailer, and truck.

The project is based on Centernet architecture, tensorflow object detection api and nuscenes dataset.

## Resources
* [CenterNet Paper](https://arxiv.org/abs/1904.07850)
* [Nuscenes dataset](https://www.nuscenes.org/nuimages)
* [Tensorflow object detection api](https://github.com/tensorflow/models/tree/master/research/object_detection)

## Quick Description
1. the configuration.py for settings.
2. The dataset.py module is used to build a json file with annotations and images' metadata.
3. The tf_record.py module will use these json files to generate a set of tfrecord files.

## Quick Start
1. Install Python3.
2. Install Tensorflow 2.
3. Install [Tensorflow object detection api](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md).
4. Download the [nuImages](https://www.nuscenes.org/download) dataset from nuscenes.
5. Change path settings in configuration.py to match your preferences.
6. Generte json file for training and validation data.
7. Generate tfrecord files.
8. Train the model.

## To do
* Running detection on image.
* Running detection on video.
* Controlling the model (for training or for inference).
