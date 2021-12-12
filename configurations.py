# TODO: Build a configuration parser to verify its values.

# Dataset generation configurations
DATA_PATH = "D:\\Mathilda\\24_MLProjects\\Cardetection\\nuimages-v1.0-all-samples\\samples"
DATASET_ANOTTATION_PATH = "D:\\Mathilda\\24_MLProjects\\Cardetection\\nuimages-v1.0-all-metadata"
NEW_DATASET_ANNOTATION_PATH = "D:\\Mathilda\\24_MLProjects\\Cardetection\\cardetector\\object_detection\\cardetector\\annotations"

# Map categories's names to ids
ID_MAPER = {
    "animal": 1,
    "human": 2,
    "movable_object": 3,
    "static_object": 4,
    "bicycle": 5,
    "bus": 6,
    "car": 7,
    "construction": 8,
    "emergency": 9,
    "motorcycle": 10,
    "trailer": 11,
    "truck": 12
}

# Map categories's ids to names
NAME_MAPPER = {
    1: "animal",
    2: "human",
    3: "movable_object",
    4: "static_object",
    5: "bicycle",
    6: "bus",
    7: "car",
    8: "construction",
    9: "emergency",
    10: "motorcycle",
    11: "trailer",
    12: "truck"
}

# Map dataset tokens to our new categories' names
TOKEN_MAPPER = {
    "63a94dfa99bb47529567cd90d3b58384": "animal",
    "1fa93b757fc74fb197cdd60001ad8abf": "human",
    "b1c6de4c57f14a5383d9f963fbdcb5cb": "human",
    "909f1237d34a49d6bdd27c2fe4581d79": "human",
    "403fede16c88426885dd73366f16c34a": "human",
    "e3c7da112cd9475a9a10d45015424815": "human",
    "6a5888777ca14867a8aee3fe539b56c4": "human",
    "b2d7c6c701254928a9e4d6aac9446d79": "human",
    "653f7efbb9514ce7b81d44070d6208c1": "movable_object",
    "063c5e7f638343d3a7230bc3641caf97": "movable_object",
    "d772e4bae20f493f98e15a76518b31d7": "movable_object",
    "85abebdccd4d46c7be428af5a6173947": "movable_object",
    "0a30519ee16a4619b4f4acfe2d78fb55": "static_object",
    "fc95c87b806f48f8a1faea2dcc2222a4": "bicycle",
    "003edbfb9ca849ee8a7496e9af3025d4": "bus",
    "fedb11688db84088883945752e480c2c": "bus",
    "fd69059b62a3469fbaef25340c0eab7f": "car",
    "5b3cd6f2bca64b83aa3d0008df87d0e4": "construction",
    "732cce86872640628788ff1bb81006d4": "emergency",
    "7b2ff083a64e4d53809ae5d9be563504": "emergency",
    "dfd26f200ade4d24b540184e16050022": "motorcycle",
    "90d0f6f8e7c749149b1b6c3a029841a8": "trailer",
    "6021b5187b924d64be64a702e5570edf": "truck"
}

# Model settings
LABELMAP = {
    id: {
        'id': id,
        'name': name
    }
    for name, id in ID_MAPER.items()
}

# TF RECORD SETTING
TFRECORED = {
    "train_path": r'D:\Mathilda\24_MLProjects\Cardetection\cardetector\object_detection\cardetector\annotations\train.record',
    "val_path": r'D:\Mathilda\24_MLProjects\Cardetection\cardetector\object_detection\cardetector\annotations\val.record',
    "train_shards": 20,
    "val_shards": 5
}

# Detection model configurations
INFERENCE_MODEL = {
    'config': r'D:\\Mathilda\\24_MLProjects\\Cardetection\\cardetector\\object_detection\\cardetector\\centernet\\pipeline.config',
    'checkpoints': r'D:\\Mathilda\\24_MLProjects\\Cardetection\\cardetector\\object_detection\\cardetector\\centernet\\checkpoint\\ckpt-0',
    'threshold': 0.3
}

# Training model configurations
TRAINING_MODEL = {
    'pipeline_config_path': r'D:\\Mathilda\\24_MLProjects\\Cardetection\\cardetector\\object_detection\\cardetector\\model\\pipeline.config',
    'model_dir': r'D:\\Mathilda\\24_MLProjects\\Cardetection\\cardetector\\object_detection\\cardetector\\model'
}

# Exporting model configurations
EXPORTING_MODEL = {
    'config': TRAINING_MODEL['pipeline_config_path'],
    'checkpoints_dir': TRAINING_MODEL['model_dir'],
    'output_dir': r'D:\\Mathilda\\24_MLProjects\\Cardetection\\cardetector\\object_detection\\cardetector\\centernet'
}