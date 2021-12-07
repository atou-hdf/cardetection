import os
import json
import configurations as cfg


def is_mapped(token):
    """Verrify if we want to map this token to a category"""

    return token in cfg.TOKEN_MAPPER.keys()

def map_token_name(token):
    """Map category token to a name"""

    assert is_mapped(token), "Token is not mappable !"
    return cfg.TOKEN_MAPPER[token]

def map_token_id(token):
    """Map category token to an id"""

    assert is_mapped(token), "Token is not mappable !"
    return cfg.ID_MAPER[cfg.TOKEN_MAPPER[token]]

def process_categories(categories: list) -> list:
    """Process categories from annotations to customized categories"""

    proccessed_cat = []
    pro_cat_ids = set()
    for category in categories:
        cat_token = category['token']
        if is_mapped(cat_token):
            cat_id = map_token_id(cat_token)
            cat_name =  map_token_name(cat_token)

            # Verify if we alerady added this category to our list
            if cat_id not in pro_cat_ids:
                pro_cat_ids.add(cat_id)
                proccessed_cat.append(
                    {
                        "id": cat_id,
                        "name": cat_name
                    }
                )
    return proccessed_cat

def process_annotations(annotations: list) -> list:
    """Adapt data annotations to our architecture"""

    processed_annot = []
    for annot in annotations:
        category_token = annot['category_token']
        if is_mapped(category_token):
            processed_annot.append(
                {
                    "id": annot['token'],
                    "image_id": annot['sample_data_token'],
                    "category_id": map_token_id(category_token),
                    "bbox": annot['bbox']
                }
            )
    return processed_annot

def process_images(images: list) -> list:
    """Adapt image metadata to our architecture"""

    processed_images = []
    for image in images:
        if image['is_key_frame']: # Only key frames have annotations
            image_name = image['filename']
            image_name = image_name.replace('samples/', "") # remove samples from path
            image_name = image_name.replace('/', '\\') # match windows style
            processed_images.append(
                {
                    "id": image['token'],
                    "width": image['width'],
                    "height": image['height'],
                    "file_name": image_name
                }
            )
    return processed_images

def build_annotations(folder):
    """Build annotations from a folder"""

    ANOTTATION_PATH = os.path.join(cfg.DATASET_ANOTTATION_PATH, folder)
    annotations_dict = {
        "images": [],
        "annotations": [],
        "categories": [],
        "license": {
            "name": "nuImages",
            "url": "https://www.nuscenes.org/nuimages"
        }
    }
    for file_name in os.listdir(ANOTTATION_PATH):
        file_path = os.path.join(ANOTTATION_PATH, file_name)

        # Build categories
        if file_name == "category.json":
            with open(file_path) as file:
                categories = json.load(file)
                annotations_dict["categories"] = process_categories(categories)

        # Build annotations
        elif file_name == "object_ann.json":
            with open(file_path) as file:
                annotations = json.load(file)
                annotations_dict["annotations"] = process_annotations(annotations)

        # Build images
        elif file_name == "sample_data.json":
            with open(file_path) as file:
                images = json.load(file)
                annotations_dict["images"] = process_images(images)
            
    return annotations_dict

def create_json_annotations():
    # Dump training data into a json file
    print("Generating training annotations...")
    trai_annotations_path = os.path.join(cfg.NEW_DATASET_ANNOTATION_PATH, "train.json")
    trai_annotations = build_annotations("v1.0-train")
    trai_json_object = json.dumps(trai_annotations, indent = 4)
    with open(trai_annotations_path, 'w') as outfile:
        outfile.write(trai_json_object)
    print("Training annotations successfully generated")

    # Dump validation data into a json file
    print("Generating validation annotations...")
    val_annotations_path = os.path.join(cfg.NEW_DATASET_ANNOTATION_PATH, "val.json")
    val_annotations = build_annotations("v1.0-val")
    val_json_object = json.dumps(val_annotations, indent = 4)
    with open(val_annotations_path, 'w') as outfile:
        outfile.write(val_json_object)
    print("Validation annotations successfully generated")