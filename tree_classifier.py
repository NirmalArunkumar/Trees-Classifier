import json
import xml.etree.ElementTree as ET
from pathlib import Path
from fastai.vision.all import *
import numpy as np
from collections import defaultdict


# Define the paths to the dataset and annotations
dataset_path = Path('C:/Users/nirma/OneDrive/Documents/studies/CarbonArray')

# Ensure paths are correctly set
print("Starting script execution...")

# Ensure paths are correctly set
images_path = dataset_path / "Images"
annotations_path = dataset_path / "Annotations"

print(f"Images path: {images_path}")
print(f"Annotations path: {annotations_path}")

# Get list of image files
image_files = list(images_path.glob("*.jpg"))
print(f"Image files: {image_files}")

# Function to convert XML annotations to COCO format
def xml_to_coco(image_files, annotations_path):
    coco_output = {
        "images": [],
        "annotations": [],
        "categories": [],
    }
    category_set = set()
    annotation_id = 1

    for image_id, image_path in enumerate(image_files, 1):
        image = {
            "id": image_id,
            "file_name": image_path.name,
            "height": 1536,  # Should be set according to your dataset
            "width": 1536,   # Should be set according to your dataset
        }

        annotation_file = annotations_path / (image_path.stem + '.xml')
        tree = ET.parse(annotation_file)
        root = tree.getroot()

        size = root.find('size')
        image["width"] = int(size.find('width').text)
        image["height"] = int(size.find('height').text)

        coco_output["images"].append(image)

        for obj in root.findall('object'):
            category = obj.find('tree').text
            category_set.add(category)
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            width = xmax - xmin
            height = ymax - ymin
            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category,
                "bbox": [xmin, ymin, width, height],
                "area": width * height,
                "segmentation": [],
                "iscrowd": 0,
            }
            coco_output["annotations"].append(annotation)
            annotation_id += 1

    for category_id, category_name in enumerate(sorted(category_set), 1):
        category = {
            "id": category_id,
            "name": category_name,
        }
        coco_output["categories"].append(category)

    return coco_output

# Convert XML annotations to COCO format JSON
coco_output = xml_to_coco(image_files, annotations_path)
with open("annotations.json", "w") as f:
    json.dump(coco_output, f)

print("Annotations converted to COCO format and saved as annotations.json")

# Load COCO JSON annotations
def get_coco_annotations(json_path):
    with open(json_path) as f:
        data = json.load(f)
    return data

# Define a function to get the items from the JSON
def get_annotations_from_json(img_path, annotations, categories):
    img_id = next(img["id"] for img in annotations["images"] if img["file_name"] == img_path.name)
    annos = [anno for anno in annotations["annotations"] if anno["image_id"] == img_id]
    bboxes = [anno["bbox"] for anno in annos]
    labels = [categories[anno["category_id"]-1] for anno in annos]  # -1 because categories are 1-indexed
    return TensorBBox(bboxes), TensorMultiCategory(labels)

# Load annotations and categories
annotations = get_coco_annotations("annotations.json")
categories = [cat["name"] for cat in annotations["categories"]]

# Define bbox_label_transform function
def bbox_label_transform(o):
    bboxes, labels = get_annotations_from_json(o, annotations, categories)
    bboxes, labels = clip_remove_empty(bboxes, labels)
    return bboxes, labels

# Define label function for from_path_func
def label_func(fn):
    bboxes, labels = bbox_label_transform(fn)
    return bboxes, labels

# Create DataLoaders using from_path_func
print("Creating DataLoaders with from_path_func...")
try:
    dls = ImageDataLoaders.from_path_func(
        images_path,
        get_image_files(images_path),
        label_func,
        valid_pct=0.2,
        item_tfms=Resize(128),
        batch_tfms=Normalize.from_stats(*imagenet_stats)
    )
    print("DataLoaders created successfully")
except Exception as e:
    print(f"Error creating dataloaders: {e}")

# Further check if dls is defined
if 'dls' in locals():
    print("DataLoaders are available.")
else:
    print("DataLoaders creation failed.")

