import xml.etree.ElementTree as ET
from pathlib import Path
from fastai.vision.all import *
from yolov5 import YOLOv5

# Define the path to the dataset and annotations
dataset_path = Path('/path/to/your/forest/dataset')
annotations_path = dataset_path / 'annotations.xml'

# Function to get annotations
def get_annotations(image_path, annotations_path):
    # Load annotations
    tree = ET.parse(annotations_path)
    root = tree.getroot()

    # Create a mapping from image file name to tree label
    image_id_map = {img.find('filename').text: img.find('object').find('tree').text for img in root.findall('object')}
    
    # Create a mapping from tree label to tree label (you can customize this if needed)
    category_id_map = {cat.find('object').find('tree').text: cat.find('object').find('tree').text for cat in root.findall('object')}
    
    # Get the image file name
    image_file_name = image_path.name
    
    # Get the tree label from the file name
    tree_label = image_id_map.get(image_file_name)
    
    if tree_label is None:
        return []

    # Find annotations for the tree label
    tree_annotations = [ann for ann in root.findall('object') if ann.find('tree').text == tree_label]
    
    # Convert the annotations to the desired format
    bboxes_and_labels = []
    for ann in tree_annotations:
        bbox = ann.find('bndbox')
        # Convert bbox from [xmin, ymin, xmax, ymax] to [x_min, y_min, x_max, y_max]
        bbox = [int(bbox.find(coord).text) for coord in ['xmin', 'ymin', 'xmax', 'ymax']]
        label = category_id_map[ann.find('tree').text]
        bboxes_and_labels.append((bbox, label))
    
    return bboxes_and_labels

# Load your dataset using Fastai
data = DataBlock(
    blocks=(ImageBlock, BBoxBlock, BBoxLblBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(seed=42),
    get_y=[lambda o: get_annotations(o, annotations_path)],
    item_tfms=Resize(460),
    batch_tfms=aug_transforms(size=224, min_scale=0.75)
)

dls = data.dataloaders(dataset_path / 'images')

# Load the YOLOv5 model
model = YOLOv5('yolov5s.pt')  # Load pre-trained YOLOv5 small model

# Train the model
learn = Learner(dls, model, metrics=[error_rate])
learn.fine_tune(3)

# Save the trained model
learn.save('forest_tree_detector')

# Inference on test images
test_files = get_image_files(dataset_path / 'test_images')
test_dl = dls.test_dl(test_files)

# Run predictions
results = learn.get_preds(dl=test_dl)

# Post-process results to count trees per species and size
species_size_counts = defaultdict(lambda: defaultdict(int))

for result in results:
    for bbox, label in zip(result[0], result[1]):
        species = label  # Extract species from the label
        size = get_tree_size_from_bbox(bbox)  # Define this function based on bbox dimensions
        species_size_counts[species][size] += 1

# Format the output
output = []
for species, size_counts in species_size_counts.items():
    sizes_output = ', '.join(f'{count} {size}' for size, count in size_counts.items())
    total_count = sum(size_counts.values())
    output.append(f'{species} trees - {total_count} ({sizes_output})')

# Print the formatted output
for line in output:
    print(line)
