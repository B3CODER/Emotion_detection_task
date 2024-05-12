import json
import os
import shutil

# Load JSON data
with open('annotations.json', 'r') as f:
    data = json.load(f)

# Iterate through JSON data
for item in data['train']:
    image_path = item['path']
    labels = item['labels']
    
    # Check if image path exists in your dataset
    if os.path.exists(image_path):
        # Create directories for labels if they don't exist
        for label in labels:
            label_dir = os.path.join('labeled_images', label)
            os.makedirs(label_dir, exist_ok=True)
        
        # Copy image to each label directory
        for label in labels:
            label_dir = os.path.join('labeled_images', label)
            shutil.copy(image_path, label_dir)
        
        print(f"Image '{image_path}' saved with labels: {labels}")
    else:
        print(f"Image Path '{image_path}' not found. Skipping...")
