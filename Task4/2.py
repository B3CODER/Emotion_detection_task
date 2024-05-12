import json
import os
import shutil
import cv2

# Load JSON data
with open('annotations.json', 'r') as f:
    data = json.load(f)

# Function to draw bounding boxes on the image
def draw_bounding_boxes(image, bbox):
    if isinstance(bbox, list) and len(bbox) == 4:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    else:
        print("Invalid bounding box format.")

# Iterate through JSON data
for item in data['train']:
    image_path = item['path']
    labels = item['labels']
    bbox = item['bbox']
    
    # Check if image path exists in your dataset
    if os.path.exists(image_path):
        # Read image
        image = cv2.imread(image_path)
        
        # Draw bounding boxes on image
        draw_bounding_boxes(image, bbox)
        
        # Save image with labels and bounding boxes
        image_name = os.path.basename(image_path)
        for label in labels:
            label_dir = os.path.join('labeled_images', label)
            os.makedirs(label_dir, exist_ok=True)
            cv2.imwrite(os.path.join(label_dir, image_name), image)
        
        print(f"Image '{image_path}' saved with labels: {labels} and bounding boxes.")

