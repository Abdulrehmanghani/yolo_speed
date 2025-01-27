import os
import cv2
import json
from shutil import copy2
from ultralytics import YOLO

# Load YOLOv8 model for person detection
person_model = YOLO("yolov8l.pt")  # YOLO model for person detection

# Input and output directories
input_dir = "frames"  # Directory where original images are stored
output_dir = "output"  # Directory where results will be saved

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Function to convert YOLO bbox to LabelMe polygon (rectangle)
def convert_to_labelme_bbox(yolo_bbox):
    xmin, ymin, xmax, ymax = map(float, yolo_bbox)
    return [[xmin, ymin], [xmax, ymax]]

# Loop through all images in the input directory
for image_filename in os.listdir(input_dir):
    if image_filename.endswith(".jpg") or image_filename.endswith(".png"):
        # Read the image
        image_path = os.path.join(input_dir, image_filename)
        image = cv2.imread(image_path)
        
        # Perform inference to detect persons (class 0)
        results = person_model(image, classes=[0], conf=0.6, verbose=False)

        # Get person bounding boxes (class 0)
        person_boxes = results[0].boxes.xyxy  # Person class bounding boxes

        # Load the existing LabelMe JSON file (if it exists)
        json_filename = image_filename.replace(".jpg", ".json").replace(".png", ".json")
        json_path = os.path.join(input_dir, json_filename)
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
        else:
            data = {"shapes": [], "imagePath": image_filename, "imageData": None, "imageHeight": image.shape[0], "imageWidth": image.shape[1]}
        
        # Loop through the detected person bounding boxes
        for person_box in person_boxes:
            # Get the coordinates of the person box
            xmin_person, ymin_person, xmax_person, ymax_person = map(int, person_box)

            # Append detected person boxes to LabelMe JSON
            points = convert_to_labelme_bbox(person_box)
            new_shape = {
                "label": "person",
                "points": points,
                "group_id": None,
                "description": "",
                "shape_type": "rectangle",
                "flags": {},
                "mask": None
            }
            data['shapes'].append(new_shape)

        # Save the updated JSON data with the same name in the output directory
        updated_json_filename = image_filename.replace(".jpg", ".json").replace(".png", ".json")
        updated_json_path = os.path.join(output_dir, updated_json_filename)
        with open(updated_json_path, 'w') as f:
            json.dump(data, f, indent=4)

        # Copy the original image to the output folder with the same name
        copy2(image_path, os.path.join(output_dir, image_filename))  # Copy original image

        print(f"Updated JSON and image saved for {image_filename}.")

print("Processing completed for all images.")
