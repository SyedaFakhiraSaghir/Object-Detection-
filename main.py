
import kagglehub
import os
from ultralytics import YOLO
import torch
import shutil   
import yaml

local_dataset_dir = os.path.join(os.getcwd(), "coco128", "coco128")
dataset_yaml = os.path.join(local_dataset_dir, "data.yaml")

print(f" Dataset directory: {local_dataset_dir}")
print(f" YAML file: {dataset_yaml}")

if not os.path.exists(local_dataset_dir):
    print(" Downloading dataset...")
    path = kagglehub.dataset_download("ultralytics/coco128")
    # Create the nested directory structure
    os.makedirs(os.path.dirname(local_dataset_dir), exist_ok=True)
    shutil.copytree(path, local_dataset_dir)
    print("Dataset downloaded successfully")
else:
    print("Dataset already exists")

# Verify the dataset structure
print("\n🔍 Verifying dataset structure:")
images_dir = os.path.join(local_dataset_dir, "images", "train2017")
labels_dir = os.path.join(local_dataset_dir, "labels", "train2017")

print(f"Images directory: {images_dir} - Exists: {os.path.exists(images_dir)}")
print(f"Labels directory: {labels_dir} - Exists: {os.path.exists(labels_dir)}")
print(f"YAML file: {dataset_yaml} - Exists: {os.path.exists(dataset_yaml)}")

# Fix the data.yaml file if paths are incorrect
if os.path.exists(dataset_yaml):
    with open(dataset_yaml, 'r') as f:
        data = yaml.safe_load(f)
    
    print(f"\n Original data.yaml content:")
    print(f"Path: {data.get('path', 'Not found')}")
    print(f"Train: {data.get('train', 'Not found')}")
    print(f"Val: {data.get('val', 'Not found')}")
    
    # Update paths to be absolute
    data['path'] = local_dataset_dir
    if not data.get('train', '').startswith(local_dataset_dir):
        data['train'] = os.path.join(local_dataset_dir, 'images/train2017')
    if not data.get('val', '').startswith(local_dataset_dir):
        data['val'] = os.path.join(local_dataset_dir, 'images/train2017')
    
    # Write back the fixed YAML
    with open(dataset_yaml, 'w') as f:
        yaml.dump(data, f)
    
    print("Updated data.yaml with absolute paths")

# Load YOLO model
print("\nLoading YOLO model...")
model = YOLO("yolov5su.pt")

# Train model with correct dataset path
print(f"Starting training with: {dataset_yaml}")
train_results = model.train(
    data=dataset_yaml,
    epochs=50,
    imgsz=640,
    batch=16,
    lr0=0.001,
    lrf=0.01,
    weight_decay=0.0005,
    patience=10,
    optimizer='Adam',
    project="object_detection_runs",
    name="exp1",
    exist_ok=True
)

print("Training completed!")

#Hyperparameter tuning (optional)
print("\n🔧 Starting hyperparameter tuning...")
for lr in [0.001, 0.0005, 0.0001]:
    print(f"\n--- Training with learning rate: {lr} ---")
    model = YOLO("yolov5su.pt")
    model.train(
        data=dataset_yaml,
        epochs=20,
        imgsz=640,
        batch=16,
        lr0=lr,
        optimizer="Adam",
        project="object_detection_runs",
        name=f"lr_{lr}",
        exist_ok=True
    )

#Validate model
print("\nValidating model...")
val_results = model.val()

# Evaluate metrics
print("\n--- Evaluation Results ---")
print("mAP50:", val_results.box.map50)
print("mAP50-95:", val_results.box.map)
print("Precision:", val_results.box.precision)
print("Recall:", val_results.box.recall)