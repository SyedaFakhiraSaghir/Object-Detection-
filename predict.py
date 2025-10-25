# ==================== PART 3: INFERENCE ON NEW IMAGE ====================
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2

# Load trained model
model = YOLO("object_detection_runs/exp1/weights/best.pt")

# Path to new image (you can replace this with your own)
img_path = "test_image.jpg"  # make sure this image exists in your working dir

# Perform detection
results = model(img_path, conf=0.5)

# Save and show results
results.save(save_dir="inference_results")

# Visualize with bounding boxes
img = cv2.imread("inference_results/image0.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10, 8))
plt.imshow(img)
plt.axis('off')
plt.title("Detected Objects")
plt.show()
