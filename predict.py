# ==================== PART 3: RUN INFERENCE ON NEW IMAGE ====================
from ultralytics import YOLO
import cv2
from matplotlib import pyplot as plt

# 1️⃣ Load trained YOLO model
model = YOLO("object_detection_runs/exp1/weights/best.pt")

# 2️⃣ Specify the input image
image_path = "image.png"   # Change to your image filename

# 3️⃣ Run object detection
results = model.predict(source=image_path, conf=0.25, save=True, show=False)

# 4️⃣ Display the detected image
for r in results:
    im_array = r.plot()  # BGR image with detections
    im_rgb = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 8))
    plt.imshow(im_rgb)
    plt.axis("off")
    plt.title("Detected Objects")
    plt.show()

print("\n✅ Object detection complete! Check 'runs/predict' folder for saved results.")
