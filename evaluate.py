# ==================== PART 2: PLOTS & CONFUSION MATRIX ====================
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from ultralytics import YOLO

# Load trained model
model = YOLO("object_detection_runs/exp1/weights/best.pt")

# Evaluate model to generate metrics and confusion matrix
results = model.val(save_json=True, plots=True)

# Visualize training results
results_dir = "object_detection_runs/exp1/results.png"
if os.path.exists(results_dir):
    img = plt.imread(results_dir)
    plt.imshow(img)
    plt.axis('off')
    plt.title("Training Metrics Summary")
    plt.show()

# Load confusion matrix data
conf_matrix = results.confusion_matrix.matrix
class_names = results.names

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
