from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# 1. Load your best trained model (from runs directory)
model_path = "object_detection_runs/exp1/weights/best.pt"
if not os.path.exists(model_path):
    model_path = "object_detection_runs/exp1/weights/last.pt"
    
print(f"Loading model from: {model_path}")
model = YOLO(model_path)

# 2. Validate on validation data
results = model.val()

# 3. Print evaluation metrics
print("\n--- Evaluation Metrics ---")
print(f"mAP50: {results.box.map50:.4f}")
print(f"mAP50-95: {results.box.map:.4f}")

# Correct way to access precision and recall (they are arrays, so we take the mean)
if hasattr(results.box, 'p') and results.box.p is not None:
    print(f"Mean Precision: {np.mean(results.box.p):.4f}")
else:
    print("Precision: Data not available")

if hasattr(results.box, 'r') and results.box.r is not None:
    print(f"Mean Recall: {np.mean(results.box.r):.4f}")
else:
    print("Recall: Data not available")

# 4. Print class-wise metrics
print("\n--- Class-wise Metrics ---")
if hasattr(results.box, 'ap_class_index') and hasattr(results.box, 'ap50'):
    class_names = list(model.names.values())
    for i, class_idx in enumerate(results.box.ap_class_index):
        class_name = class_names[class_idx] if class_idx < len(class_names) else f"Class {class_idx}"
        precision = results.box.p[i] if hasattr(results.box, 'p') and i < len(results.box.p) else "N/A"
        recall = results.box.r[i] if hasattr(results.box, 'r') and i < len(results.box.r) else "N/A"
        print(f"{class_name}: mAP50={results.box.ap50[i]:.4f}, Precision={precision}, Recall={recall}")

# 5. Generate confusion matrix
print("\nGenerating confusion matrix...")
if hasattr(results, 'confusion_matrix') and results.confusion_matrix is not None:
    cm_array = results.confusion_matrix.matrix
    class_names = list(model.names.values())
    
    # Only show classes that actually appear in the confusion matrix
    num_classes = min(len(class_names), cm_array.shape[0] - 1)  # -1 for background
    
    plt.figure(figsize=(15, 12))
    
    # Normalize the confusion matrix (excluding background)
    cm_normalized = cm_array[:num_classes, :num_classes].astype('float')
    row_sums = cm_normalized.sum(axis=1)
    cm_normalized = cm_normalized / row_sums[:, np.newaxis]
    
    # Create heatmap
    sns.heatmap(cm_normalized, 
                xticklabels=class_names[:num_classes], 
                yticklabels=class_names[:num_classes],
                annot=True, 
                fmt='.2f', 
                cmap='Blues',
                cbar_kws={'label': 'Normalized Percentage'})
    
    plt.title("Normalized Confusion Matrix", fontsize=16, fontweight='bold')
    plt.xlabel("Predicted Labels", fontsize=12)
    plt.ylabel("True Labels", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("confusion_matrix_normalized.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Also create raw counts confusion matrix
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm_array[:num_classes, :num_classes], 
                xticklabels=class_names[:num_classes], 
                yticklabels=class_names[:num_classes],
                annot=True, 
                fmt='d', 
                cmap='Blues',
                cbar_kws={'label': 'Count'})
    
    plt.title("Confusion Matrix (Raw Counts)", fontsize=16, fontweight='bold')
    plt.xlabel("Predicted Labels", fontsize=12)
    plt.ylabel("True Labels", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("confusion_matrix_raw.png", dpi=300, bbox_inches='tight')
    plt.show()
    
else:
    print("Confusion matrix not available in results")

# 6. Plot performance curves if available
print("\nGenerating performance curves...")
if hasattr(results, 'curves'):
    curves = results.curves
    
    # Precision-Confidence curve
    if 'P' in curves:
        plt.figure(figsize=(10, 6))
        plt.plot(curves['P'][0], curves['P'][1], 'b-', linewidth=2, label='Precision')
        plt.xlabel('Confidence Threshold')
        plt.ylabel('Precision')
        plt.title('Precision vs Confidence')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig("precision_curve.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    # Recall-Confidence curve
    if 'R' in curves:
        plt.figure(figsize=(10, 6))
        plt.plot(curves['R'][0], curves['R'][1], 'r-', linewidth=2, label='Recall')
        plt.xlabel('Confidence Threshold')
        plt.ylabel('Recall')
        plt.title('Recall vs Confidence')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig("recall_curve.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    # F1-Confidence curve
    if 'F1' in curves:
        plt.figure(figsize=(10, 6))
        plt.plot(curves['F1'][0], curves['F1'][1], 'g-', linewidth=2, label='F1-Score')
        plt.xlabel('Confidence Threshold')
        plt.ylabel('F1-Score')
        plt.title('F1-Score vs Confidence')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig("f1_curve.png", dpi=300, bbox_inches='tight')
        plt.show()

# 7. Plot class-wise performance
print("\nGenerating class-wise performance plot...")
if hasattr(results.box, 'ap_class_index') and hasattr(results.box, 'ap50'):
    classes = results.box.ap_class_index
    ap_scores = results.box.ap50
    
    plt.figure(figsize=(12, 8))
    bars = plt.bar(range(len(classes)), ap_scores, color='skyblue', edgecolor='navy', alpha=0.7)
    
    # Add value labels on bars
    for bar, value in zip(bars, ap_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('mAP@0.50', fontsize=12)
    plt.title('Class-wise mAP@0.50 Performance', fontsize=14, fontweight='bold')
    plt.xticks(range(len(classes)), [class_names[i] for i in classes], rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig('class_performance.png', dpi=300, bbox_inches='tight')
    plt.show()

# 8. Save detailed metrics to file
def save_detailed_report(results, model, filename="detailed_evaluation_report.txt"):
    """Save comprehensive evaluation report"""
    with open(filename, 'w') as f:
        f.write("YOLO Model Detailed Evaluation Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("OVERALL METRICS:\n")
        f.write(f"mAP@0.50: {results.box.map50:.4f}\n")
        f.write(f"mAP@0.50:0.95: {results.box.map:.4f}\n")
        
        if hasattr(results.box, 'p') and results.box.p is not None:
            f.write(f"Mean Precision: {np.mean(results.box.p):.4f}\n")
        if hasattr(results.box, 'r') and results.box.r is not None:
            f.write(f"Mean Recall: {np.mean(results.box.r):.4f}\n")
        
        f.write(f"\nSPEED ANALYSIS:\n")
        f.write(f"Pre-process time: {results.speed['preprocess']:.2f} ms\n")
        f.write(f"Inference time: {results.speed['inference']:.2f} ms\n")
        f.write(f"Post-process time: {results.speed['postprocess']:.2f} ms\n")
        
        f.write(f"\nCLASS-WISE METRICS:\n")
        if hasattr(results.box, 'ap_class_index') and hasattr(results.box, 'ap50'):
            class_names = list(model.names.values())
            for i, class_idx in enumerate(results.box.ap_class_index):
                class_name = class_names[class_idx] if class_idx < len(class_names) else f"Class {class_idx}"
                precision = results.box.p[i] if hasattr(results.box, 'p') and i < len(results.box.p) else "N/A"
                recall = results.box.r[i] if hasattr(results.box, 'r') and i < len(results.box.r) else "N/A"
                f.write(f"{class_name}: mAP50={results.box.ap50[i]:.4f}, Precision={precision}, Recall={recall}\n")

save_detailed_report(results, model)

print("\nEvaluation complete!")
print("Generated files:")
print("  - confusion_matrix_normalized.png")
print("  - confusion_matrix_raw.png")
print("  - precision_curve.png (if available)")
print("  - recall_curve.png (if available)")
print("  - f1_curve.png (if available)")
print("  - class_performance.png")
print("  - detailed_evaluation_report.txt")