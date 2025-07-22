import os
import json
from pathlib import Path
from PIL import Image
from collections import defaultdict

from roboflow import Roboflow
from rfdetr import RFDETRBase


def iou(boxA, boxB):
    # box = [x, y, w, h]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    union = boxAArea + boxBArea - interArea
    return interArea / union if union > 0 else 0


def evaluate_predictions(ground_truths, predictions, iou_thresh=0.5):
    TP = 0
    FP = 0
    FN = 0

    matched_gt = set()

    for pred_box, pred_cls in predictions:
        match_found = False
        for idx, (gt_box, gt_cls) in enumerate(ground_truths):
            if idx in matched_gt:
                continue
            if pred_cls == gt_cls and iou(pred_box, gt_box) >= iou_thresh:
                TP += 1
                matched_gt.add(idx)
                match_found = True
                break
        if not match_found:
            FP += 1

    FN = len(ground_truths) - len(matched_gt)
    return TP, FP, FN


# === Roboflow Setup ===
ROBOFLOW_KEY = os.getenv("ROBOFLOW_KEY")  # Or hardcode
rf = Roboflow(api_key=ROBOFLOW_KEY)
project = rf.workspace("roboflowlearn").project("birds-vs-drones-abtzu")
dataset = project.version(1).download("coco")

# Load test annotations
test_dir = Path(dataset.location) / "test"
ann_path = test_dir / "_annotations.coco.json"
with open(ann_path) as f:
    coco = json.load(f)

image_map = {img["id"]: img["file_name"] for img in coco["images"]}
gt_by_image = defaultdict(list)
for ann in coco["annotations"]:
    gt_by_image[ann["image_id"]].append((ann["bbox"], ann["category_id"]))

# Load your model
model = RFDETRBase()

# Evaluate
total_TP = total_FP = total_FN = 0

for img_id, img_name in image_map.items():
    img_path = test_dir / img_name
    image = Image.open(img_path)
    preds = model.predict(image, threshold=0.5)

    pred_boxes = [
        (p[0], p[1]) for p in preds  # p = (xywh, class_id)
    ]
    gts = gt_by_image[img_id]
    TP, FP, FN = evaluate_predictions(gts, pred_boxes)
    total_TP += TP
    total_FP += FP
    total_FN += FN

# Calculate final metrics
precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) else 0
recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) else 0
f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0

print("\n📊 Evaluation Results (Roboflow Test Set)")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1 Score:  {f1:.3f}")

