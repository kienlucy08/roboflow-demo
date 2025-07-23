import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os
import json

def load_results(csv_path="output/inference_results.csv"):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find: {csv_path}")
    return pd.read_csv(csv_path)

def load_coco_ground_truth(json_path):
    with open(json_path, 'r') as f:
        coco = json.load(f)
    # Build mapping from category_id to category_name
    category_map = {cat['id']: cat['name'] for cat in coco['categories']}
    # Build image_id to filename map
    image_map = {img['id']: img['file_name'] for img in coco['images']}
    # Create DataFrame of ground truth labels
    data = []
    for ann in coco['annotations']:
        image_id = ann['image_id']
        category_id = ann['category_id']
        file_name = image_map[image_id]
        class_name = category_map[category_id]
        data.append({'image': file_name, 'true_class': class_name})

    return pd.DataFrame(data)

def load_predictions(csv_path):
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"class": "predicted_class"})  # Update if needed
    return df[["image", "predicted_class"]]

def compute_metrics(df):
    print("\n=== Summary Metrics ===")
    
    # Overall avg confidence
    overall_avg = df["confidence"].mean()
    overall_avg_perc = overall_avg * 100
    print(f"=== Overall Average Confidence: {overall_avg_perc:.3f} ===")

    # Per class avg
    class_avg = df.groupby("class")["confidence"].mean()
    class_avg_perc = class_avg * 100
    print("\n=== Average Confidence per Class: ===")
    print(class_avg_perc.round(2).astype(str) + "%")

    # Count per class
    class_count = df["class"].value_counts()
    print("\n=== Detection Count per Class: ===")
    print(class_count)

    # Count per image
    image_count = df["image"].value_counts()
    print("\n=== Detections Per Image: ===")
    print(image_count)

    return {
        "overall_avg": overall_avg,
        "class_avg": class_avg,
        "class_count": class_count,
        "image_count": image_count
    }

def generate_plots(df, output_dir="output/stats/"):
    sns.set(style="whitegrid")
    os.makedirs(output_dir, exist_ok=True)

    # Detection count per class
    plt.figure(figsize=(8, 4))
    sns.barplot(x=df["class"].value_counts().index, y=df["class"].value_counts().values, palette="viridis")
    plt.title("Detection Count per Class")
    plt.ylabel("Count")
    plt.xlabel("Class")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "class_count.png"))
    plt.close()

    # Average confidence per class
    plt.figure(figsize=(8, 4))
    class_avg = df.groupby("class")["confidence"].mean()
    sns.barplot(x=class_avg.index, y=class_avg.values, palette="coolwarm")
    plt.title("Average Confidence per Class")
    plt.ylabel("Confidence")
    plt.xlabel("Class")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "class_confidence.png"))
    plt.close()

    # Confidence distribution
    plt.figure(figsize=(8, 4))
    sns.histplot(df["confidence"], bins=20, kde=True, color="skyblue")
    plt.title("Confidence Score Distribution")
    plt.xlabel("Confidence")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confidence_hist.png"))
    plt.close()

    print("=== Saved analysis plots to: ===", output_dir)

def build_confusion_matrix(coco_json="Birds-Vs-Drones-2/test/_annotations.coco.json",
                           pred_csv="output/inference_results.csv",
                           output_dir="output/stats/"):
    os.makedirs(output_dir, exist_ok=True)

    gt_df = load_coco_ground_truth(coco_json)
    pred_df = load_predictions(pred_csv)

    merged_df = pd.merge(gt_df, pred_df, on="image")

    if merged_df.empty:
        raise ValueError("Merged dataframe is empty. Check if image names match between JSON and CSV.")

    y_true = merged_df["true_class"]
    y_pred = merged_df["predicted_class"]

    labels = sorted(list(set(y_true) | set(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    disp.plot(cmap='Blues', values_format='d')
    disp.ax_.set_title("Confusion Matrix")
    disp.figure_.tight_layout()
    disp.figure_.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close(disp.figure_)
    print("Saved confusion matrix to:", os.path.join(output_dir, "confusion_matrix.png"))

def build_confusion_matrix_from_predictions(
    coco_json_path="Birds-Vs-Drones-2/test/_annotations.coco.json",
    predictions_csv_path="output/inference_results.csv",
    output_dir: str = "output/stats/"
):
    os.makedirs(output_dir, exist_ok=True)

    # --- Load ground truth ---
    with open(coco_json_path, 'r') as f:
        coco = json.load(f)

    category_map = {cat["id"]: cat["name"] for cat in coco["categories"]}
    image_map = {img["id"]: img["file_name"] for img in coco["images"]}

    # Build ground truth dataframe
    gt_data = []
    for ann in coco["annotations"]:
        image_id = ann["image_id"]
        category_id = ann["category_id"]
        image_name = image_map[image_id].lower()
        true_class = category_map[category_id]
        gt_data.append({"image": image_name, "true_class": true_class})

    gt_df = pd.DataFrame(gt_data)

    # --- Load predictions ---
    pred_df = pd.read_csv(predictions_csv_path)
    pred_df["image"] = pred_df["image"].apply(lambda x: x.strip().lower())
    pred_df = pred_df.rename(columns={"class": "predicted_class"})

    # --- Merge on image ---
    merged_df = pd.merge(gt_df, pred_df, on="image")

    if merged_df.empty:
        raise ValueError("No matching records found. Check filename formatting between JSON and CSV.")

    y_true = merged_df["true_class"]
    y_pred = merged_df["predicted_class"]

    labels = sorted(list(set(y_true) | set(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    disp.plot(cmap='Blues', values_format='d')
    disp.ax_.set_title("Model Confusion Matrix (Test Set)")
    disp.figure_.tight_layout()

    out_path = os.path.join(output_dir, "confusion_matrix.png")
    disp.figure_.savefig(out_path)
    plt.close(disp.figure_)

    print(f"Confusion matrix saved to: {out_path}")

def analyze(csv_path="output/inference_results.csv"):
    df = load_results(csv_path)
    compute_metrics(df)
    generate_plots(df)
    build_confusion_matrix_from_predictions()

def main():
    try:
        analyze()
    except Exception as e:
        print(f"Error during analysis: {e}")

if __name__ == "__main__":
    main()