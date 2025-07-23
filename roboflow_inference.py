from inference import get_model
import supervision as sv
import cv2
import os
import csv

def load_model(model_id="birds-vs-drones-abtzu/3", api_key=None):
    print(f"\nLoading model: {model_id}\n")
    return get_model(model_id=model_id, api_key=api_key)

def get_image_paths(image_dir="Birds-Vs-Drones-2/test/"):
    return [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]

def run_inference_on_images(model, image_paths, output_dir="output/data", log_file="output/inference_results.csv"):
    os.makedirs(output_dir, exist_ok=True)

    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "class", "confidence", "x", "y", "width", "height"])

        for path in image_paths:
            image = cv2.imread(path)
            results = model.infer(image)[0]

            # Log predictions
            for r in results.predictions:
                print(f"{os.path.basename(path)} — {r.class_name} @ {r.confidence:.2f}")

                writer.writerow([
                    os.path.basename(path),
                    r.class_name,
                    round(r.confidence, 3),
                    round(r.x, 2),
                    round(r.y, 2),
                    round(r.width, 2),
                    round(r.height, 2)
                ])

            # Annotate and save
            detections = sv.Detections.from_inference(results)
            annotated = sv.BoxAnnotator().annotate(image, detections)
            annotated = sv.LabelAnnotator().annotate(annotated, detections)

            save_path = os.path.join(output_dir, os.path.basename(path))
            cv2.imwrite(save_path, annotated)
            print(f"Saved annotated image to: {save_path}")

    print(f"\nInference log saved to: {log_file}")

def run_full_inference_pipeline():
    api_key = os.getenv("ROBOFLOW_KEY")
    model = load_model(api_key=api_key)
    image_paths = get_image_paths("Birds-Vs-Drones-2/test/")
    run_inference_on_images(model, image_paths)

# Optional main runner
if __name__ == "__main__":
    run_full_inference_pipeline()
    
    # sv.plot_image(annotated, size=(10, 10))  # or cv2.imwrite(...) if saving