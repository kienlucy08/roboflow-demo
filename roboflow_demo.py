import io
import os
import requests
import supervision as sv
from PIL import Image
from rfdetr import RFDETRBase
from roboflow import Roboflow
import json
import shutil
from pathlib import Path

ROBOFLOW_KEY = os.getenv("ROBOFLOW_KEY")

if __name__ == "__main__":
    # authenticate
    rf = Roboflow(api_key=ROBOFLOW_KEY)
    project = rf.workspace("roboflowlearn").project("my-first-project-c5pgo")

    # download
    dataset = project.version(2).download("coco")
    #data info
    annotation_file = Path(dataset.location) / "train" / "_annotations.coco.json"
    if not annotation_file.exists():
        print(" Annotation file missing in train/. Creating train split from test...")
        os.makedirs(Path(dataset.location) / "train", exist_ok=True)

        # Move images from test/ to train/
        test_images = list((Path(dataset.location) / "test").glob("*.jpg"))
        for img in test_images:
            shutil.copy(img, Path(dataset.location) / "train" / img.name)

        # Move annotation file from test/ to train/
        test_ann = Path(dataset.location) / "test" / "_annotations.coco.json"
        shutil.copy(test_ann, Path(dataset.location) / "train" / "_annotations.coco.json")

    with open(Path(dataset.location) / "train" / "_annotations.coco.json") as f:
        coco = json.load(f)

    print(f"Images: {len(coco.get('images', []))}")
    print(f"Annotations: {len(coco.get('annotations', []))}")
    print(f"Categories: {len(coco.get('categories', []))}")

    model = RFDETRBase()
    history = []

    def callback2(data):
        history.append(data)

    model.callbacks["on_fit_epoch_end"].append(callback2)

    # Add dummy validation split to avoid crash
    val_dir = os.path.join(dataset.location, "valid")
    os.makedirs(val_dir, exist_ok=True)
    shutil.copy(
        os.path.join(dataset.location, "train", "_annotations.coco.json"),
        os.path.join(val_dir, "_annotations.coco.json")
    )

    # Train
    model.train(dataset_dir=dataset.location, epochs=15, batch_size=4, lr=1e-4, eval=False)




# url = "https://media.roboflow.com/notebooks/examples/dog-2.jpeg"

# image = Image.open(io.BytesIO(requests.get(url).content))
# detections = model.predict(image, threshold=0.5)

# annotated_image = image.copy()
# annotated_image = sv.BoxAnnotator().annotate(annotated_image, detections)
# annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections)

# sv.plot_image(annotated_image)

# print(sv.__version__)