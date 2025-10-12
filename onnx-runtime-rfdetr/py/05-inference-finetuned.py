import os
from dotenv import load_dotenv

from PIL import Image
import supervision as sv
from roboflow import download_dataset
from rfdetr import (
    RFDETRBase,
    RFDETRNano,
    RFDETRSmall,
    RFDETRMedium,
    RFDETRLarge
)


model_dict = {
    "base": RFDETRBase,
    "nano": RFDETRNano,
    "small": RFDETRSmall,
    "medium": RFDETRMedium,
    "large": RFDETRLarge
}

load_dotenv()
os.environ["ROBOFLOW_API_KEY"] = os.environ.get('ROBOFLOW_API_KEY')

# -- Configurations
MODEL_SIZE = "base"
DATASET_URL = "https://universe.roboflow.com/roboflow-jvuqo/basketball-player-detection-2/13"
MODEL_FORMAT = "coco"
CHECKPOINT_PATH = "./output/checkpoint_best_total.pth"
TEST_IMAGE = "./basketball-player-detection-2-13/test/boston-celtics-new-york-knicks-game-1-q1-10_25-10_20_mov-0008_jpg.rf.89b2669c2b5bf17e789dfb5028a2ceb1.jpg"


def main() -> None:
    dataset = download_dataset(
        dataset_url=DATASET_URL,
        model_format=MODEL_FORMAT
    )

    model = model_dict.get(MODEL_SIZE)(pretrain_weights=CHECKPOINT_PATH)
    model.optimize_for_inference()

    ds = sv.DetectionDataset.from_coco(
        images_directory_path=f"{dataset.location}/test",
        annotations_path=f"{dataset.location}/test/_annotations.coco.json",
    )

    path, image, annotations = ds[0]
    image = Image.open(TEST_IMAGE)

    detections = model.predict(image, threshold=0.5)

    text_scale = sv.calculate_optimal_text_scale(resolution_wh=image.size)
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=image.size)
    color = sv.ColorPalette.from_hex([
        "#ffff00", "#ff9b00", "#ff66ff", "#3399ff", "#ff66b2", "#ff8080",
        "#b266ff", "#9999ff", "#66ffff", "#33ff99", "#66ff66", "#99ff00"
    ])

    bbox_annotator = sv.BoxAnnotator(color=color,thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        color=color,
        text_color=sv.Color.BLACK,
        text_scale=text_scale)

    annotations_labels = [
        f"{ds.classes[class_id]}"
        for class_id
        in annotations.class_id
    ]

    detections_labels = [
        f"{ds.classes[class_id]} {confidence:.2f}"
        for class_id, confidence
        in zip(detections.class_id, detections.confidence)
    ]

    annotation_image = image.copy()
    annotation_image = bbox_annotator.annotate(annotation_image, annotations)
    annotation_image = label_annotator.annotate(annotation_image, annotations, annotations_labels)

    detections_image = image.copy()
    detections_image = bbox_annotator.annotate(detections_image, detections)
    detections_image = label_annotator.annotate(detections_image, detections, detections_labels)

    sv.plot_images_grid(
        images=[annotation_image, detections_image],
        grid_size=(1, 2),
        titles=["Annotation", "Detection"]
    )
    detections_image.save("./output/annotation_image.png")
    return


if __name__ == "__main__":
    main()
