import os

from PIL import Image
from dotenv import load_dotenv
from roboflow import download_dataset
import supervision as sv
from tqdm import tqdm
from supervision.metrics import MeanAveragePrecision
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


def main():
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

    targets = []
    predictions = []

    for path, image, annotations in tqdm(ds):
        image = Image.open(path)
        detections = model.predict(image, threshold=0)

        targets.append(annotations)
        predictions.append(detections)
    map_metric = MeanAveragePrecision()
    map_result = map_metric.update(predictions, targets).compute()
    print(map_result)
    return


if __name__ == "__main__":
    main()
