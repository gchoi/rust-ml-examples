import os

from dotenv import load_dotenv
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
NUM_EPOCHS = 10
BATCH_SIZE = 8


def main():
    dataset = download_dataset(
        dataset_url=DATASET_URL,
        model_format=MODEL_FORMAT
    )

    model = model_dict.get(MODEL_SIZE)()
    model.train(
        dataset_dir=dataset.location,
        epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        grad_accum_steps=2
    )


if __name__ == "__main__":
    main()
