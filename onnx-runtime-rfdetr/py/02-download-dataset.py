import os

from dotenv import load_dotenv
from roboflow import download_dataset


load_dotenv()
os.environ["ROBOFLOW_API_KEY"] = os.environ.get('ROBOFLOW_API_KEY')


def main() -> None:
    dataset = download_dataset(
        dataset_url="https://universe.roboflow.com/roboflow-jvuqo/basketball-player-detection-2/13",
        model_format="coco"
    )
    return


if __name__ == "__main__":
    main()
