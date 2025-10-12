import os
from io import BytesIO
import requests

import supervision as sv
from inference import get_model
from PIL import Image


IMAGE_URL = "https://media.roboflow.com/dog.jpeg"

def main() -> None:
    image = Image.open(BytesIO(requests.get(IMAGE_URL).content))

    model = get_model("rfdetr-base")

    predictions = model.infer(image, confidence=0.5)[0]

    detections = sv.Detections.from_inference(predictions)

    labels = [prediction.class_name for prediction in predictions.predictions]

    annotated_image = image.copy()
    annotated_image = sv.BoxAnnotator(color=sv.ColorPalette.ROBOFLOW).annotate(annotated_image, detections)
    annotated_image = sv.LabelAnnotator(color=sv.ColorPalette.ROBOFLOW).annotate(annotated_image, detections, labels)
    annotated_image.show()
    return


if __name__ == "__main__":
    main()
