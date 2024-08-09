import streamlit as st
import json
import os
from typing import Any, Dict, List
import cv2
import numpy as np
from PIL import Image
import random


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    data = []
    with open(file_path, "r") as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data


def draw_bounding_boxes(
    image: np.ndarray,
    predictions: Dict,
    is_prediction: bool = False,
    confidence_threshold: float = 0.5,
) -> np.ndarray:
    for box_idx in range(len(predictions["bboxes"])):
        bbox = predictions["bboxes"][box_idx]
        cx, cy, w, h = bbox
        class_id = predictions["class_ids"][box_idx]
        if is_prediction:
            confidence = predictions["confidences"][box_idx]
            if confidence < confidence_threshold:
                continue
            color = (0, 0, 255)  # Red for predictions
        else:
            color = (0, 255, 0)  # Green for targets

        x1 = int((cx - w / 2) * image.shape[1])
        y1 = int((cy - h / 2) * image.shape[0])
        x2 = int((cx + w / 2) * image.shape[1])
        y2 = int((cy + h / 2) * image.shape[0])

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        if is_prediction:
            image_text = f"{class_id}: {confidence:.2f}"
        else:
            image_text = f"{class_id}"
        cv2.putText(
            image, image_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )
    return image


def process_image(
    file_info: Dict[str, Any], image_dir: str, confidence_threshold: float
) -> np.ndarray:
    img_path = os.path.join(image_dir, file_info["img_path"])
    predictions = file_info["predictions"]
    targets = file_info["targets"]

    image = cv2.imread(img_path)
    if image is None:
        raise ValueError(f"Image at path {img_path} could not be loaded.")

    pred_image = image.copy()
    targ_image = image.copy()

    pred_image = draw_bounding_boxes(
        pred_image,
        predictions,
        is_prediction=True,
        confidence_threshold=confidence_threshold,
    )
    targ_image = draw_bounding_boxes(targ_image, targets, is_prediction=False)
    black_bar = np.ones((image.shape[0], 10, 3), dtype=np.uint8)

    combined_image = np.hstack((pred_image, black_bar, targ_image))
    return combined_image


def main():
    st.title("Predictions vs Targets Bounding Boxes")

    input_file = st.text_input(
        "Path to input JSONL file",
        "/Users/derek/Desktop/cv-training/rt-detr/drone_v2_all_layers.jsonl",
    )
    image_dir = st.text_input(
        "Path to image directory",
        "/Users/derek/Desktop/cv-training/datasets/drone_dataset/valid/images",
    )

    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)
    if st.button("Show Random Image"):
        if not os.path.exists(input_file):
            st.error("Input JSONL file does not exist.")
        elif not os.path.exists(image_dir):
            st.error("Image directory does not exist.")
        else:
            data = load_jsonl(input_file)
            rand_idx = random.randint(0, len(data) - 1)
            combined_image = process_image(
                data[rand_idx], image_dir, confidence_threshold
            )

            st.image(
                combined_image,
                channels="BGR",
                caption="Predictions (left) vs Targets (right)",
            )


if __name__ == "__main__":
    main()
