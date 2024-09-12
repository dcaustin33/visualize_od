import json
import os
import random
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image

from matcher import HungarianMatcher
from postprocessing import box_metrics


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
    img_path = os.path.join(image_dir, file_info["img_name"])
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


def filter(
    metrics: Dict[str, Any],
    iou_less_than: Optional[float] = None,
    f1_less_than: Optional[float] = None,
    false_positives_greater_than: Optional[int] = None,
    false_negatives_greater_than: Optional[int] = None,
) -> bool:
    """Function to allow for filtering if a condition is met"""
    all_outputs = []
    if iou_less_than is not None:
        for iou in metrics["matched_ious"]:
            if iou < iou_less_than:
                all_outputs.append(True)
    if f1_less_than is not None:
        if metrics["f1"] < f1_less_than:
            all_outputs.append(True)
    if false_positives_greater_than is not None:
        if metrics["fp"] > false_positives_greater_than:
            all_outputs.append(True)
    if false_negatives_greater_than is not None:
        if metrics["fn"] > false_negatives_greater_than:
            all_outputs.append(True)
    return all(all_outputs) if all_outputs else False


def main():
    st.title("Predictions vs Targets Bounding Boxes")

    input_file = st.text_input(
        "Path to input JSONL file",
        "drone_v2_maciullo_r34_3decoder.jsonl",
    )
    image_dir = st.text_input(
        "Path to image directory",
        "maciullo_real_world_drones/train/images",
    )
    matcher = HungarianMatcher(
        weight_dict={
            "cost_class": 2,
            "cost_bbox": 5,
            "cost_giou": 2,
        },
        alpha=0.25,
        gamma=2,
    )

    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)
    data = load_jsonl(input_file)
    filter_conditions = {}

    if st.checkbox("Filter by IOU"):
        filter_conditions["iou_less_than"] = st.slider(
            "IOU Less Than", 0.0, 1.0, 0.5, 0.01
        )

    if st.checkbox("Filter by F1 Score"):
        filter_conditions["f1_less_than"] = st.slider(
            "F1 Less Than", 0.0, 1.0, 0.5, 0.01
        )

    if st.checkbox("Filter by False Positives"):
        filter_conditions["false_positives_greater_than"] = st.slider(
            "False Positives Greater Than", 0, 100, 0, 1
        )

    if st.checkbox("Filter by False Negatives"):
        filter_conditions["false_negatives_greater_than"] = st.slider(
            "False Negatives Greater Than", 0, 100, 0, 1
        )
    if st.button("Show Random Image"):
        if not os.path.exists(input_file):
            st.error("Input JSONL file does not exist.")
        elif not os.path.exists(image_dir):
            st.error("Image directory does not exist.")
        else:
            count = 0
            while True:
                count += 1
                if count > 1000:
                    st.error(
                        "Could not find an image that satisfies the filter conditions. "
                        "Showing random image."
                    )
                    break
                rand_idx = random.randint(0, len(data) - 1)
                if not os.path.exists(
                    os.path.join(image_dir, data[rand_idx]["img_name"])
                ):
                    continue
                metrics = box_metrics(
                    preds_logits=torch.tensor(
                        data[rand_idx]["predictions"]["confidences"]
                    ).view(-1, 1),
                    preds_boxes=torch.tensor(data[rand_idx]["predictions"]["bboxes"]),
                    target_classes=torch.tensor(data[rand_idx]["targets"]["class_ids"]),
                    target_boxes=torch.tensor(data[rand_idx]["targets"]["bboxes"]),
                    matcher=matcher,
                    loss_value=0,
                    confidence_threshold=confidence_threshold,
                ).to_dict()

                # check if there are any conditions
                if filter_conditions:
                    if filter(metrics, **filter_conditions):
                        break
                else:
                    break
            combined_image = process_image(
                data[rand_idx], image_dir, confidence_threshold
            )

            st.image(
                combined_image,
                channels="BGR",
                caption="Predictions (left) vs Targets (right)",
            )
            st.write(f"F1: {metrics['f1']}")
            st.write(f"FP: {metrics['fp']}")
            st.write(f"FN: {metrics['fn']}")
            st.write(f"TP: {metrics['tp']}")
            st.write(f"IOU: {metrics['matched_ious']}")


if __name__ == "__main__":
    main()
