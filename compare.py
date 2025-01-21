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
    file_info: Dict[str, Any],
    file_info_2: Dict[str, Any],
    image_dir: str,
    confidence_threshold: float
) -> np.ndarray:
    img_path = os.path.join(image_dir, file_info["img_name"])
    predictions_1 = file_info["predictions"]
    predictions_2 = file_info_2["predictions"]
    targets = file_info["targets"]

    image = cv2.imread(img_path)
    if image is None:
        raise ValueError(f"Image at path {img_path} could not be loaded.")

    pred_image_1 = image.copy()
    pred_image_2 = image.copy()
    target_image = image.copy()

    pred_image_1 = draw_bounding_boxes(
        pred_image_1,
        predictions_1,
        is_prediction=True,
        confidence_threshold=confidence_threshold,
    )
    pred_image_2 = draw_bounding_boxes(
        pred_image_2,
        predictions_2,
        is_prediction=True,
        confidence_threshold=confidence_threshold,
    )
    target_image = draw_bounding_boxes(
        target_image,
        targets,
        is_prediction=False,
    )
    black_bar = np.ones((image.shape[0], 10, 3), dtype=np.uint8)

    combined_image = np.hstack((pred_image_1, black_bar, pred_image_2, black_bar, target_image))
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

def find_idx_2(img_name: str, data_2: List[Dict[str, Any]]) -> int:
    for i in range(len(data_2)):
        if data_2[i]["img_name"] == img_name:
            return i
    return -1

def main():
    st.set_page_config(layout="wide")
    st.title("Comparison of Two Prediction Sets and Ground Truth")

    input_file_1 = st.text_input(
        "Path to first input JSONL file",
        "dv6_normal_r34.jsonl",
    )
    input_file_2 = st.text_input(
        "Path to second input JSONL file",
        "dv6_synth_r34.jsonl",
    )
    image_dir = st.text_input(
        "Path to image directory",
        "drone_v6_normal/valid/images"
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
    data_1 = load_jsonl(input_file_1)
    data_2 = load_jsonl(input_file_2)
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
        if not os.path.exists(input_file_1) or not os.path.exists(input_file_2):
            st.error("One or both input JSONL files do not exist.")
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
                rand_idx = random.randint(0, min(len(data_1), len(data_2)) - 1)
                st.write(f"Image name: {data_1[rand_idx]['img_name']}")
                if not os.path.exists(
                    os.path.join(image_dir, data_1[rand_idx]["img_name"])
                ):
                    continue
                metrics_1 = box_metrics(
                    preds_logits=torch.tensor(
                        data_1[rand_idx]["predictions"]["confidences"]
                    ).view(-1, 1),
                    preds_boxes=torch.tensor(data_1[rand_idx]["predictions"]["bboxes"]),
                    target_classes=torch.tensor(data_1[rand_idx]["targets"]["class_ids"]),
                    target_boxes=torch.tensor(data_1[rand_idx]["targets"]["bboxes"]),
                    matcher=matcher,
                    loss_value=0,
                    confidence_threshold=confidence_threshold,
                    classes=80
                ).to_dict()

                idx_2 = find_idx_2(data_1[rand_idx]["img_name"], data_2)
                if idx_2 == -1:
                    print(f"Could not find {data_1[rand_idx]['img_name']} in data_2")
                    continue
                else:
                    print(f"Found {data_1[rand_idx]['img_name']} in data_2")
                    print(f"idx_2: {idx_2}. Img: {data_2[idx_2]['img_name']}")
                    metrics_2 = box_metrics(
                        preds_logits=torch.tensor(
                        data_2[idx_2]["predictions"]["confidences"]
                    ).view(-1, 1),
                    preds_boxes=torch.tensor(data_2[idx_2]["predictions"]["bboxes"]),
                    target_classes=torch.tensor(data_2[idx_2]["targets"]["class_ids"]),
                    target_boxes=torch.tensor(data_2[idx_2]["targets"]["bboxes"]),
                    matcher=matcher,
                    loss_value=0,
                    confidence_threshold=confidence_threshold,
                    classes=80
                ).to_dict()

                # check if there are any conditions
                if filter_conditions:
                    if filter(metrics_1, **filter_conditions) or filter(metrics_2, **filter_conditions):
                        break
                else:
                    break
            combined_image = process_image(
                data_1[rand_idx], data_2[idx_2], image_dir, confidence_threshold
            )

            st.image(
                combined_image,
                channels="BGR",
                caption="Predictions Set 1 (left) vs Predictions Set 2 (middle) vs Ground Truth (right)",
            )
            st.write("Metrics for Prediction Set 1:")
            st.write(f"F1: {metrics_1['f1']}")
            st.write(f"FP: {metrics_1['fp']}")
            st.write(f"FN: {metrics_1['fn']}")
            st.write(f"TP: {metrics_1['tp']}")
            st.write(f"IOU: {metrics_1['matched_ious']}")
            
            st.write("Metrics for Prediction Set 2:")
            st.write(f"F1: {metrics_2['f1']}")
            st.write(f"FP: {metrics_2['fp']}")
            st.write(f"FN: {metrics_2['fn']}")
            st.write(f"TP: {metrics_2['tp']}")
            st.write(f"IOU: {metrics_2['matched_ious']}")


if __name__ == "__main__":
    main()
