import base64
import json
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import imageio
import numpy as np
import streamlit as st
import torch
from PIL import Image


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    data = []
    with open(file_path, "r") as file:
        for idx, line in enumerate(file):
            data.append(json.loads(line.strip()))
            if idx == 30000:
                break
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
            image, image_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
        )
    return image


def process_image(
    file_info: Dict[str, Any],
    file_info_2: Dict[str, Any],
    image_dir: str,
    confidence_threshold: float,
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

    combined_image = np.hstack(
        (pred_image_1, black_bar, pred_image_2, black_bar, target_image)
    )
    return combined_image


def find_idx_2(img_name: str, data_2: List[Dict[str, Any]]) -> int:
    for i in range(len(data_2)):
        if data_2[i]["img_name"] == img_name:
            return i
    return -1


def get_previous_2_frames(img_path: str, increment: int = 1) -> Tuple[str, str, str]:
    parent_dir = os.path.dirname(img_path)
    local_file_path = img_path.split("/")[-1]
    dataset_name = local_file_path.split("_")[0]
    cam = local_file_path.split("_")[1]
    frame_number = int(local_file_path.split("_")[-1].replace(".jpg", ""))
    prev_frame_1 = os.path.join(
        parent_dir, f"{dataset_name}_{cam}_frame_{frame_number - increment:04d}.jpg"
    )
    prev_frame_2 = os.path.join(
        parent_dir, f"{dataset_name}_{cam}_frame_{frame_number - 2 * increment:04d}.jpg"
    )
    return prev_frame_2, prev_frame_1, img_path

def get_frames(img_path: str) -> Tuple[str, str, str]:
    if "__" in img_path:
        parent_dir = os.path.dirname(img_path)
        local_file_path = img_path.split("/")[-1]
        extension = local_file_path.split(".")[-1]
        stem = local_file_path.split("__")[0]
        frame1 = os.path.join(parent_dir, f"{stem}__+0.{extension}")
        frame2 = os.path.join(parent_dir, f"{stem}__+1.{extension}")
        frame3 = os.path.join(parent_dir, f"{stem}__+2.{extension}")
        return frame1, frame2, frame3
    else:
        return img_path, img_path, img_path

def get_full_ground_truth(
    img_path1: str, img_path2: str, data: Dict[str, Any], img_dir: str
) -> Tuple[np.ndarray, List[np.ndarray]]:
    image_1 = cv2.imread(os.path.join(img_dir, img_path1))
    # image_1 = cv2.resize(image_1, (768, 768))
    image_2 = cv2.imread(os.path.join(img_dir, img_path2))
    # image_2 = cv2.resize(image_2, (768, 768))
    image_3 = cv2.imread(os.path.join(img_dir, data["img_name"]))
    # image_3 = cv2.resize(image_3, (768, 768))
    image_3 = draw_bounding_boxes(image_3, data["targets"], is_prediction=False)
    black_bar = np.ones((image_1.shape[0], 10, 3), dtype=np.uint8)

    combined_image = np.hstack((image_1, black_bar, image_2, black_bar, image_3))
    
    return combined_image, [image_1, image_2, image_3]

def display_gif(frames: List[np.ndarray]) -> None:
    """Display until the user clicks the Show Random Image button"""
    # Convert frames from BGR to RGB
    frames_rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
    
    # Convert frames to PIL images
    pil_images = [Image.fromarray(frame_rgb) for frame_rgb in frames_rgb]
    
    # Create an animated GIF in memory
    import io
    gif_bytes = io.BytesIO()
    pil_images[0].save(
        gif_bytes,
        format='GIF',
        save_all=True,
        append_images=pil_images[1:],
        loop=0,  # Loop indefinitely
        duration=500  # Duration between frames in milliseconds
    )
    gif_bytes.seek(0)
    
    # Encode the GIF in base64
    import base64
    gif_base64 = base64.b64encode(gif_bytes.read()).decode('utf-8')
    
    # Create an HTML element to display the GIF
    gif_html = f'<img src="data:image/gif;base64,{gif_base64}" alt="gif" />'
    
    # Display the GIF
    st.markdown(gif_html, unsafe_allow_html=True)


def main():
    
    st.set_page_config(layout="wide")
    st.title("Comparison of Two Prediction Sets and Ground Truth")

    input_file_1 = st.text_input(
        "Path to first input JSONL file",
        "/Users/derek/Desktop/cv-training/rt-detr/rt_detr/dv7_evaluated_maciullo_9.jsonl",
    )
    input_file_2 = st.text_input(
        "Path to second input JSONL file",
        "/Users/derek/Desktop/cv-training/rt-detr/rt_detr/dv11_eval_on_dv7_14.jsonl",
    )
    image_dir = st.text_input("Path to image directory", "/Users/derek/Desktop/drone_visualize/drone_v7_synthetic/valid/images")

    drone_tracking = st.checkbox("Drone Tracking", value=False)
    video_mode = st.checkbox("Video Mode", value=False)

    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)
    increment_frame = st.slider("Increment Frame", 1, 15, 1, 1)
    data_1 = load_jsonl(input_file_1)
    data_2 = load_jsonl(input_file_2)
    filter_conditions = {}

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
                
                if not os.path.exists(
                    os.path.join(image_dir, data_1[rand_idx]["img_name"])
                ):
                    continue
                if video_mode and not drone_tracking and "__" not in data_1[rand_idx]["img_name"]:
                    continue

                idx_2 = find_idx_2(data_1[rand_idx]["img_name"], data_2)
                print(f"idx_2: {idx_2}")
                if idx_2 == -1:
                    print(f"Could not find {data_1[rand_idx]['img_name']} in data_2")
                    continue
                else:
                    print(f"Found {data_1[rand_idx]['img_name']} in data_2\n")
                    print(f"idx_2: {idx_2}. Img: {data_2[idx_2]['img_name']}\n")

                if drone_tracking:
                    frames = get_previous_2_frames(
                        data_1[rand_idx]["img_name"], increment_frame
                    )
                else:
                    frames = get_frames(data_1[rand_idx]["img_name"])
                    
                print(f"prev_frame_1: {frames[0]}")
                print(f"prev_frame_2: {frames[1]}")

                if not os.path.exists(
                    os.path.join(image_dir, frames[0])
                ) or not os.path.exists(os.path.join(image_dir, frames[1])):
                    print(
                        f"Could not find previous frames for {frames[2]}"
                        f"File names: {frames[0]}, {frames[1]}, {frames[2]}"
                    )
                    continue
                gt_image, gt_frames = get_full_ground_truth(
                    frames[0], frames[1], data_1[rand_idx], image_dir
                )
                pred_image = cv2.imread(os.path.join(image_dir, frames[2]))
                pred_image = cv2.resize(pred_image, (768, 768))
                pred_image = cv2.cvtColor(pred_image, cv2.COLOR_BGR2RGB)

                predicted_image1 = draw_bounding_boxes(
                    pred_image.copy(),
                    data_1[rand_idx]["predictions"],
                    is_prediction=True,
                    confidence_threshold=confidence_threshold,
                )
                predicted_image2 = draw_bounding_boxes(
                    pred_image.copy(),
                    data_2[idx_2]["predictions"],
                    is_prediction=True,
                    confidence_threshold=confidence_threshold,
                )
                black_bar = np.ones((predicted_image1.shape[0], 10, 3), dtype=np.uint8)
                combined_predictions = np.hstack((predicted_image1, black_bar, predicted_image2))

                break

            # st.image(
            #     gt_image,
            #     channels="BGR",
            #     caption="Predictions Set 1 (left) vs Predictions Set 2 (middle) vs Ground Truth (right)",
            # )
            st.write(f"Image name: {data_1[rand_idx]['img_name']}")
            st.image(
                combined_predictions,
                caption="Predictions Set 1 (left) vs Predictions Set 2 (right)",
            )
            display_gif(gt_frames)


if __name__ == "__main__":
    main()
