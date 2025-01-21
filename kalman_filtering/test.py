import json
import os

import cv2
import numpy as np
from object_tracking.kalman_filter.schema_n_adaptive_q import (
    KalmanNDTrackerAdaptiveQ,
    KalmanStateVectorNDAdaptiveQ,
)
from tqdm import tqdm

STATE = KalmanStateVectorNDAdaptiveQ
TRACKER = KalmanNDTrackerAdaptiveQ


def strip_image_names_into_tuples(image_names: list[str]):
    """
    Strips image names into tuples of (dataset_name, camera_number, frame_number)
    """
    outputs = []
    for image_name in image_names:
        dataset_name = image_name.split("_")[0]
        camera_number = image_name.split("_")[1]
        frame_number = int(image_name.split("_")[-1].split(".")[0])
        outputs.append((image_name, dataset_name, camera_number, frame_number))
    # Sort by frame number
    return outputs


def filter_image_names_by_dataset_and_camera(
    image_names: list[tuple[str, str, str, int]], dataset_name: str, camera_number: str
):
    """Filters image names by dataset and camera number - sorts by frame number"""
    filtered = [
        image_name
        for image_name in image_names
        if image_name[1] == dataset_name and image_name[2] == camera_number
    ]
    return sorted(filtered, key=lambda x: x[3])


def read_jsonl(path: str) -> list[dict]:
    with open(path, "r") as file:
        return [json.loads(line) for line in file]


def update_results_with_dataset_info(data: list[dict]):
    for d in data:
        d["dataset_name"] = d["img_name"].split("_")[0]
        d["camera_number"] = d["img_name"].split("_")[1]
        d["frame_number"] = int(d["img_name"].split("_")[-1].split(".")[0])
    return data


def filter_and_sort_results(data: list[dict], dataset_name: str, camera_number: int):
    """Filters to a specific dataset and camera number and then sorts by frame number"""
    filtered = [
        d
        for d in data
        if d["dataset_name"] == dataset_name and d["camera_number"] == camera_number
    ]
    return sorted(filtered, key=lambda x: x["frame_number"])


def find_data_from_image_name(data: list[dict], image_name: str):
    return next((d for d in data if d["img_name"] == image_name), None)


def associate_objects(
    tracked_objects: list[tuple[KalmanNDTrackerAdaptiveQ, int]],
    detected_object_measurements: list[np.array],
    frame_number: int,
    p_value_threshold: float = 0.3,
    frame_threshold: int = 30,
    R: float = 3,
    Q: float = 3,
    h_matrix: np.array = None,
    stop: bool = False,
):
    """Associates detected objects with existing tracked objects using Mahalanobis distance.

    Creates new tracked objects for unmatched detections. Removes tracked objects not seen
    for frame_threshold frames.

    Args:
        tracked_objects: List of tuples containing (KalmanTracker, last_seen_frame)
        detected_object_measurements: List of measurement arrays for detected objects
        frame_number: Current frame number
        p_value_threshold: Threshold for associating detections with tracked objects
        frame_threshold: Number of frames before removing unmatched tracked objects
        R: Measurement noise parameter
        Q: Process noise parameter
        h_matrix: Optional measurement matrix

    Returns:
        Tuple containing:
        - Dictionary mapping tracked object indices to detection indices
        - List of remaining tracked objects
        - List of new tracked objects
    """
    associations = {}
    for i, (tracked_object, last_seen_frame) in enumerate(tracked_objects):
        if last_seen_frame + frame_threshold < frame_number:
            tracked_objects.pop(i)

    new_objects = []

    # takes care off the empty current objects case
    if len(tracked_objects) == 0:
        for detected_object_measurement in detected_object_measurements:
            vector = np.concatenate(
                [
                    detected_object_measurement,
                    np.zeros(detected_object_measurement.shape[0]),
                ]
            )
            state_vector = STATE(states=vector)
            new_objects.append(
                [
                    TRACKER(state=state_vector, R=R, Q=Q, h=h_matrix),
                    frame_number,
                ]
            )
        return {}, [], new_objects

    # predicts the next state to use for association
    for i, (tracked_object, last_seen_frame) in enumerate(tracked_objects):
        tracked_object.predict(dt=1)

    for i, detected_object_measurement in enumerate(detected_object_measurements):
        max_p_value = -np.inf
        max_idx = None
        for j, (tracked_object, _) in enumerate(tracked_objects):
            p_value = tracked_object.compute_p_value_from_measurement(
                detected_object_measurement
            )
            if p_value > max_p_value:
                max_p_value = p_value
                max_idx = j
        if max_p_value > p_value_threshold and max_idx not in associations:
            associations[max_idx] = i
        elif max_p_value > p_value_threshold and max_idx in associations:
            # something else has already taken it but it is pretty confident
            # lets not make a new object
            continue
        else:
            # now we have to initialize a new object
            # concat velocities of all the measurements to 0
            vector = np.concatenate(
                [
                    detected_object_measurement,
                    np.zeros(detected_object_measurement.shape[0]),
                ]
            )
            state_vector = STATE(states=vector)
            new_object = TRACKER(state=state_vector, R=R, Q=Q, h=h_matrix)
            new_objects.append([new_object, frame_number])

    if stop:
        import pdb; pdb.set_trace()

    return associations, tracked_objects, new_objects


def plot_objects(
    tracked_objects: list[tuple[KalmanNDTrackerAdaptiveQ, int]],
    current_measurements: list[np.array],
    image: np.array,
):
    for i, measurement in enumerate(current_measurements):
        cx, cy = measurement[:2].copy()
        cx *= image.shape[1]
        cy *= image.shape[0]
        cv2.circle(image, (int(cx), int(cy)), 10, (0, 255, 0), -1)
    for i, (tracked_object, _) in enumerate(tracked_objects):
        cx, cy = tracked_object.previous_measurements[-1][:2].copy()
        cx *= image.shape[1]
        cy *= image.shape[0]

        cv2.circle(image, (int(cx), int(cy)), 5, (0, 0, 255), -1)

    return image


if __name__ == "__main__":
    path = "/Users/derek/Desktop/drone_visualize/dv7_eval_dtd_dataset3_cam1.jsonl"
    images_path = "/Users/derek/Desktop/drone_visualize/dtd_dataset3_cam1/train/images"
    dataset_name = "dataset3"
    camera_number = "cam1"

    data = read_jsonl(path)
    data = update_results_with_dataset_info(data)
    filtered_data = filter_and_sort_results(data, dataset_name, camera_number)

    images = os.listdir(images_path)
    image_names = strip_image_names_into_tuples(images)
    filtered_image_names = filter_image_names_by_dataset_and_camera(
        image_names, dataset_name, camera_number
    )

    current_objects = []
    h_matrix = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ]
    )

    image = cv2.imread(os.path.join(images_path, filtered_image_names[0][0]))
    height, width, _ = image.shape
    output_video = cv2.VideoWriter(
        "output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (width, height)
    )

    for idx, image_name in tqdm(enumerate(filtered_image_names)):
        image_path = os.path.join(images_path, image_name[0])

        image = cv2.imread(image_path)
        data = find_data_from_image_name(filtered_data, image_name[0])
        if data is None:
            continue
        confidences = np.array(data["predictions"]["confidences"])
        confidences_truth = confidences > 0.1
        measurements = np.array(data["predictions"]["bboxes"])[confidences_truth]
        

        associations, current_objects, new_objects = associate_objects(
            current_objects, measurements, image_name[3], h_matrix=h_matrix
        )

        for key, value in associations.items():
            current_objects[key][0].update(measurements[value], predict=False)
            current_objects[key][1] = image_name[3]

        current_objects.extend(new_objects)

        image = plot_objects(current_objects, measurements, image)
        output_video.write(image)

    output_video.release()
