import random
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from metrics_schema import Metrics


def calculate_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters:
    - box1, box2: Lists or tuples with 4 elements [x1, y1, x2, y2]
      where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner.

    Returns:
    - iou: Intersection over Union (IoU) value.
    """
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2

    # Calculate the coordinates of the intersection rectangle
    inter_x1 = max(x1, x1_p)
    inter_y1 = max(y1, y1_p)
    inter_x2 = min(x2, x2_p)
    inter_y2 = min(y2, y2_p)

    # Calculate the area of the intersection rectangle
    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    inter_area = inter_width * inter_height

    # Calculate the area of both the predicted and ground truth rectangles
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_p - x1_p) * (y2_p - y1_p)

    # Calculate the union area
    union_area = box1_area + box2_area - inter_area

    # Calculate the IoU
    iou = inter_area / union_area if union_area != 0 else 0

    return iou.item()

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def get_color(label):
    random.seed(label)
    return tuple(random.choices(range(256), k=3))


def annotate_bounding_boxes(
    image: Image.Image, boxes: torch.Tensor, labels: Optional[torch.Tensor] = None
) -> Image.Image:
    """
    Function to annotate the bounding boxes with the class labels.

    Args:
        image (PIL.Image.Image): The image to annotate.
        boxes (torch.Tensor): The boxes to annotate (num_boxes, 4).
        labels (Optional[torch.Tensor]): The labels to annotate the boxes with (num_boxes,).

    Returns:
        PIL.Image.Image: The annotated image.
    """
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    label_colors = {}
    for i, box in enumerate(boxes):
        x_min, y_min, x_max, y_max = box
        label = labels[i] if labels is not None else None
        label = (
            label.item() if label is not None and type(label) is torch.Tensor else label
        )

        if label is not None:
            if label not in label_colors:
                label_colors[label] = get_color(label)
            color = label_colors[label]
        else:
            color = "red"

        draw.rectangle([(x_min, y_min), (x_max, y_max)], outline=color, width=2)
        if labels is not None:
            draw.text((x_min, y_min), str(label), fill=color, font=font)

    return image


def extract_relevant_boxes(
    query_logits: torch.Tensor,
    query_boxes: torch.Tensor,
    confidence_threshold: float = 0.5,
    softmax: bool = False,
    need_norm: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Function to take in the logits and return the relevant boxes
    and logits that are above the confidence threshold.

    Args:
        query_logits (torch.Tensor): The logits from the model (num_queries, num_classes)
        query_boxes (torch.Tensor): The boxes from the model (num_queries, 4)
        confidence_threshold (float): The confidence threshold to filter out the boxes
        softmax (bool): Whether to apply softmax to the logits or not. If False we apply sigmoid.
            This has to do with using the focal loss
        need_norm (bool): Whether to use softmax or sigmoid
    """
    if need_norm:
        if softmax:
            probs = torch.softmax(query_logits, -1)
        else:
            probs = torch.sigmoid(query_logits)
    else:
        probs = query_logits

    scores, labels = torch.max(probs, -1)
    mask = scores > confidence_threshold
    relevant_boxes = query_boxes[mask]
    corresponding_logits = scores[mask]
    all_logits = query_logits[mask]
    relevant_labels = labels[mask]
    # filter out the zeros
    return relevant_boxes, corresponding_logits, relevant_labels, all_logits


def box_metrics(
    preds_logits: torch.Tensor,
    preds_boxes: torch.Tensor,
    target_classes: torch.Tensor,
    target_boxes: torch.Tensor,
    matcher: torch.nn.Module,
    loss_value: float,
    confidence_threshold: float = 0.5,
):
    """
    Input should be from one image, we will output an object detailing the boxes
    that are missed.

    Args:
        preds_logits (torch.Tensor): The logits from the model (num_queries, num_classes)
        preds_boxes (torch.Tensor): The boxes from the model (num_queries, 4)
        target_classes (torch.Tensor): The target classes for the image (num_target_boxes,)
        target_boxes (torch.Tensor): The target boxes for the image (num_target_boxes, 4)
        matcher (torch.nn.Module): The matcher to use to match the boxes
        confidence_threshold (float): The confidence threshold to filter out the boxes
    """
    pred_boxes, pred_logits, pred_labels, all_logits = extract_relevant_boxes(
        preds_logits.detach(), preds_boxes.detach(), confidence_threshold
    )
    pred_empty = len(pred_boxes) == 0
    pred_input = {
        "pred_boxes": pred_boxes.unsqueeze(0),
        "pred_logits": all_logits.unsqueeze(0),
    }
    target = {
        "labels": target_classes,
        "boxes": target_boxes,
    }
    if len(pred_boxes) == 0:
        matches = [[[], []]]
    else:
        matches = matcher(pred_input, [target])
    all_preds = set(i for i in range(pred_logits.shape[0]))
    all_golds = set(range(len(target_classes)))

    # get unmatched boxes statistics
    if pred_empty:
        unmatched_preds = []
        unmatched_gold = all_golds
    else:
        unmatched_preds = all_preds - set(matches[0][0].tolist())
        unmatched_gold = all_golds - set(matches[0][1].tolist())
    unmatched_pred_area = []
    unmatched_pred_location = []
    unmatched_gold_area = []
    unmatched_gold_location = []

    for box in unmatched_preds:
        unmatched_pred_area.append(
            (pred_boxes[box][2] * pred_boxes[box][3] * 100).item()
        )
        unmatched_pred_location.append((pred_boxes[box][:2] * 100))

    for box in unmatched_gold:
        unmatched_gold_area.append(
            (target_boxes[box][2] * target_boxes[box][3] * 100).item()
        )
        unmatched_gold_location.append((target_boxes[box][:2] * 100))

    # get matched boxes statistics
    matched_pred_area = []
    matched_pred_location = []
    matched_gold_area = []
    matched_gold_location = []
    for box_idx in range(len(matches[0][0])):
        box = matches[0][0][box_idx]
        matched_pred_area.append((pred_boxes[box][2] * pred_boxes[box][3] * 100).item())
        matched_pred_location.append((pred_boxes[box][:2] * 100))

    for box_idx in range(len(matches[0][1])):
        box = matches[0][1][box_idx]
        matched_gold_area.append(target_boxes[box][2] * target_boxes[box][3] * 100)
        matched_gold_location.append(target_boxes[box][:2] * 100)

    # get the iou of the matched boxes
    matched_ious = []
    matched_l1_errors = []
    matched_classification_accuracy = []
    for pred_box, gold_box in zip(matches[0][0], matches[0][1]):
        xyxy = box_cxcywh_to_xyxy(
            pred_boxes[pred_box],
        )
        xyxy_target = box_cxcywh_to_xyxy(target_boxes[gold_box])
        matched_ious.append(calculate_iou(xyxy, xyxy_target))
        matched_l1_errors.append(
            (
                torch.abs(pred_boxes[pred_box][:2] - target_boxes[gold_box][:2]).sum()
            ).item()
        )
        matched_classification_accuracy.append(
            (pred_labels[pred_box] == target_classes[gold_box]).item()
        )
    tp = sum([1 if i > 0.3 else 0 for i in matched_ious])
    fn = len(unmatched_gold) + sum([1 for i in matched_ious if i <= 0.3])
    fp = len(unmatched_preds) + sum([1 for i in matched_ious if i <= 0.3])
    return Metrics(
        tp=tp,
        fp=fp,
        fn=fn,
        loss=loss_value,
        unmatched_pred_areas=unmatched_pred_area,
        unmatched_pred_x_locations=[box[0].item() for box in unmatched_pred_location],
        unmatched_pred_y_locations=[box[1].item() for box in unmatched_pred_location],
        unmatched_gold_areas=unmatched_gold_area,
        unmatched_gold_x_locations=[box[0].item() for box in unmatched_gold_location],
        unmatched_gold_y_locations=[box[1].item() for box in unmatched_gold_location],
        matched_pred_areas=matched_pred_area,
        matched_pred_x_locations=[box[0].item() for box in matched_pred_location],
        matched_pred_y_locations=[box[1].item() for box in matched_pred_location],
        matched_gold_areas=matched_gold_area,
        matched_gold_x_locations=[box[0].item() for box in matched_gold_location],
        matched_gold_y_locations=[box[1].item() for box in matched_gold_location],
        matched_ious=matched_ious,
        matched_l1_errors=matched_l1_errors,
        matched_classification_accuracy=matched_classification_accuracy,
    )
