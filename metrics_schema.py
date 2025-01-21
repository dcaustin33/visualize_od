from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class Metrics:
    tp: int
    fp: int
    fn: int
    loss: float

    unmatched_pred_areas: List[float]
    unmatched_pred_x_locations: List[float]
    unmatched_pred_y_locations: List[float]
    unmatched_gold_areas: List[float]
    unmatched_gold_x_locations: List[float]
    unmatched_gold_y_locations: List[float]

    matched_pred_areas: List[float]
    matched_pred_x_locations: List[float]
    matched_pred_y_locations: List[float]
    matched_gold_areas: List[float]
    matched_gold_x_locations: List[float]
    matched_gold_y_locations: List[float]

    matched_ious: List[float]
    matched_l1_errors: List[float]
    matched_classification_accuracy: List[bool]

    @property
    def precision(self):
        return self.tp / (self.tp + self.fp + 1e-6)

    @property
    def recall(self):
        return self.tp / (self.tp + self.fn + 1e-6)

    @property
    def f1(self):
        return (
            2 * (self.precision * self.recall) / (self.precision + self.recall + 1e-6)
        )

    @property
    def total_pred_boxes(self):
        return self.tp + self.fp

    @property
    def total_gold_boxes(self):
        return self.tp + self.fn

    def __add__(self, other: "Metrics") -> "Metrics":
        return Metrics(
            tp=self.tp + other.tp,
            fp=self.fp + other.fp,
            fn=self.fn + other.fn,
            loss=self.loss + other.loss,
            unmatched_pred_areas=self.unmatched_pred_areas + other.unmatched_pred_areas,
            unmatched_pred_x_locations=self.unmatched_pred_x_locations
            + other.unmatched_pred_x_locations,
            unmatched_pred_y_locations=self.unmatched_pred_y_locations
            + other.unmatched_pred_y_locations,
            unmatched_gold_areas=self.unmatched_gold_areas + other.unmatched_gold_areas,
            unmatched_gold_x_locations=self.unmatched_gold_x_locations
            + other.unmatched_gold_x_locations,
            unmatched_gold_y_locations=self.unmatched_gold_y_locations
            + other.unmatched_gold_y_locations,
            matched_pred_areas=self.matched_pred_areas + other.matched_pred_areas,
            matched_pred_x_locations=self.matched_pred_x_locations
            + other.matched_pred_x_locations,
            matched_pred_y_locations=self.matched_pred_y_locations
            + other.matched_pred_y_locations,
            matched_gold_areas=self.matched_gold_areas + other.matched_gold_areas,
            matched_gold_x_locations=self.matched_gold_x_locations
            + other.matched_gold_x_locations,
            matched_gold_y_locations=self.matched_gold_y_locations
            + other.matched_gold_y_locations,
            matched_ious=self.matched_ious + other.matched_ious,
            matched_l1_errors=self.matched_l1_errors + other.matched_l1_errors,
            matched_classification_accuracy=self.matched_classification_accuracy
            + other.matched_classification_accuracy,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "unmatched_pred_areas": self.unmatched_pred_areas,
            "unmatched_pred_x_locations": self.unmatched_pred_x_locations,
            "unmatched_pred_y_locations": self.unmatched_pred_y_locations,
            "unmatched_gold_areas": self.unmatched_gold_areas,
            "unmatched_gold_x_locations": self.unmatched_gold_x_locations,
            "unmatched_gold_y_locations": self.unmatched_gold_y_locations,
            "matched_pred_areas": self.matched_pred_areas,
            "matched_pred_x_locations": self.matched_pred_x_locations,
            "matched_pred_y_locations": self.matched_pred_y_locations,
            "matched_gold_areas": self.matched_gold_areas,
            "matched_gold_x_locations": self.matched_gold_x_locations,
            "matched_gold_y_locations": self.matched_gold_y_locations,
            "matched_ious": self.matched_ious,
            "matched_l1_errors": self.matched_l1_errors,
            "matched_classification_accuracy": self.matched_classification_accuracy,
            "f1": self.f1,
            "precision": self.precision,
            "recall": self.recall,
        }

    def wandb_loggable_output(
        self,
        split: str = "Train",
    ) -> Dict[str, Any]:
        epsilon = 1e-10  # Small value to avoid division by zero
        output = {
            "total_true_positives": self.tp,
            "total_false_positives": self.fp,
            "total_false_negatives": self.fn,
            "f1": self.f1,
            "precision": self.precision,
            "recall": self.recall,
            "average_unmatched_pred_area": sum(self.unmatched_pred_areas)
            / (len(self.unmatched_pred_areas) + epsilon),
            "average_unmatched_pred_x_locations": sum(self.unmatched_pred_x_locations)
            / (len(self.unmatched_pred_x_locations) + epsilon),
            "average_unmatched_pred_y_locations": sum(self.unmatched_pred_y_locations)
            / (len(self.unmatched_pred_y_locations) + epsilon),
            "average_unmatched_gold_areas": sum(self.unmatched_gold_areas)
            / (len(self.unmatched_gold_areas) + epsilon),
            "average_unmatched_gold_x_locations": sum(self.unmatched_gold_x_locations)
            / (len(self.unmatched_gold_x_locations) + epsilon),
            "average_unmatched_gold_y_locations": sum(self.unmatched_gold_y_locations)
            / (len(self.unmatched_gold_y_locations) + epsilon),
            "average_matched_pred_areas": sum(self.matched_pred_areas)
            / (len(self.matched_pred_areas) + epsilon),
            "average_matched_pred_x_locations": sum(self.matched_pred_x_locations)
            / (len(self.matched_pred_x_locations) + epsilon),
            "average_matched_pred_y_locations": sum(self.matched_pred_y_locations)
            / (len(self.matched_pred_y_locations) + epsilon),
            "average_matched_gold_areas": sum(self.matched_gold_areas)
            / (len(self.matched_gold_areas) + epsilon),
            "average_matched_gold_x_locations": sum(self.matched_gold_x_locations)
            / (len(self.matched_gold_x_locations) + epsilon),
            "average_matched_gold_y_locations": sum(self.matched_gold_y_locations)
            / (len(self.matched_gold_y_locations) + epsilon),
            "average_matched_ious": sum(self.matched_ious)
            / (len(self.matched_ious) + epsilon),
            "average_matched_l1_errors": sum(self.matched_l1_errors)
            / (len(self.matched_l1_errors) + epsilon),
            "average_matched_classification_accuracy": sum(
                self.matched_classification_accuracy
            )
            / (len(self.matched_classification_accuracy) + epsilon),
            "total_pred_boxes": self.total_pred_boxes,
            "total_gold_boxes": self.total_gold_boxes,
            "total_divisors": 1,
        }
        return {f"{split}_{k}": v for k, v in output.items()}

    def add_wandb_outputs(
        self, other_wandb_output: Dict[str, Any], split: str = "train"
    ) -> Dict[str, Any]:
        wandb_output = self.wandb_loggable_output(split)
        divisor = (
            wandb_output[f"{split}_total_divisors"]
            + other_wandb_output[f"{split}_total_divisors"]
        )
        for key in wandb_output:
            if "total" not in key:
                wandb_output[key] = (
                    wandb_output[key]  + other_wandb_output[key] * (divisor - 1)
                ) / divisor
            else:
                wandb_output[key] += other_wandb_output[key]
        return wandb_output

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Metrics":
        return cls(
            tp=data["tp"],
            fp=data["fp"],
            fn=data["fn"],
            unmatched_pred_areas=data["unmatched_pred_areas"],
            unmatched_pred_x_locations=data["unmatched_pred_x_locations"],
            unmatched_pred_y_locations=data["unmatched_pred_y_locations"],
            unmatched_gold_areas=data["unmatched_gold_areas"],
            unmatched_gold_x_locations=data["unmatched_gold_x_locations"],
            unmatched_gold_y_locations=data["unmatched_gold_y_locations"],
            matched_pred_areas=data["matched_pred_areas"],
            matched_pred_x_locations=data["matched_pred_x_locations"],
            matched_pred_y_locations=data["matched_pred_y_locations"],
            matched_gold_areas=data["matched_gold_areas"],
            matched_gold_x_locations=data["matched_gold_x_locations"],
            matched_gold_y_locations=data["matched_gold_y_locations"],
            matched_ious=data["matched_ious"],
            matched_l1_errors=data["matched_l1_errors"],
            matched_classification_accuracy=data["matched_classification_accuracy"],
        )