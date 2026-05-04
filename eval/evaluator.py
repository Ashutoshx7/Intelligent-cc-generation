"""IoU-based evaluation framework for CC suggestions."""
import json
import logging

logger = logging.getLogger(__name__)


def compute_temporal_iou(pred: dict, gt: dict) -> float:
    """
    Compute temporal Intersection-over-Union between two events.

    IoU = overlap_duration / union_duration

    Args:
        pred: Predicted event with start_time, end_time.
        gt: Ground truth event with start_time, end_time.

    Returns:
        IoU score between 0.0 and 1.0.
    """
    overlap_start = max(pred["start_time"], gt["start_time"])
    overlap_end = min(pred["end_time"], gt["end_time"])

    if overlap_end <= overlap_start:
        return 0.0

    overlap = overlap_end - overlap_start
    pred_dur = pred["end_time"] - pred["start_time"]
    gt_dur = gt["end_time"] - gt["start_time"]
    union = pred_dur + gt_dur - overlap

    return overlap / union if union > 0 else 0.0


def evaluate(predicted: list, ground_truth: list, iou_threshold: float = 0.5) -> dict:
    """
    Compute precision, recall, F1, and overcaption rate.

    The overcaption rate is the key metric for this tool — a high overcaption
    rate means the tool is suggesting too many unnecessary CCs, which defeats
    its purpose.

    Args:
        predicted: List of accepted CC events from pipeline.
        ground_truth: List of manually annotated events.
        iou_threshold: Minimum IoU to count as a true positive.

    Returns:
        Dict with precision, recall, f1, overcaption_rate, tp, fp, fn.
    """
    tp, fp = 0, 0
    matched_gt = set()

    for pred in predicted:
        best_iou = 0
        best_idx = -1
        for j, gt in enumerate(ground_truth):
            iou = compute_temporal_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_idx = j

        if best_iou >= iou_threshold and best_idx not in matched_gt:
            tp += 1
            matched_gt.add(best_idx)
        else:
            fp += 1  # Over-caption or duplicate

    fn = len(ground_truth) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    overcaption_rate = fp / (tp + fp) if (tp + fp) > 0 else 0.0

    results = {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "overcaption_rate": round(overcaption_rate, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "total_predicted": len(predicted),
        "total_ground_truth": len(ground_truth),
    }

    logger.info(f"Evaluation: P={precision:.3f} R={recall:.3f} "
                f"F1={f1:.3f} Overcaption={overcaption_rate:.3f}")
    return results


def load_ground_truth(path: str) -> list:
    """
    Load ground truth annotations from JSON.

    Expected format:
    [
        {"label": "gunshot", "start_time": 12.5, "end_time": 13.2},
        {"label": "glass breaking", "start_time": 28.3, "end_time": 28.9}
    ]
    """
    with open(path, 'r') as f:
        return json.load(f)
