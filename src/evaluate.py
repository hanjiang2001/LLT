import json

import numpy as np
import pandas as pd
from tabulate import tabulate


def segment_iou(target_segment, candidate_segments):
    start = np.maximum(target_segment[0], candidate_segments[:, 0])
    end = np.minimum(target_segment[1], candidate_segments[:, 1])
    intersection = (end - start).clip(0)
    union = (
        candidate_segments[:, 1]
        - candidate_segments[:, 0]
        + (target_segment[1] - target_segment[0])
        - intersection
    )
    return intersection.astype(float) / union


def interpolated_precision_recall(precision, recall):
    precision = np.hstack([[0], precision, [0]])
    recall = np.hstack([[0], recall, [1]])
    for index in range(len(precision) - 1)[::-1]:
        precision[index] = max(precision[index], precision[index + 1])
    indices = np.where(recall[1:] != recall[:-1])[0] + 1
    return np.sum((recall[indices] - recall[indices - 1]) * precision[indices])


def compute_average_precision(ground_truth, prediction, thresholds):
    average_precision = np.zeros(len(thresholds))
    if prediction.empty or ground_truth.empty:
        return average_precision, np.empty(0)
    matched = np.full((len(thresholds), len(ground_truth)), -1)
    prediction = prediction.loc[prediction["score"].values.argsort()[::-1]].reset_index(drop=True)
    true_positive = np.zeros((len(thresholds), len(prediction)))
    false_positive = np.zeros_like(true_positive)
    groups = ground_truth.groupby("video-id")
    overlaps = []
    for index, proposal in prediction.iterrows():
        if proposal["video-id"] not in groups.groups:
            false_positive[:, index] = 1
            continue
        targets = groups.get_group(proposal["video-id"]).reset_index()
        ious = segment_iou(
            proposal[["t-start", "t-end"]].values, targets[["t-start", "t-end"]].values
        )
        overlaps.extend(ious)
        for threshold_index, threshold in enumerate(thresholds):
            for target_index in ious.argsort()[::-1]:
                if ious[target_index] < threshold:
                    false_positive[threshold_index, index] = 1
                    break
                original_index = targets.loc[target_index, "index"]
                if matched[threshold_index, original_index] >= 0:
                    continue
                true_positive[threshold_index, index] = 1
                matched[threshold_index, original_index] = index
                break
            if (
                false_positive[threshold_index, index] == 0
                and true_positive[threshold_index, index] == 0
            ):
                false_positive[threshold_index, index] = 1
    true_positive = np.cumsum(true_positive, axis=1).astype(float)
    false_positive = np.cumsum(false_positive, axis=1).astype(float)
    recall = true_positive / len(ground_truth)
    precision = true_positive / (true_positive + false_positive)
    for index in range(len(thresholds)):
        average_precision[index] = interpolated_precision_recall(precision[index], recall[index])
    return average_precision, np.asarray(overlaps)


def evaluate(dataset, predictions, class_names, annotations_path, fps_path, output_dir):
    thresholds = (
        np.array([0.3, 0.4, 0.5, 0.6, 0.7]) if dataset == "thumos" else np.array([0.5, 0.75, 0.95])
    )
    with open(annotations_path, encoding="utf-8") as stream:
        annotations = json.load(stream)
    with open(fps_path, encoding="utf-8") as stream:
        fps_data = json.load(stream)
    ground_truth_rows = []
    prediction_rows = []
    for video_name, proposals in predictions.items():
        for annotation in annotations[video_name]["annotations"]:
            if annotation["label"] in class_names:
                ground_truth_rows.append(
                    {
                        "video-id": video_name,
                        "t-start": float(annotation["segment"][0]),
                        "t-end": float(annotation["segment"][1]),
                        "label": annotation["label"],
                    }
                )
        fps = next(
            (
                fps_data[f"{video_name}{extension}"]
                for extension in [".mp4", ".mkv", ".webm"]
                if f"{video_name}{extension}" in fps_data
            ),
            30,
        )
        for proposal in proposals:
            prediction_rows.append(
                {
                    "video-id": video_name,
                    "t-start": proposal["segment"][0] / fps,
                    "t-end": proposal["segment"][1] / fps,
                    "score": proposal["score"],
                    "label": class_names[proposal["label"]],
                }
            )
    ground_truth = pd.DataFrame(
        ground_truth_rows, columns=["video-id", "t-start", "t-end", "label"]
    )
    prediction = pd.DataFrame(
        prediction_rows, columns=["video-id", "t-start", "t-end", "score", "label"]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    ground_truth.to_csv(output_dir / "ground_truth.csv", index=False)
    prediction.to_csv(output_dir / "predicted.csv", index=False)
    class_ap = np.zeros((len(class_names), len(thresholds)))
    all_ious = []
    for index, name in enumerate(class_names):
        targets = ground_truth[ground_truth["label"] == name].reset_index(drop=True)
        proposals = prediction[prediction["label"] == name].reset_index(drop=True)
        average_precision, ious = compute_average_precision(targets, proposals, thresholds)
        class_ap[index] = [round(value * 100, 1) for value in average_precision]
        all_ious.extend(ious)
    headers = [f"AP@{threshold:g}" for threshold in thresholds]
    class_results = pd.DataFrame(class_ap, index=class_names, columns=headers)
    class_results.index.name = "Class"
    class_results.to_csv(output_dir / "class_AP.csv")
    print(tabulate(class_results.reset_index(), headers="keys", tablefmt="github", showindex=False))
    mean_ap = class_ap.mean(axis=0)
    metrics = {f"AP@{threshold:g}": float(value) for threshold, value in zip(thresholds, mean_ap)}
    metrics["avg_AP"] = float(mean_ap.mean())
    metrics["average_tIoU"] = float(np.mean(all_ious)) if all_ious else 0.0
    print(json.dumps(metrics, indent=2))
    return metrics
