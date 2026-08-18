import argparse
import csv
import time
from pathlib import Path

import numpy as np

from inference import localize, read_grayscale


def read_label(label_path):
    values = {}

    with open(label_path, "r", encoding="utf-8") as file:
        for line in file:
            key, value = line.strip().split("=")
            values[key] = int(value)

    return values["x"], values["y"]


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate inference.py on a generated dataset."
    )

    parser.add_argument(
        "--dataset-dir",
        default="data/final_dataset",
        help="Directory containing reference, search, and labels folders."
    )

    parser.add_argument(
        "--output-csv",
        default="results/final_dataset_evaluation.csv",
        help="Output CSV path."
    )

    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    reference_dir = dataset_dir / "reference"
    search_dir = dataset_dir / "search"
    label_dir = dataset_dir / "labels"

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if not reference_dir.exists():
        raise FileNotFoundError(
            f"Reference folder not found: {reference_dir}"
        )

    reference_files = sorted(reference_dir.glob("*.png"))

    if not reference_files:
        raise RuntimeError(
            f"No PNG reference files found in: {reference_dir}"
        )

    results = []

    for reference_path in reference_files:
        pair_id = reference_path.stem
        search_path = search_dir / f"{pair_id}.png"
        label_path = label_dir / f"{pair_id}.txt"

        if not search_path.exists():
            raise FileNotFoundError(
                f"Missing search image: {search_path}"
            )

        if not label_path.exists():
            raise FileNotFoundError(
                f"Missing label: {label_path}"
            )

        reference = read_grayscale(str(reference_path))
        search = read_grayscale(str(search_path))
        true_x, true_y = read_label(label_path)

        start_time = time.perf_counter()

        prediction = localize(reference, search)

        runtime_ms = (
            time.perf_counter() - start_time
        ) * 1000.0

        error_pixels = float(np.sqrt(
            (prediction["x"] - true_x) ** 2
            + (prediction["y"] - true_y) ** 2
        ))

        item = {
            "pair_id": pair_id,
            "true_x": true_x,
            "true_y": true_y,
            "predicted_x": prediction["x"],
            "predicted_y": prediction["y"],
            "error_pixels": round(error_pixels, 3),
            "runtime_ms": round(runtime_ms, 3),
            "best_score": prediction["best_score"],
            "ambiguity_ratio": prediction["ambiguity_ratio"],
            "near_best_candidate_count": (
                prediction["near_best_candidate_count"]
            ),
            "selected_scale_multiplier": (
                prediction["selected_scale_multiplier"]
            ),
            "selected_angle_degrees": (
                prediction["selected_angle_degrees"]
            ),
            "template_width": prediction["template_width"],
            "template_height": prediction["template_height"]
        }

        results.append(item)

        print(
            f"{pair_id}: "
            f"truth=({true_x}, {true_y}) "
            f"prediction=({prediction['x']}, {prediction['y']}) "
            f"error={error_pixels:.2f}px "
            f"time={runtime_ms:.1f}ms "
            f"scale={prediction['selected_scale_multiplier']} "
            f"angle={prediction['selected_angle_degrees']}"
        )

    with open(
        output_csv,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys()
        )

        writer.writeheader()
        writer.writerows(results)

    errors = np.array(
        [item["error_pixels"] for item in results],
        dtype=np.float32
    )

    runtimes = np.array(
        [item["runtime_ms"] for item in results],
        dtype=np.float32
    )

    accuracy_5 = float(np.mean(errors <= 5.0) * 100.0)
    accuracy_10 = float(np.mean(errors <= 10.0) * 100.0)
    accuracy_25 = float(np.mean(errors <= 25.0) * 100.0)

    print()
    print("========== FINAL DATASET EVALUATION ==========")
    print("Number of pairs:", len(results))
    print("Accuracy within 5 pixels:", round(accuracy_5, 2), "%")
    print("Accuracy within 10 pixels:", round(accuracy_10, 2), "%")
    print("Accuracy within 25 pixels:", round(accuracy_25, 2), "%")
    print(
        "Mean localization error:",
        round(float(np.mean(errors)), 2),
        "pixels"
    )
    print(
        "Median localization error:",
        round(float(np.median(errors)), 2),
        "pixels"
    )
    print(
        "Maximum localization error:",
        round(float(np.max(errors)), 2),
        "pixels"
    )
    print(
        "Mean runtime:",
        round(float(np.mean(runtimes)), 2),
        "ms per pair"
    )
    print(
        "Maximum runtime:",
        round(float(np.max(runtimes)), 2),
        "ms"
    )
    print("Results CSV saved to:")
    print(output_csv)


if __name__ == "__main__":
    main()
