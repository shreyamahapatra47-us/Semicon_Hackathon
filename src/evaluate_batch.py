import os
import csv
import time
import cv2
import numpy as np


PROJECT_FOLDER = r"C:\Users\Shreya\OneDrive\Desktop\drift_sense_hackathon"

REFERENCE_FOLDER = os.path.join(PROJECT_FOLDER, "data", "reference")
SEARCH_FOLDER = os.path.join(PROJECT_FOLDER, "data", "search")
LABEL_FOLDER = os.path.join(PROJECT_FOLDER, "data", "labels")
RESULTS_FOLDER = os.path.join(PROJECT_FOLDER, "results")

os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Keep the same robust settings that worked for pair_002.
SCALES_TO_TRY = [0.96, 0.98, 1.00, 1.02, 1.04]
ANGLES_TO_TRY = [-1.0, -0.5, 0.0, 0.5, 1.0]


def normalize_image(image):
    image = image.astype(np.float32)
    image = image - np.mean(image)

    standard_deviation = np.std(image)

    if standard_deviation > 1e-6:
        image = image / standard_deviation

    return image


def read_label(label_path):
    values = {}

    with open(label_path, "r", encoding="utf-8") as file:
        for line in file:
            key, value = line.strip().split("=")
            values[key] = int(value)

    return values["x"], values["y"]


def rotate_template(image, angle_degrees):
    height, width = image.shape

    rotation_matrix = cv2.getRotationMatrix2D(
        (width / 2.0, height / 2.0),
        angle_degrees,
        1.0
    )

    return cv2.warpAffine(
        image,
        rotation_matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )


def find_candidate_peaks(
    correlation_map,
    maximum_score,
    threshold_ratio=0.985,
    suppression_radius=18,
    maximum_candidates=12
):
    working_map = correlation_map.copy()

    threshold = maximum_score * threshold_ratio

    candidates = []

    for _ in range(maximum_candidates):
        _, peak_score, _, peak_location = cv2.minMaxLoc(
            working_map
        )

        if peak_score < threshold:
            break

        x, y = peak_location

        candidates.append({
            "top_left_x": x,
            "top_left_y": y,
            "score": float(peak_score)
        })

        x_start = max(0, x - suppression_radius)
        x_end = min(
            working_map.shape[1],
            x + suppression_radius + 1
        )

        y_start = max(0, y - suppression_radius)
        y_end = min(
            working_map.shape[0],
            y + suppression_radius + 1
        )

        working_map[y_start:y_end, x_start:x_end] = -1.0

    return candidates


def localize(reference_path, search_path):
    reference = cv2.imread(
        reference_path,
        cv2.IMREAD_GRAYSCALE
    )

    search = cv2.imread(
        search_path,
        cv2.IMREAD_GRAYSCALE
    )

    if reference is None or search is None:
        raise FileNotFoundError(
            "Could not read reference or search image."
        )

    search_normalized = normalize_image(search)

    all_candidates = []

    for scale in SCALES_TO_TRY:
        template_size = int(round(100 * scale))

        scaled_reference = cv2.resize(
            reference,
            (template_size, template_size),
            interpolation=cv2.INTER_AREA
        )

        for angle in ANGLES_TO_TRY:
            template = rotate_template(
                scaled_reference,
                angle
            )

            template_normalized = normalize_image(template)

            correlation_map = cv2.matchTemplate(
                search_normalized,
                template_normalized,
                cv2.TM_CCOEFF_NORMED
            )

            _, maximum_score, _, _ = cv2.minMaxLoc(
                correlation_map
            )

            candidates = find_candidate_peaks(
                correlation_map,
                maximum_score,
                threshold_ratio=0.985,
                suppression_radius=18,
                maximum_candidates=12
            )

            for candidate in candidates:
                candidate["scale"] = scale
                candidate["angle"] = angle
                candidate["template_size"] = template_size
                all_candidates.append(candidate)

    if not all_candidates:
        raise RuntimeError("No candidates found.")

    best_score = max(
        candidate["score"]
        for candidate in all_candidates
    )

    near_best_candidates = [
        candidate
        for candidate in all_candidates
        if candidate["score"] >= best_score * 0.985
    ]

    search_center_x = search.shape[1] / 2.0
    search_center_y = search.shape[0] / 2.0

    for candidate in near_best_candidates:
        template_size = candidate["template_size"]

        candidate["center_x"] = (
            candidate["top_left_x"] + template_size / 2.0
        )

        candidate["center_y"] = (
            candidate["top_left_y"] + template_size / 2.0
        )

        candidate["distance_to_search_center"] = np.sqrt(
            (candidate["center_x"] - search_center_x) ** 2
            + (candidate["center_y"] - search_center_y) ** 2
        )

    selected = min(
        near_best_candidates,
        key=lambda candidate: candidate["distance_to_search_center"]
    )

    predicted_x = int(round(selected["center_x"]))
    predicted_y = int(round(selected["center_y"]))

    all_scores = sorted(
        [candidate["score"] for candidate in all_candidates],
        reverse=True
    )

    if len(all_scores) >= 2:
        ambiguity_ratio = all_scores[1] / all_scores[0]
    else:
        ambiguity_ratio = 0.0

    return {
        "predicted_x": predicted_x,
        "predicted_y": predicted_y,
        "best_score": float(best_score),
        "selected_scale": float(selected["scale"]),
        "selected_angle": float(selected["angle"]),
        "near_best_candidates": len(near_best_candidates),
        "ambiguity_ratio": float(ambiguity_ratio)
    }


all_results = []

for pair_index in range(30):
    pair_id = f"batch_{pair_index:03d}"

    reference_path = os.path.join(
        REFERENCE_FOLDER,
        f"{pair_id}.png"
    )

    search_path = os.path.join(
        SEARCH_FOLDER,
        f"{pair_id}.png"
    )

    label_path = os.path.join(
        LABEL_FOLDER,
        f"{pair_id}.txt"
    )

    true_x, true_y = read_label(label_path)

    start_time = time.perf_counter()

    prediction = localize(
        reference_path,
        search_path
    )

    runtime_ms = (
        time.perf_counter() - start_time
    ) * 1000.0

    error_pixels = np.sqrt(
        (prediction["predicted_x"] - true_x) ** 2
        + (prediction["predicted_y"] - true_y) ** 2
    )

    result = {
        "pair_id": pair_id,
        "true_x": true_x,
        "true_y": true_y,
        "predicted_x": prediction["predicted_x"],
        "predicted_y": prediction["predicted_y"],
        "error_pixels": round(float(error_pixels), 3),
        "runtime_ms": round(float(runtime_ms), 3),
        "best_score": round(prediction["best_score"], 4),
        "selected_scale": prediction["selected_scale"],
        "selected_angle": prediction["selected_angle"],
        "near_best_candidates": prediction["near_best_candidates"],
        "ambiguity_ratio": round(prediction["ambiguity_ratio"], 4)
    }

    all_results.append(result)

    print(
        f"{pair_id}: "
        f"truth=({true_x}, {true_y}) "
        f"prediction=({prediction['predicted_x']}, "
        f"{prediction['predicted_y']}) "
        f"error={error_pixels:.2f}px "
        f"time={runtime_ms:.1f}ms"
    )

results_path = os.path.join(
    RESULTS_FOLDER,
    "batch_evaluation.csv"
)

with open(
    results_path,
    "w",
    newline="",
    encoding="utf-8"
) as csv_file:
    writer = csv.DictWriter(
        csv_file,
        fieldnames=all_results[0].keys()
    )

    writer.writeheader()
    writer.writerows(all_results)

errors = np.array(
    [result["error_pixels"] for result in all_results],
    dtype=np.float32
)

runtimes = np.array(
    [result["runtime_ms"] for result in all_results],
    dtype=np.float32
)

accuracy_5 = float(np.mean(errors <= 5.0) * 100.0)
accuracy_10 = float(np.mean(errors <= 10.0) * 100.0)
accuracy_25 = float(np.mean(errors <= 25.0) * 100.0)

print()
print("========== BATCH EVALUATION SUMMARY ==========")
print("Number of pairs:", len(all_results))
print("Accuracy within 5 pixels:", round(accuracy_5, 2), "%")
print("Accuracy within 10 pixels:", round(accuracy_10, 2), "%")
print("Accuracy within 25 pixels:", round(accuracy_25, 2), "%")
print("Mean localization error:", round(float(np.mean(errors)), 2), "pixels")
print("Median localization error:", round(float(np.median(errors)), 2), "pixels")
print("Maximum localization error:", round(float(np.max(errors)), 2), "pixels")
print("Mean runtime:", round(float(np.mean(runtimes)), 2), "ms per pair")
print("Maximum runtime:", round(float(np.max(runtimes)), 2), "ms")
print("Results CSV saved at:")
print(results_path)
