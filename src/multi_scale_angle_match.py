import os
import cv2
import numpy as np


PROJECT_FOLDER = r"C:\Users\Shreya\OneDrive\Desktop\drift_sense_hackathon"

PAIR_ID = "pair_002"

REFERENCE_PATH = os.path.join(
    PROJECT_FOLDER,
    "data",
    "reference",
    f"{PAIR_ID}.png"
)

SEARCH_PATH = os.path.join(
    PROJECT_FOLDER,
    "data",
    "search",
    f"{PAIR_ID}.png"
)

LABEL_PATH = os.path.join(
    PROJECT_FOLDER,
    "data",
    "labels",
    f"{PAIR_ID}.txt"
)

# Expected reference size inside search is 100x100.
# Try small scale variation around this expected size.
SCALES_TO_TRY = [0.96, 0.98, 1.00, 1.02, 1.04]

# Try small angular mismatch around zero.
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

    rotation_center = (
        width / 2.0,
        height / 2.0
    )

    rotation_matrix = cv2.getRotationMatrix2D(
        rotation_center,
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


reference = cv2.imread(
    REFERENCE_PATH,
    cv2.IMREAD_GRAYSCALE
)

search = cv2.imread(
    SEARCH_PATH,
    cv2.IMREAD_GRAYSCALE
)

if reference is None:
    raise FileNotFoundError(
        f"Reference not found: {REFERENCE_PATH}"
    )

if search is None:
    raise FileNotFoundError(
        f"Search image not found: {SEARCH_PATH}"
    )

true_x, true_y = read_label(LABEL_PATH)

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
        transformed_template = rotate_template(
            scaled_reference,
            angle
        )

        transformed_template = normalize_image(
            transformed_template
        )

        correlation_map = cv2.matchTemplate(
            search_normalized,
            transformed_template,
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
    raise RuntimeError("No localization candidates found.")

best_score = max(
    candidate["score"]
    for candidate in all_candidates
)

# Keep all candidates whose scores are practically equivalent.
near_best_candidates = [
    candidate
    for candidate in all_candidates
    if candidate["score"] >= best_score * 0.985
]

search_center_x = search.shape[1] / 2.0
search_center_y = search.shape[0] / 2.0

for candidate in near_best_candidates:
    size = candidate["template_size"]

    candidate["center_x"] = candidate["top_left_x"] + size / 2.0
    candidate["center_y"] = candidate["top_left_y"] + size / 2.0

    candidate["distance_to_search_center"] = np.sqrt(
        (candidate["center_x"] - search_center_x) ** 2
        + (candidate["center_y"] - search_center_y) ** 2
    )

# Required periodic-region tie-break:
# select the most central candidate among similar-scoring matches.
selected = min(
    near_best_candidates,
    key=lambda candidate: candidate["distance_to_search_center"]
)

predicted_x = int(round(selected["center_x"]))
predicted_y = int(round(selected["center_y"]))

error_pixels = np.sqrt(
    (predicted_x - true_x) ** 2
    + (predicted_y - true_y) ** 2
)

all_scores = sorted(
    [
        candidate["score"]
        for candidate in all_candidates
    ],
    reverse=True
)

if len(all_scores) >= 2:
    ambiguity_ratio = all_scores[1] / all_scores[0]
else:
    ambiguity_ratio = 0.0

print("Testing pair:", PAIR_ID)
print("Scales tested:", SCALES_TO_TRY)
print("Angles tested:", ANGLES_TO_TRY)
print("Best score:", round(float(best_score), 4))
print("Near-best candidates:", len(near_best_candidates))
print("Selected scale:", selected["scale"])
print("Selected angle:", selected["angle"], "degrees")
print("Predicted center (x, y):", (predicted_x, predicted_y))
print("Ground-truth center (x, y):", (true_x, true_y))
print("Localization error in pixels:", round(float(error_pixels), 2))
print("Ambiguity ratio:", round(float(ambiguity_ratio), 4))

visualization = cv2.cvtColor(
    search,
    cv2.COLOR_GRAY2BGR
)

# Yellow circles: all approximately equivalent candidates.
for candidate in near_best_candidates:
    cv2.circle(
        visualization,
        (
            int(round(candidate["center_x"])),
            int(round(candidate["center_y"]))
        ),
        4,
        (0, 255, 255),
        1
    )

selected_size = selected["template_size"]

cv2.rectangle(
    visualization,
    (
        selected["top_left_x"],
        selected["top_left_y"]
    ),
    (
        selected["top_left_x"] + selected_size,
        selected["top_left_y"] + selected_size
    ),
    (0, 255, 0),
    2
)

# Red dot: ground truth.
cv2.circle(
    visualization,
    (true_x, true_y),
    7,
    (0, 0, 255),
    -1
)

cv2.putText(
    visualization,
    "Yellow: candidates | Green: prediction | Red: truth",
    (15, 35),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.55,
    (255, 255, 255),
    2
)

output_path = os.path.join(
    PROJECT_FOLDER,
    "results",
    f"multi_scale_angle_{PAIR_ID}.png"
)

cv2.imwrite(output_path, visualization)

print("Visualization saved at:")
print(output_path)

cv2.imshow(
    "Multi-scale and multi-angle localization",
    visualization
)

cv2.waitKey(0)
cv2.destroyAllWindows()
