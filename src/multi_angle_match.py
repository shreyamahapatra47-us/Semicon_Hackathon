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

# The search image was generated with a small rotation.
# Try a small practical range instead of assuming the rotation is known.
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
    center = (width / 2.0, height / 2.0)

    transformation = cv2.getRotationMatrix2D(
        center,
        angle_degrees,
        1.0
    )

    return cv2.warpAffine(
        image,
        transformation,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )


def find_candidate_peaks(
    correlation_map,
    maximum_score,
    threshold_ratio=0.98,
    suppression_radius=25,
    maximum_candidates=20
):
    working_map = correlation_map.copy()
    threshold = maximum_score * threshold_ratio
    candidates = []

    for _ in range(maximum_candidates):
        _, peak_score, _, peak_location = cv2.minMaxLoc(working_map)

        if peak_score < threshold:
            break

        x, y = peak_location

        candidates.append({
            "top_left_x": x,
            "top_left_y": y,
            "score": float(peak_score)
        })

        x_start = max(0, x - suppression_radius)
        x_end = min(working_map.shape[1], x + suppression_radius + 1)

        y_start = max(0, y - suppression_radius)
        y_end = min(working_map.shape[0], y + suppression_radius + 1)

        working_map[y_start:y_end, x_start:x_end] = -1.0

    return candidates


reference = cv2.imread(REFERENCE_PATH, cv2.IMREAD_GRAYSCALE)
search = cv2.imread(SEARCH_PATH, cv2.IMREAD_GRAYSCALE)

if reference is None:
    raise FileNotFoundError(f"Reference not found: {REFERENCE_PATH}")

if search is None:
    raise FileNotFoundError(f"Search image not found: {SEARCH_PATH}")

true_x, true_y = read_label(LABEL_PATH)

# Convert high-magnification reference to the search-image scale.
small_reference = cv2.resize(
    reference,
    (100, 100),
    interpolation=cv2.INTER_AREA
)

search_normalized = normalize_image(search)

all_candidates = []

for angle in ANGLES_TO_TRY:
    rotated_reference = rotate_template(
        small_reference,
        angle
    )

    rotated_reference_normalized = normalize_image(
        rotated_reference
    )

    correlation_map = cv2.matchTemplate(
        search_normalized,
        rotated_reference_normalized,
        cv2.TM_CCOEFF_NORMED
    )

    _, maximum_score, _, _ = cv2.minMaxLoc(correlation_map)

    angle_candidates = find_candidate_peaks(
        correlation_map,
        maximum_score,
        threshold_ratio=0.98,
        suppression_radius=25,
        maximum_candidates=20
    )

    for candidate in angle_candidates:
        candidate["angle"] = angle
        all_candidates.append(candidate)

if not all_candidates:
    raise RuntimeError("No candidates were found.")

# Use only candidates close to the overall strongest score.
best_score = max(
    candidate["score"]
    for candidate in all_candidates
)

near_best_candidates = [
    candidate
    for candidate in all_candidates
    if candidate["score"] >= best_score * 0.98
]

template_height, template_width = small_reference.shape

search_center_x = search.shape[1] / 2.0
search_center_y = search.shape[0] / 2.0

for candidate in near_best_candidates:
    candidate["center_x"] = (
        candidate["top_left_x"] + template_width / 2.0
    )

    candidate["center_y"] = (
        candidate["top_left_y"] + template_height / 2.0
    )

    candidate["distance_to_search_center"] = np.sqrt(
        (candidate["center_x"] - search_center_x) ** 2
        + (candidate["center_y"] - search_center_y) ** 2
    )

# Problem-statement rule:
# for approximately equivalent matches, choose the closest candidate to image center.
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

sorted_scores = sorted(
    [candidate["score"] for candidate in all_candidates],
    reverse=True
)

if len(sorted_scores) >= 2:
    ambiguity_ratio = sorted_scores[1] / sorted_scores[0]
else:
    ambiguity_ratio = 0.0

print("Testing pair:", PAIR_ID)
print("Angles tested:", ANGLES_TO_TRY)
print("Best score:", round(float(best_score), 4))
print("Near-best candidates:", len(near_best_candidates))
print("Selected template angle:", selected["angle"], "degrees")
print("Predicted center (x, y):", (predicted_x, predicted_y))
print("Ground-truth center (x, y):", (true_x, true_y))
print("Localization error in pixels:", round(float(error_pixels), 2))
print("Ambiguity ratio:", round(float(ambiguity_ratio), 4))

visualization = cv2.cvtColor(search, cv2.COLOR_GRAY2BGR)

# Yellow marks all approximately equal candidate locations.
for candidate in near_best_candidates:
    cv2.circle(
        visualization,
        (
            int(round(candidate["center_x"])),
            int(round(candidate["center_y"]))
        ),
        5,
        (0, 255, 255),
        1
    )

# Green rectangle: selected prediction.
top_left = (
    selected["top_left_x"],
    selected["top_left_y"]
)

bottom_right = (
    selected["top_left_x"] + template_width,
    selected["top_left_y"] + template_height
)

cv2.rectangle(
    visualization,
    top_left,
    bottom_right,
    (0, 255, 0),
    2
)

# Red dot: exact ground truth.
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
    f"multi_angle_{PAIR_ID}.png"
)

cv2.imwrite(output_path, visualization)

print("Visualization saved at:")
print(output_path)

cv2.imshow("Multi-angle localization", visualization)
cv2.waitKey(0)
cv2.destroyAllWindows()
