import os
import cv2
import numpy as np


PROJECT_FOLDER = r"C:\Users\Shreya\OneDrive\Desktop\drift_sense_hackathon"

PAIR_ID = "large_024"

REFERENCE_PATH = os.path.join(
    PROJECT_FOLDER,
    "data",
    "reference_context_large",
    f"{PAIR_ID}.png"
)

SEARCH_PATH = os.path.join(
    PROJECT_FOLDER,
    "data",
    "search_context_large",
    f"{PAIR_ID}.png"
)

LABEL_PATH = os.path.join(
    PROJECT_FOLDER,
    "data",
    "labels_context_large",
    f"{PAIR_ID}.txt"
)

OUTPUT_PATH = os.path.join(
    PROJECT_FOLDER,
    "results",
    "large_context_failure_case.png"
)

BASE_TEMPLATE_SIZE = 300
SCALES_TO_TRY = [0.98, 1.00, 1.02]
ANGLES_TO_TRY = [-1.0, -0.5, 0.0, 0.5, 1.0]
TIE_SCORE_RATIO = 0.998


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

    matrix = cv2.getRotationMatrix2D(
        (width / 2.0, height / 2.0),
        angle_degrees,
        1.0
    )

    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )


def find_peaks(correlation_map, maximum_score):
    working_map = correlation_map.copy()
    threshold = maximum_score * TIE_SCORE_RATIO
    candidates = []

    for _ in range(10):
        _, peak_score, _, peak_location = cv2.minMaxLoc(working_map)

        if peak_score < threshold:
            break

        x, y = peak_location

        candidates.append({
            "top_left_x": x,
            "top_left_y": y,
            "score": float(peak_score)
        })

        radius = 60

        x1 = max(0, x - radius)
        x2 = min(working_map.shape[1], x + radius + 1)

        y1 = max(0, y - radius)
        y2 = min(working_map.shape[0], y + radius + 1)

        working_map[y1:y2, x1:x2] = -1.0

    return candidates


reference = cv2.imread(
    REFERENCE_PATH,
    cv2.IMREAD_GRAYSCALE
)

search = cv2.imread(
    SEARCH_PATH,
    cv2.IMREAD_GRAYSCALE
)

if reference is None or search is None:
    raise FileNotFoundError("Could not read reference or search image.")

true_x, true_y = read_label(LABEL_PATH)

search_normalized = normalize_image(search)
all_candidates = []

for scale in SCALES_TO_TRY:
    template_size = int(round(BASE_TEMPLATE_SIZE * scale))

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

        response = cv2.matchTemplate(
            search_normalized,
            template_normalized,
            cv2.TM_CCOEFF_NORMED
        )

        _, maximum_score, _, _ = cv2.minMaxLoc(response)

        candidates = find_peaks(
            response,
            maximum_score
        )

        for candidate in candidates:
            candidate["scale"] = scale
            candidate["angle"] = angle
            candidate["template_size"] = template_size
            all_candidates.append(candidate)

best_score = max(candidate["score"] for candidate in all_candidates)

near_best = [
    candidate
    for candidate in all_candidates
    if candidate["score"] >= best_score * TIE_SCORE_RATIO
]

for candidate in near_best:
    size = candidate["template_size"]

    candidate["center_x"] = candidate["top_left_x"] + size / 2.0
    candidate["center_y"] = candidate["top_left_y"] + size / 2.0

    candidate["distance_to_center"] = np.sqrt(
        (candidate["center_x"] - 500.0) ** 2
        + (candidate["center_y"] - 500.0) ** 2
    )

selected = min(
    near_best,
    key=lambda candidate: candidate["distance_to_center"]
)

predicted_x = int(round(selected["center_x"]))
predicted_y = int(round(selected["center_y"]))

error_pixels = np.sqrt(
    (predicted_x - true_x) ** 2
    + (predicted_y - true_y) ** 2
)

# Search view
search_view = cv2.cvtColor(search, cv2.COLOR_GRAY2BGR)

# Yellow rectangles: near-equal periodic alternatives.
for candidate in near_best:
    size = candidate["template_size"]

    cv2.rectangle(
        search_view,
        (
            candidate["top_left_x"],
            candidate["top_left_y"]
        ),
        (
            candidate["top_left_x"] + size,
            candidate["top_left_y"] + size
        ),
        (0, 255, 255),
        2
    )

# Green: selected prediction.
selected_size = selected["template_size"]

cv2.rectangle(
    search_view,
    (
        selected["top_left_x"],
        selected["top_left_y"]
    ),
    (
        selected["top_left_x"] + selected_size,
        selected["top_left_y"] + selected_size
    ),
    (0, 255, 0),
    4
)

# Red: known ground truth.
cv2.circle(
    search_view,
    (true_x, true_y),
    10,
    (0, 0, 255),
    -1
)

cv2.putText(
    search_view,
    "Yellow: tied candidates | Green: selected | Red: ground truth",
    (20, 35),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.60,
    (255, 255, 255),
    2
)

cv2.putText(
    search_view,
    f"Error = {error_pixels:.2f} px | Candidates = {len(near_best)}",
    (20, 70),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.60,
    (255, 255, 255),
    2
)

# Resize the reference so it can be placed next to the search visualization.
reference_view = cv2.resize(
    reference,
    (400, 400),
    interpolation=cv2.INTER_AREA
)

reference_view = cv2.cvtColor(
    reference_view,
    cv2.COLOR_GRAY2BGR
)

cv2.putText(
    reference_view,
    "Reference image",
    (20, 35),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.80,
    (0, 255, 0),
    2
)

combined_height = max(
    search_view.shape[0],
    reference_view.shape[0]
)

combined_width = (
    search_view.shape[1]
    + reference_view.shape[1]
)

combined = np.zeros(
    (combined_height, combined_width, 3),
    dtype=np.uint8
)

combined[
    0:search_view.shape[0],
    0:search_view.shape[1]
] = search_view

combined[
    0:reference_view.shape[0],
    search_view.shape[1]:search_view.shape[1] + reference_view.shape[1]
] = reference_view

cv2.imwrite(OUTPUT_PATH, combined)

print("Failure-case visualization saved:")
print(OUTPUT_PATH)
print("Ground truth:", (true_x, true_y))
print("Prediction:", (predicted_x, predicted_y))
print("Error:", round(float(error_pixels), 2), "pixels")
print("Near-equal candidates:", len(near_best))

cv2.imshow("Large-context failure analysis", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()
