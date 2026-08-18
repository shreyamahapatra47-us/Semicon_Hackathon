import os
import cv2
import numpy as np


PROJECT_FOLDER = r"C:\Users\Shreya\OneDrive\Desktop\drift_sense_hackathon"

# Change only this line when you want to test another pair.
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


def find_candidate_peaks(
    correlation_map,
    maximum_score,
    threshold_ratio=0.995,
    suppression_radius=25,
    maximum_candidates=50
):
    working_map = correlation_map.copy()
    threshold = maximum_score * threshold_ratio
    candidates = []

    for _ in range(maximum_candidates):
        _, peak_score, _, peak_location = cv2.minMaxLoc(working_map)

        if peak_score < threshold:
            break

        peak_x, peak_y = peak_location

        candidates.append({
            "top_left_x": peak_x,
            "top_left_y": peak_y,
            "score": float(peak_score)
        })

        x_start = max(0, peak_x - suppression_radius)
        x_end = min(working_map.shape[1], peak_x + suppression_radius + 1)

        y_start = max(0, peak_y - suppression_radius)
        y_end = min(working_map.shape[0], peak_y + suppression_radius + 1)

        working_map[y_start:y_end, x_start:x_end] = -1.0

    return candidates


reference = cv2.imread(REFERENCE_PATH, cv2.IMREAD_GRAYSCALE)
search = cv2.imread(SEARCH_PATH, cv2.IMREAD_GRAYSCALE)

if reference is None:
    raise FileNotFoundError(f"Reference not found: {REFERENCE_PATH}")

if search is None:
    raise FileNotFoundError(f"Search image not found: {SEARCH_PATH}")

if not os.path.exists(LABEL_PATH):
    raise FileNotFoundError(f"Label not found: {LABEL_PATH}")

true_x, true_y = read_label(LABEL_PATH)

small_reference = cv2.resize(
    reference,
    (100, 100),
    interpolation=cv2.INTER_AREA
)

search_normalized = normalize_image(search)
reference_normalized = normalize_image(small_reference)

correlation_map = cv2.matchTemplate(
    search_normalized,
    reference_normalized,
    cv2.TM_CCOEFF_NORMED
)

_, maximum_value, _, _ = cv2.minMaxLoc(correlation_map)

candidates = find_candidate_peaks(
    correlation_map,
    maximum_score=maximum_value,
    threshold_ratio=0.995,
    suppression_radius=25,
    maximum_candidates=50
)

template_height, template_width = small_reference.shape

search_center_x = search.shape[1] / 2
search_center_y = search.shape[0] / 2

for candidate in candidates:
    center_x = candidate["top_left_x"] + template_width / 2
    center_y = candidate["top_left_y"] + template_height / 2

    candidate["center_x"] = center_x
    candidate["center_y"] = center_y

    candidate["distance_to_search_center"] = np.sqrt(
        (center_x - search_center_x) ** 2
        + (center_y - search_center_y) ** 2
    )

selected = min(
    candidates,
    key=lambda candidate: candidate["distance_to_search_center"]
)

predicted_x = int(round(selected["center_x"]))
predicted_y = int(round(selected["center_y"]))

error_pixels = np.sqrt(
    (predicted_x - true_x) ** 2
    + (predicted_y - true_y) ** 2
)

print("Testing pair:", PAIR_ID)
print("Maximum correlation score:", round(float(maximum_value), 4))
print("Number of near-equal candidate peaks:", len(candidates))
print("Predicted center (x, y):", (predicted_x, predicted_y))
print("Ground-truth center (x, y):", (true_x, true_y))
print("Localization error in pixels:", round(float(error_pixels), 2))

if len(candidates) >= 2:
    sorted_scores = sorted(
        [candidate["score"] for candidate in candidates],
        reverse=True
    )

    ambiguity_ratio = sorted_scores[1] / sorted_scores[0]

    print("Ambiguity ratio (second-best / best):", round(ambiguity_ratio, 4))

visualization = cv2.cvtColor(search, cv2.COLOR_GRAY2BGR)

for candidate in candidates:
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

cv2.circle(
    visualization,
    (true_x, true_y),
    7,
    (0, 0, 255),
    -1
)

cv2.putText(
    visualization,
    "Yellow: candidates | Green: predicted | Red: true",
    (15, 35),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.55,
    (255, 255, 255),
    2
)

output_path = os.path.join(
    PROJECT_FOLDER,
    "results",
    f"result_{PAIR_ID}.png"
)

cv2.imwrite(output_path, visualization)

print("Visualization saved at:")
print(output_path)

cv2.imshow(f"Localization result: {PAIR_ID}", visualization)
cv2.waitKey(0)
cv2.destroyAllWindows()
