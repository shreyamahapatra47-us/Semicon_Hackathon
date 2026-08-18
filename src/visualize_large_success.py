import os
import cv2
import numpy as np


PROJECT_FOLDER = r"C:\Users\Shreya\OneDrive\Desktop\drift_sense_hackathon"

PAIR_ID = "large_000"

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
    "large_context_success_case.png"
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

best_result = None

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

        _, score, _, top_left = cv2.minMaxLoc(response)

        if best_result is None or score > best_result["score"]:
            best_result = {
                "score": float(score),
                "top_left_x": top_left[0],
                "top_left_y": top_left[1],
                "template_size": template_size,
                "scale": scale,
                "angle": angle
            }

predicted_x = int(round(
    best_result["top_left_x"]
    + best_result["template_size"] / 2.0
))

predicted_y = int(round(
    best_result["top_left_y"]
    + best_result["template_size"] / 2.0
))

error_pixels = np.sqrt(
    (predicted_x - true_x) ** 2
    + (predicted_y - true_y) ** 2
)

search_view = cv2.cvtColor(search, cv2.COLOR_GRAY2BGR)

cv2.rectangle(
    search_view,
    (
        best_result["top_left_x"],
        best_result["top_left_y"]
    ),
    (
        best_result["top_left_x"] + best_result["template_size"],
        best_result["top_left_y"] + best_result["template_size"]
    ),
    (0, 255, 0),
    4
)

# Red dot = ground truth. It should lie at the center of the green prediction.
cv2.circle(
    search_view,
    (true_x, true_y),
    10,
    (0, 0, 255),
    -1
)

cv2.putText(
    search_view,
    "Green: predicted region | Red: ground-truth center",
    (20, 35),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.60,
    (255, 255, 255),
    2
)

cv2.putText(
    search_view,
    f"Error = {error_pixels:.2f} px | "
    f"Scale = {best_result['scale']} | "
    f"Angle = {best_result['angle']} deg",
    (20, 70),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.55,
    (255, 255, 255),
    2
)

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

print("Success-case visualization saved:")
print(OUTPUT_PATH)
print("Ground truth:", (true_x, true_y))
print("Prediction:", (predicted_x, predicted_y))
print("Error:", round(float(error_pixels), 2), "pixels")
print("Best score:", round(best_result["score"], 4))
print("Selected scale:", best_result["scale"])
print("Selected angle:", best_result["angle"])

cv2.imshow("Large-context success case", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()
