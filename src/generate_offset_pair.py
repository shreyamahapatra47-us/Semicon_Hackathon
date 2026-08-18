import os
import cv2
import numpy as np


PROJECT_FOLDER = r"C:\Users\Shreya\OneDrive\Desktop\drift_sense_hackathon"

REFERENCE_FOLDER = os.path.join(PROJECT_FOLDER, "data", "reference")
SEARCH_FOLDER = os.path.join(PROJECT_FOLDER, "data", "search")
LABEL_FOLDER = os.path.join(PROJECT_FOLDER, "data", "labels")

os.makedirs(REFERENCE_FOLDER, exist_ok=True)
os.makedirs(SEARCH_FOLDER, exist_ok=True)
os.makedirs(LABEL_FOLDER, exist_ok=True)


def create_large_dram_layout(
    image_size=12000,
    horizontal_pitch=80,
    vertical_pitch=70,
    line_width=5,
    via_radius=5
):
    image = np.full((image_size, image_size), 35, dtype=np.uint8)

    for y in range(40, image_size, horizontal_pitch):
        cv2.line(
            image,
            (0, y),
            (image_size - 1, y),
            color=155,
            thickness=line_width
        )

    for x in range(35, image_size, vertical_pitch):
        cv2.line(
            image,
            (x, 0),
            (x, image_size - 1),
            color=120,
            thickness=line_width
        )

    for y in range(40, image_size, horizontal_pitch):
        for x in range(35, image_size, vertical_pitch):
            cv2.circle(
                image,
                (x, y),
                via_radius,
                color=230,
                thickness=-1
            )

    # Coarse bands survive 10x downsampling.
    for y in range(300, image_size, 500):
        cv2.rectangle(
            image,
            (0, y),
            (image_size - 1, min(y + 20, image_size - 1)),
            color=90,
            thickness=-1
        )

    for x in range(450, image_size, 900):
        cv2.rectangle(
            image,
            (x, 0),
            (min(x + 14, image_size - 1), image_size - 1),
            color=75,
            thickness=-1
        )

    return image


layout = create_large_dram_layout()

# Target is the physical center of the high-resolution reference crop.
target_center_x_hr = 6200
target_center_y_hr = 5750

# Search crop is centered at a different physical position.
# Thus, the reference target appears at an offset location in search.
search_center_x_hr = 6000
search_center_y_hr = 6000

reference_size_hr = 1000
search_size_hr = 10000

half_reference = reference_size_hr // 2
half_search = search_size_hr // 2

reference = layout[
    target_center_y_hr - half_reference: target_center_y_hr + half_reference,
    target_center_x_hr - half_reference: target_center_x_hr + half_reference
]

search_hr = layout[
    search_center_y_hr - half_search: search_center_y_hr + half_search,
    search_center_x_hr - half_search: search_center_x_hr + half_search
]

search = cv2.resize(
    search_hr,
    (1000, 1000),
    interpolation=cv2.INTER_AREA
)

# Convert physical high-resolution displacement to coordinates in 10x-downsampled search.
true_x_search = int(round(
    500 + (target_center_x_hr - search_center_x_hr) / 10
))

true_y_search = int(round(
    500 + (target_center_y_hr - search_center_y_hr) / 10
))

reference_path = os.path.join(REFERENCE_FOLDER, "pair_001.png")
search_path = os.path.join(SEARCH_FOLDER, "pair_001.png")
label_path = os.path.join(LABEL_FOLDER, "pair_001.txt")

cv2.imwrite(reference_path, reference)
cv2.imwrite(search_path, search)

with open(label_path, "w", encoding="utf-8") as file:
    file.write(f"x={true_x_search}\n")
    file.write(f"y={true_y_search}\n")

print("Offset reference-search pair created.")
print("Reference image:", reference_path)
print("Search image:", search_path)
print("True target center in search:", (true_x_search, true_y_search))

cv2.imshow("Reference: pair_001", reference)
cv2.imshow("Search: pair_001", search)

cv2.waitKey(0)
cv2.destroyAllWindows()
