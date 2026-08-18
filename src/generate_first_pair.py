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

    # Larger-scale bands: useful structure that survives downsampling
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

# Fixed target center in the large high-resolution layout.
target_center_x_hr = 6000
target_center_y_hr = 6000

# Reference is 1000x1000 at high magnification.
reference_size_hr = 1000
half_reference = reference_size_hr // 2

reference = layout[
    target_center_y_hr - half_reference: target_center_y_hr + half_reference,
    target_center_x_hr - half_reference: target_center_x_hr + half_reference
]

# Search crop is 10,000x10,000 physically, later shrunk by 10x to 1000x1000.
search_size_hr = 10000
half_search = search_size_hr // 2

search_hr = layout[
    target_center_y_hr - half_search: target_center_y_hr + half_search,
    target_center_x_hr - half_search: target_center_x_hr + half_search
]

search = cv2.resize(
    search_hr,
    (1000, 1000),
    interpolation=cv2.INTER_AREA
)

# Since the target was placed exactly at the center of the large crop,
# its correct location is the center of the search image.
true_x_search = 500
true_y_search = 500

reference_path = os.path.join(REFERENCE_FOLDER, "pair_000.png")
search_path = os.path.join(SEARCH_FOLDER, "pair_000.png")
label_path = os.path.join(LABEL_FOLDER, "pair_000.txt")

cv2.imwrite(reference_path, reference)
cv2.imwrite(search_path, search)

with open(label_path, "w", encoding="utf-8") as file:
    file.write(f"x={true_x_search}\n")
    file.write(f"y={true_y_search}\n")

print("First reference-search pair created.")
print("Reference image:", reference_path)
print("Search image:", search_path)
print("Ground truth center:", (true_x_search, true_y_search))

cv2.imshow("Reference: 1000 x 1000 high-resolution", reference)
cv2.imshow("Search: 1000 x 1000 low-resolution", search)

cv2.waitKey(0)
cv2.destroyAllWindows()
