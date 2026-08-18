import os
import cv2
import numpy as np


PROJECT_FOLDER = r"C:\Users\Shreya\OneDrive\Desktop\drift_sense_hackathon"
OUTPUT_PATH = os.path.join(PROJECT_FOLDER, "data", "dram_layout.png")


def create_dram_layout(
    image_size=2000,
    horizontal_pitch=80,
    vertical_pitch=70,
    line_width=5,
    via_radius=5
):
    image = np.full((image_size, image_size), 35, dtype=np.uint8)

    # Horizontal word-line style structures
    for y in range(40, image_size, horizontal_pitch):
        cv2.line(
            image,
            (0, y),
            (image_size - 1, y),
            color=155,
            thickness=line_width
        )

    # Vertical bit-line style structures
    for x in range(35, image_size, vertical_pitch):
        cv2.line(
            image,
            (x, 0),
            (x, image_size - 1),
            color=120,
            thickness=line_width
        )

    # Via/contact-like features at intersections
    for y in range(40, image_size, horizontal_pitch):
        for x in range(35, image_size, vertical_pitch):
            cv2.circle(
                image,
                (x, y),
                via_radius,
                color=230,
                thickness=-1
            )

    # Add broad bands to represent larger die-level structure
    for y in range(300, image_size, 500):
        cv2.rectangle(
            image,
            (0, y),
            (image_size - 1, min(y + 18, image_size - 1)),
            color=90,
            thickness=-1
        )

    return image


layout = create_dram_layout()

cv2.imwrite(OUTPUT_PATH, layout)

print("DRAM-style layout created successfully.")
print("Saved at:")
print(OUTPUT_PATH)

cv2.imshow("DRAM-style synthetic layout", layout)
cv2.waitKey(0)
cv2.destroyAllWindows()
