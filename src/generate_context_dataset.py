import os
import csv
import cv2
import numpy as np


PROJECT_FOLDER = r"C:\Users\Shreya\OneDrive\Desktop\drift_sense_hackathon"

REFERENCE_FOLDER = os.path.join(
    PROJECT_FOLDER, "data", "reference_context"
)

SEARCH_FOLDER = os.path.join(
    PROJECT_FOLDER, "data", "search_context"
)

LABEL_FOLDER = os.path.join(
    PROJECT_FOLDER, "data", "labels_context"
)

os.makedirs(REFERENCE_FOLDER, exist_ok=True)
os.makedirs(SEARCH_FOLDER, exist_ok=True)
os.makedirs(LABEL_FOLDER, exist_ok=True)


def create_contextual_dram_layout(
    image_size=11000,
    horizontal_pitch=80,
    vertical_pitch=70,
    seed=4040
):
    rng = np.random.default_rng(seed)

    image = np.full(
        (image_size, image_size),
        35,
        dtype=np.uint8
    )

    # Fine periodic DRAM-like structure.
    for y in range(40, image_size, horizontal_pitch):
        cv2.line(
            image,
            (0, y),
            (image_size - 1, y),
            155,
            5
        )

    for x in range(35, image_size, vertical_pitch):
        cv2.line(
            image,
            (x, 0),
            (x, image_size - 1),
            120,
            5
        )

    for y in range(40, image_size, horizontal_pitch):
        for x in range(35, image_size, vertical_pitch):
            brightness = int(rng.integers(185, 236))
            radius = int(rng.integers(4, 7))

            cv2.circle(
                image,
                (x, y),
                radius,
                brightness,
                -1
            )

    # Coarse sub-array separators.
    for y in range(300, image_size, 1500):
        cv2.rectangle(
            image,
            (0, y),
            (image_size - 1, min(y + 25, image_size - 1)),
            82,
            -1
        )

    for x in range(500, image_size, 1700):
        cv2.rectangle(
            image,
            (x, 0),
            (min(x + 20, image_size - 1), image_size - 1),
            76,
            -1
        )

    # Local low-frequency contextual patches.
    # These are created locally, so memory use stays low.
    for _ in range(60):
        center_x = int(rng.integers(300, image_size - 300))
        center_y = int(rng.integers(300, image_size - 300))

        patch_size = int(rng.integers(180, 420))
        half_patch = patch_size // 2

        x1 = max(0, center_x - half_patch)
        x2 = min(image_size, center_x + half_patch)
        y1 = max(0, center_y - half_patch)
        y2 = min(image_size, center_y + half_patch)

        patch_height = y2 - y1
        patch_width = x2 - x1

        local_mask = np.zeros(
            (patch_height, patch_width),
            dtype=np.uint8
        )

        cv2.ellipse(
            local_mask,
            (patch_width // 2, patch_height // 2),
            (
                int(patch_width * rng.uniform(0.25, 0.45)),
                int(patch_height * rng.uniform(0.25, 0.45))
            ),
            float(rng.uniform(0, 180)),
            0,
            360,
            255,
            -1
        )

        local_mask = cv2.GaussianBlur(
            local_mask,
            (0, 0),
            sigmaX=float(rng.uniform(15, 35))
        )

        intensity_shift = float(rng.uniform(-40, 40))

        patch = image[y1:y2, x1:x2].astype(np.float32)
        patch = patch + intensity_shift * (
            local_mask.astype(np.float32) / 255.0
        )

        image[y1:y2, x1:x2] = np.clip(
            patch,
            0,
            255
        ).astype(np.uint8)

    # Sparse irregular routing / density context.
    for _ in range(30):
        x1 = int(rng.integers(100, image_size - 600))
        y1 = int(rng.integers(100, image_size - 600))

        length = int(rng.integers(150, 600))
        width = int(rng.integers(10, 30))
        brightness = int(rng.integers(55, 115))

        if rng.integers(0, 2) == 0:
            x2 = min(image_size - 1, x1 + length)
            y2 = y1
        else:
            x2 = x1
            y2 = min(image_size - 1, y1 + length)

        cv2.line(
            image,
            (x1, y1),
            (x2, y2),
            brightness,
            width
        )

    return image


def add_edge_brightening(image, strength):
    image_float = image.astype(np.float32)

    gx = cv2.Sobel(image_float, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image_float, cv2.CV_32F, 0, 1, ksize=3)

    magnitude = cv2.magnitude(gx, gy)

    magnitude = cv2.normalize(
        magnitude,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return np.clip(
        image_float + strength * magnitude,
        0,
        255
    ).astype(np.uint8)


def add_noise(image, gaussian_sigma, poisson_strength, rng):
    image_float = image.astype(np.float32)

    poisson_input = np.clip(
        image_float * poisson_strength,
        0,
        None
    )

    poisson_image = rng.poisson(poisson_input).astype(np.float32)
    poisson_image = poisson_image / poisson_strength

    gaussian_noise = rng.normal(
        0,
        gaussian_sigma,
        image.shape
    ).astype(np.float32)

    return np.clip(
        poisson_image + gaussian_noise,
        0,
        255
    ).astype(np.uint8)


def rotate_image(image, angle_degrees):
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


def rotate_point(x, y, angle_degrees):
    matrix = cv2.getRotationMatrix2D(
        (500.0, 500.0),
        angle_degrees,
        1.0
    )

    point = np.array([x, y, 1.0], dtype=np.float32)
    transformed = matrix @ point

    return float(transformed[0]), float(transformed[1])


def create_pair(pair_index, layout, rng):
    offset_x = int(rng.integers(-120, 121))
    offset_y = int(rng.integers(-120, 121))

    # Search crop covers physical coordinates from 500 to 10500.
    search_center_hr = 5500

    target_x_hr = search_center_hr + offset_x * 10
    target_y_hr = search_center_hr + offset_y * 10

    reference_half = 500
    search_half = 5000

    reference_clean = layout[
        target_y_hr - reference_half:target_y_hr + reference_half,
        target_x_hr - reference_half:target_x_hr + reference_half
    ]

    search_high_resolution = layout[
        search_center_hr - search_half:search_center_hr + search_half,
        search_center_hr - search_half:search_center_hr + search_half
    ]

    search_clean = cv2.resize(
        search_high_resolution,
        (1000, 1000),
        interpolation=cv2.INTER_AREA
    )

    rotation_angle = float(rng.uniform(-1.0, 1.0))

    reference_rng = np.random.default_rng(
        int(rng.integers(0, 1_000_000))
    )

    search_rng = np.random.default_rng(
        int(rng.integers(0, 1_000_000))
    )

    reference = add_edge_brightening(
        reference_clean,
        float(rng.uniform(0.12, 0.25))
    )

    reference = cv2.GaussianBlur(
        reference,
        (3, 3),
        float(rng.uniform(0.3, 0.8))
    )

    reference = add_noise(
        reference,
        float(rng.uniform(2.0, 4.0)),
        float(rng.uniform(1.5, 2.5)),
        reference_rng
    )

    search = add_edge_brightening(
        search_clean,
        float(rng.uniform(0.25, 0.45))
    )

    search = cv2.GaussianBlur(
        search,
        (5, 5),
        float(rng.uniform(0.7, 1.3))
    )

    search = rotate_image(
        search,
        rotation_angle
    )

    search = add_noise(
        search,
        float(rng.uniform(5.0, 9.0)),
        float(rng.uniform(0.9, 1.5)),
        search_rng
    )

    true_x_before_rotation = 500.0 + offset_x
    true_y_before_rotation = 500.0 + offset_y

    true_x, true_y = rotate_point(
        true_x_before_rotation,
        true_y_before_rotation,
        rotation_angle
    )

    pair_id = f"context_{pair_index:03d}"

    cv2.imwrite(
        os.path.join(REFERENCE_FOLDER, f"{pair_id}.png"),
        reference
    )

    cv2.imwrite(
        os.path.join(SEARCH_FOLDER, f"{pair_id}.png"),
        search
    )

    with open(
        os.path.join(LABEL_FOLDER, f"{pair_id}.txt"),
        "w",
        encoding="utf-8"
    ) as file:
        file.write(f"x={int(round(true_x))}\n")
        file.write(f"y={int(round(true_y))}\n")

    return {
        "pair_id": pair_id,
        "true_x": int(round(true_x)),
        "true_y": int(round(true_y)),
        "rotation_degrees": round(rotation_angle, 3)
    }


print("Building contextual layout. Please wait...")

layout = create_contextual_dram_layout()

print("Layout built. Creating 30 image pairs...")

dataset_rng = np.random.default_rng(3030)
metadata_rows = []

for index in range(30):
    item = create_pair(index, layout, dataset_rng)
    metadata_rows.append(item)

    print(
        f"Created {item['pair_id']} "
        f"at ({item['true_x']}, {item['true_y']})"
    )

metadata_path = os.path.join(
    PROJECT_FOLDER,
    "data",
    "context_dataset_metadata.csv"
)

with open(
    metadata_path,
    "w",
    newline="",
    encoding="utf-8"
) as csv_file:
    writer = csv.DictWriter(
        csv_file,
        fieldnames=metadata_rows[0].keys()
    )

    writer.writeheader()
    writer.writerows(metadata_rows)

print()
print("Context dataset complete.")
print("Pairs created:", len(metadata_rows))
print("Metadata:", metadata_path)
