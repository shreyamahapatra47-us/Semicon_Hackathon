import os
import csv
import cv2
import numpy as np


PROJECT_FOLDER = r"C:\Users\Shreya\OneDrive\Desktop\drift_sense_hackathon"

REFERENCE_FOLDER = os.path.join(PROJECT_FOLDER, "data", "reference_variation")
SEARCH_FOLDER = os.path.join(PROJECT_FOLDER, "data", "search_variation")
LABEL_FOLDER = os.path.join(PROJECT_FOLDER, "data", "labels_variation")

os.makedirs(REFERENCE_FOLDER, exist_ok=True)
os.makedirs(SEARCH_FOLDER, exist_ok=True)
os.makedirs(LABEL_FOLDER, exist_ok=True)


def create_dram_layout_with_variation(
    image_size=12000,
    horizontal_pitch=80,
    vertical_pitch=70,
    line_width=5,
    via_radius=5,
    seed=9001
):
    rng = np.random.default_rng(seed)

    image = np.full((image_size, image_size), 35, dtype=np.uint8)

    # Base word-line and bit-line layout.
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

    # Coarse structures preserve multi-scale context after downsampling.
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

    # Every cell has small fixed variations.
    # This represents simplified local process variation.
    for y in range(40, image_size, horizontal_pitch):
        for x in range(35, image_size, vertical_pitch):
            random_value = rng.random()

            # Rare weak/missing contact.
            if random_value < 0.025:
                brightness = int(rng.integers(70, 115))
                radius = int(rng.integers(2, 4))

            # Some larger or brighter contacts.
            elif random_value < 0.14:
                brightness = int(rng.integers(205, 256))
                radius = int(rng.integers(5, 8))

            # Typical contacts with small variation.
            else:
                brightness = int(rng.integers(165, 231))
                radius = int(rng.integers(4, 7))

            cv2.circle(
                image,
                (x, y),
                radius,
                brightness,
                thickness=-1
            )

    # Sparse local patches imitate local fabrication/process variation.
    for _ in range(180):
        patch_x = int(rng.integers(100, image_size - 200))
        patch_y = int(rng.integers(100, image_size - 200))

        patch_width = int(rng.integers(12, 35))
        patch_height = int(rng.integers(12, 35))

        brightness_shift = int(rng.integers(-25, 26))

        patch = image[
            patch_y:patch_y + patch_height,
            patch_x:patch_x + patch_width
        ].astype(np.int16)

        patch = np.clip(
            patch + brightness_shift,
            0,
            255
        ).astype(np.uint8)

        image[
            patch_y:patch_y + patch_height,
            patch_x:patch_x + patch_width
        ] = patch

    return image


def add_edge_brightening(image, strength):
    image_float = image.astype(np.float32)

    gradient_x = cv2.Sobel(
        image_float,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    gradient_y = cv2.Sobel(
        image_float,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    magnitude = cv2.magnitude(gradient_x, gradient_y)

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


def add_independent_noise(image, gaussian_sigma, poisson_strength, rng):
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

    target_center_x_hr = 6000 + offset_x * 10
    target_center_y_hr = 6000 + offset_y * 10

    reference_size_hr = 1000
    search_size_hr = 10000

    half_reference = reference_size_hr // 2
    half_search = search_size_hr // 2

    reference_clean = layout[
        target_center_y_hr - half_reference:target_center_y_hr + half_reference,
        target_center_x_hr - half_reference:target_center_x_hr + half_reference
    ]

    search_hr = layout[
        6000 - half_search:6000 + half_search,
        6000 - half_search:6000 + half_search
    ]

    search_clean = cv2.resize(
        search_hr,
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

    reference = add_independent_noise(
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

    search = add_independent_noise(
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

    pair_id = f"variation_{pair_index:03d}"

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


layout = create_dram_layout_with_variation()

dataset_rng = np.random.default_rng(2027)

metadata_rows = []

for index in range(30):
    item = create_pair(
        index,
        layout,
        dataset_rng
    )

    metadata_rows.append(item)

    print(
        f"Created {item['pair_id']} "
        f"at ({item['true_x']}, {item['true_y']})"
    )

metadata_path = os.path.join(
    PROJECT_FOLDER,
    "data",
    "variation_dataset_metadata.csv"
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
print("Variation dataset complete.")
print("Pairs created:", len(metadata_rows))
print("Metadata:", metadata_path)
