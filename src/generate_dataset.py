import os
import csv
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

    enhanced = image_float + strength * magnitude

    return np.clip(enhanced, 0, 255).astype(np.uint8)


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

    noisy = poisson_image + gaussian_noise

    return np.clip(noisy, 0, 255).astype(np.uint8)


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


def rotate_point(x, y, angle_degrees, center_x=500.0, center_y=500.0):
    matrix = cv2.getRotationMatrix2D(
        (center_x, center_y),
        angle_degrees,
        1.0
    )

    point = np.array(
        [x, y, 1.0],
        dtype=np.float32
    )

    transformed = matrix @ point

    return float(transformed[0]), float(transformed[1])


def create_pair(pair_index, layout, rng):
    # Random target offset in search coordinates.
    # Keep it near search center, following the expected navigation context.
    offset_x = int(rng.integers(-120, 121))
    offset_y = int(rng.integers(-120, 121))

    target_center_x_hr = 6000 + offset_x * 10
    target_center_y_hr = 6000 + offset_y * 10

    search_center_x_hr = 6000
    search_center_y_hr = 6000

    reference_size_hr = 1000
    search_size_hr = 10000

    half_reference = reference_size_hr // 2
    half_search = search_size_hr // 2

    reference_clean = layout[
        target_center_y_hr - half_reference:target_center_y_hr + half_reference,
        target_center_x_hr - half_reference:target_center_x_hr + half_reference
    ]

    search_high_resolution = layout[
        search_center_y_hr - half_search:search_center_y_hr + half_search,
        search_center_x_hr - half_search:search_center_x_hr + half_search
    ]

    search_clean = cv2.resize(
        search_high_resolution,
        (1000, 1000),
        interpolation=cv2.INTER_AREA
    )

    # Random realistic capture parameters.
    reference_edge_strength = float(rng.uniform(0.12, 0.25))
    search_edge_strength = float(rng.uniform(0.25, 0.45))

    reference_gaussian_noise = float(rng.uniform(2.0, 4.0))
    search_gaussian_noise = float(rng.uniform(5.0, 9.0))

    reference_poisson_strength = float(rng.uniform(1.5, 2.5))
    search_poisson_strength = float(rng.uniform(0.9, 1.5))

    rotation_angle = float(rng.uniform(-1.0, 1.0))

    reference_seed = int(rng.integers(0, 1_000_000))
    search_seed = int(rng.integers(0, 1_000_000))

    reference_rng = np.random.default_rng(reference_seed)
    search_rng = np.random.default_rng(search_seed)

    reference = add_edge_brightening(
        reference_clean,
        reference_edge_strength
    )

    reference = cv2.GaussianBlur(
        reference,
        (3, 3),
        float(rng.uniform(0.3, 0.8))
    )

    reference = add_independent_noise(
        reference,
        reference_gaussian_noise,
        reference_poisson_strength,
        reference_rng
    )

    search = add_edge_brightening(
        search_clean,
        search_edge_strength
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
        search_gaussian_noise,
        search_poisson_strength,
        search_rng
    )

    # Target coordinate before low-magnification search rotation.
    target_x_before_rotation = 500.0 + offset_x
    target_y_before_rotation = 500.0 + offset_y

    true_x, true_y = rotate_point(
        target_x_before_rotation,
        target_y_before_rotation,
        rotation_angle
    )

    pair_id = f"batch_{pair_index:03d}"

    reference_path = os.path.join(
        REFERENCE_FOLDER,
        f"{pair_id}.png"
    )

    search_path = os.path.join(
        SEARCH_FOLDER,
        f"{pair_id}.png"
    )

    label_path = os.path.join(
        LABEL_FOLDER,
        f"{pair_id}.txt"
    )

    cv2.imwrite(reference_path, reference)
    cv2.imwrite(search_path, search)

    with open(label_path, "w", encoding="utf-8") as file:
        file.write(f"x={int(round(true_x))}\n")
        file.write(f"y={int(round(true_y))}\n")

    return {
        "pair_id": pair_id,
        "true_x": int(round(true_x)),
        "true_y": int(round(true_y)),
        "rotation_degrees": round(rotation_angle, 3),
        "reference_noise_sigma": round(reference_gaussian_noise, 3),
        "search_noise_sigma": round(search_gaussian_noise, 3),
        "reference_noise_seed": reference_seed,
        "search_noise_seed": search_seed
    }


# One shared layout represents a semiconductor die region.
# Every pair samples a different target position from it.
layout = create_large_dram_layout()

dataset_rng = np.random.default_rng(2026)

number_of_pairs = 30

metadata_rows = []

for pair_index in range(number_of_pairs):
    metadata = create_pair(
        pair_index,
        layout,
        dataset_rng
    )

    metadata_rows.append(metadata)

    print(
        f"Created {metadata['pair_id']} "
        f"at ({metadata['true_x']}, {metadata['true_y']}) "
        f"with rotation {metadata['rotation_degrees']} degrees"
    )

metadata_path = os.path.join(
    PROJECT_FOLDER,
    "data",
    "dataset_metadata.csv"
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
print("Dataset generation complete.")
print("Pairs created:", number_of_pairs)
print("Metadata saved at:")
print(metadata_path)
