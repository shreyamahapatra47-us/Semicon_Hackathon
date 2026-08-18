import argparse
import csv
import os
from pathlib import Path

import cv2
import numpy as np


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


def scale_image(image, scale_factor):
    height, width = image.shape

    matrix = cv2.getRotationMatrix2D(
        (width / 2.0, height / 2.0),
        0.0,
        scale_factor
    )

    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )


def transform_point(x, y, image_center, rotation_degrees, scale_factor):
    matrix = cv2.getRotationMatrix2D(
        image_center,
        rotation_degrees,
        scale_factor
    )

    transformed = matrix @ np.array(
        [x, y, 1.0],
        dtype=np.float32
    )

    return float(transformed[0]), float(transformed[1])


def create_dram_layout(image_size, rng):
    image = np.full((image_size, image_size), 35, dtype=np.uint8)

    horizontal_pitch = int(rng.integers(72, 90))
    vertical_pitch = int(rng.integers(62, 80))

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
            cv2.circle(
                image,
                (x, y),
                int(rng.integers(4, 7)),
                int(rng.integers(180, 236)),
                -1
            )

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

    add_context_features(image, rng)
    return image


def create_finfet_layout(image_size, rng):
    image = np.full((image_size, image_size), 30, dtype=np.uint8)

    fin_pitch = int(rng.integers(42, 58))
    gate_pitch = int(rng.integers(180, 260))

    for x in range(30, image_size, fin_pitch):
        cv2.line(
            image,
            (x, 0),
            (x, image_size - 1),
            int(rng.integers(125, 170)),
            int(rng.integers(3, 6))
        )

    for y in range(100, image_size, gate_pitch):
        cv2.rectangle(
            image,
            (0, y),
            (image_size - 1, min(y + int(rng.integers(14, 25)), image_size - 1)),
            int(rng.integers(160, 215)),
            -1
        )

    for y in range(250, image_size, 1500):
        cv2.rectangle(
            image,
            (0, y),
            (image_size - 1, min(y + 30, image_size - 1)),
            70,
            -1
        )

    add_context_features(image, rng)
    return image


def add_context_features(image, rng):
    image_size = image.shape[0]

    for _ in range(70):
        center_x = int(rng.integers(500, image_size - 500))
        center_y = int(rng.integers(500, image_size - 500))
        patch_size = int(rng.integers(250, 650))
        half = patch_size // 2

        x1 = max(0, center_x - half)
        x2 = min(image_size, center_x + half)
        y1 = max(0, center_y - half)
        y2 = min(image_size, center_y + half)

        patch_h = y2 - y1
        patch_w = x2 - x1

        mask = np.zeros((patch_h, patch_w), dtype=np.uint8)

        cv2.ellipse(
            mask,
            (patch_w // 2, patch_h // 2),
            (
                int(patch_w * rng.uniform(0.25, 0.45)),
                int(patch_h * rng.uniform(0.25, 0.45))
            ),
            float(rng.uniform(0, 180)),
            0,
            360,
            255,
            -1
        )

        mask = cv2.GaussianBlur(
            mask,
            (0, 0),
            sigmaX=float(rng.uniform(20, 50))
        )

        intensity_shift = float(rng.uniform(-50, 50))

        patch = image[y1:y2, x1:x2].astype(np.float32)
        patch = patch + intensity_shift * (
            mask.astype(np.float32) / 255.0
        )

        image[y1:y2, x1:x2] = np.clip(
            patch,
            0,
            255
        ).astype(np.uint8)

    for _ in range(35):
        x1 = int(rng.integers(100, image_size - 700))
        y1 = int(rng.integers(100, image_size - 700))
        length = int(rng.integers(200, 700))
        width = int(rng.integers(10, 35))
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


def build_pair(
    pair_index,
    layout,
    architecture,
    output_dir,
    reference_size_hr,
    rng
):
    image_size = layout.shape[0]
    search_size_hr = 10000
    search_center_hr = image_size // 2
    half_search = search_size_hr // 2
    half_reference = reference_size_hr // 2

    offset_x = int(rng.integers(-100, 101))
    offset_y = int(rng.integers(-100, 101))

    target_x_hr = search_center_hr + offset_x * 10
    target_y_hr = search_center_hr + offset_y * 10

    reference_clean = layout[
        target_y_hr - half_reference:target_y_hr + half_reference,
        target_x_hr - half_reference:target_x_hr + half_reference
    ]

    search_hr = layout[
        search_center_hr - half_search:search_center_hr + half_search,
        search_center_hr - half_search:search_center_hr + half_search
    ]

    search_clean = cv2.resize(
        search_hr,
        (1000, 1000),
        interpolation=cv2.INTER_AREA
    )

    rotation_angle = float(rng.uniform(-1.0, 1.0))
    scale_factor = float(rng.uniform(0.97, 1.03))

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

    search = scale_image(search, scale_factor)
    search = rotate_image(search, rotation_angle)

    search = add_independent_noise(
        search,
        float(rng.uniform(5.0, 9.0)),
        float(rng.uniform(0.9, 1.5)),
        search_rng
    )

    target_x_before_transform = 500.0 + offset_x
    target_y_before_transform = 500.0 + offset_y

    true_x, true_y = transform_point(
        target_x_before_transform,
        target_y_before_transform,
        (500.0, 500.0),
        rotation_angle,
        scale_factor
    )

    pair_id = f"{architecture.lower()}_{pair_index:03d}"

    reference_dir = output_dir / "reference"
    search_dir = output_dir / "search"
    label_dir = output_dir / "labels"

    reference_dir.mkdir(parents=True, exist_ok=True)
    search_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(
        str(reference_dir / f"{pair_id}.png"),
        reference
    )

    cv2.imwrite(
        str(search_dir / f"{pair_id}.png"),
        search
    )

    with open(
        label_dir / f"{pair_id}.txt",
        "w",
        encoding="utf-8"
    ) as file:
        file.write(f"x={int(round(true_x))}\n")
        file.write(f"y={int(round(true_y))}\n")

    return {
        "pair_id": pair_id,
        "architecture": architecture,
        "reference_size_hr": reference_size_hr,
        "true_x": int(round(true_x)),
        "true_y": int(round(true_y)),
        "rotation_degrees": round(rotation_angle, 4),
        "search_scale_jitter": round(scale_factor, 4)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic semiconductor navigation pairs."
    )

    parser.add_argument(
        "--architecture",
        choices=["DRAM", "FinFET"],
        default="DRAM",
        help="Synthetic layout family."
    )

    parser.add_argument(
        "--num-pairs",
        type=int,
        default=30,
        help="Number of reference/search pairs."
    )

    parser.add_argument(
        "--output-dir",
        default="data/final_dataset",
        help="Dataset output directory."
    )

    parser.add_argument(
        "--reference-size",
        type=int,
        choices=[1000, 3000],
        default=3000,
        help="High-resolution reference side length in pixels."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=5050,
        help="Random seed for reproducibility."
    )

    args = parser.parse_args()

    if args.num_pairs <= 0:
        raise ValueError("--num-pairs must be positive.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    image_size = 11000

    if args.architecture == "DRAM":
        layout = create_dram_layout(image_size, rng)
    else:
        layout = create_finfet_layout(image_size, rng)

    metadata_rows = []

    print("Generating dataset...")
    print("Architecture:", args.architecture)
    print("Pairs:", args.num_pairs)
    print("Reference size:", args.reference_size)
    print("Output:", output_dir)

    for index in range(args.num_pairs):
        item = build_pair(
            index,
            layout,
            args.architecture,
            output_dir,
            args.reference_size,
            rng
        )

        metadata_rows.append(item)

        print(
            f"Created {item['pair_id']} "
            f"at ({item['true_x']}, {item['true_y']}) "
            f"rotation={item['rotation_degrees']} "
            f"scale={item['search_scale_jitter']}"
        )

    metadata_path = output_dir / "metadata.csv"

    with open(
        metadata_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=metadata_rows[0].keys()
        )

        writer.writeheader()
        writer.writerows(metadata_rows)

    print()
    print("Dataset generation complete.")
    print("Metadata:", metadata_path)


if __name__ == "__main__":
    main()
