import argparse
import json

import cv2
import numpy as np


BASE_SCALE = 0.10

# Covers generator magnification jitter from 0.97 to 1.03.
SCALE_MULTIPLIERS = [
    0.97,
    0.98,
    0.99,
    1.00,
    1.01,
    1.02,
    1.03
]

# Covers the synthetic rotation variation range.
ANGLES_DEGREES = [
    -1.0,
    -0.5,
    0.0,
    0.5,
    1.0
]

# Only use the center-nearest rule when candidates are essentially tied.
TIE_SCORE_RATIO = 0.998


def read_grayscale(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(
            f"Could not read image: {image_path}"
        )

    return image


def normalize_image(image):
    image = image.astype(np.float32)
    image = image - np.mean(image)

    standard_deviation = np.std(image)

    if standard_deviation > 1e-6:
        image = image / standard_deviation

    return image


def rotate_template(image, angle_degrees):
    height, width = image.shape

    rotation_matrix = cv2.getRotationMatrix2D(
        (width / 2.0, height / 2.0),
        angle_degrees,
        1.0
    )

    return cv2.warpAffine(
        image,
        rotation_matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )


def find_candidate_peaks(
    correlation_map,
    maximum_score,
    threshold_ratio,
    suppression_radius,
    maximum_candidates=10
):
    working_map = correlation_map.copy()
    threshold = maximum_score * threshold_ratio

    candidates = []

    for _ in range(maximum_candidates):
        _, peak_score, _, peak_location = cv2.minMaxLoc(
            working_map
        )

        if peak_score < threshold:
            break

        x, y = peak_location

        candidates.append({
            "top_left_x": int(x),
            "top_left_y": int(y),
            "score": float(peak_score)
        })

        x_start = max(0, x - suppression_radius)
        x_end = min(
            working_map.shape[1],
            x + suppression_radius + 1
        )

        y_start = max(0, y - suppression_radius)
        y_end = min(
            working_map.shape[0],
            y + suppression_radius + 1
        )

        working_map[y_start:y_end, x_start:x_end] = -1.0

    return candidates


def localize(reference_image, search_image):
    reference_height, reference_width = reference_image.shape
    search_height, search_width = search_image.shape

    expected_width = max(
        20,
        int(round(reference_width * BASE_SCALE))
    )

    expected_height = max(
        20,
        int(round(reference_height * BASE_SCALE))
    )

    search_normalized = normalize_image(search_image)
    all_candidates = []

    for scale_multiplier in SCALE_MULTIPLIERS:
        template_width = max(
            20,
            int(round(expected_width * scale_multiplier))
        )

        template_height = max(
            20,
            int(round(expected_height * scale_multiplier))
        )

        if (
            template_width >= search_width
            or template_height >= search_height
        ):
            continue

        scaled_reference = cv2.resize(
            reference_image,
            (template_width, template_height),
            interpolation=cv2.INTER_AREA
        )

        for angle in ANGLES_DEGREES:
            template = rotate_template(
                scaled_reference,
                angle
            )

            template_normalized = normalize_image(template)

            correlation_map = cv2.matchTemplate(
                search_normalized,
                template_normalized,
                cv2.TM_CCOEFF_NORMED
            )

            _, maximum_score, _, _ = cv2.minMaxLoc(
                correlation_map
            )

            suppression_radius = max(
                20,
                min(template_width, template_height) // 5
            )

            candidates = find_candidate_peaks(
                correlation_map,
                maximum_score,
                threshold_ratio=TIE_SCORE_RATIO,
                suppression_radius=suppression_radius,
                maximum_candidates=10
            )

            for candidate in candidates:
                candidate["scale_multiplier"] = float(
                    scale_multiplier
                )

                candidate["angle_degrees"] = float(angle)

                candidate["template_width"] = int(
                    template_width
                )

                candidate["template_height"] = int(
                    template_height
                )

                all_candidates.append(candidate)

    if not all_candidates:
        raise RuntimeError(
            "No valid matching candidates found."
        )

    best_score = max(
        candidate["score"]
        for candidate in all_candidates
    )

    near_best_candidates = [
        candidate
        for candidate in all_candidates
        if candidate["score"] >= best_score * TIE_SCORE_RATIO
    ]

    search_center_x = search_width / 2.0
    search_center_y = search_height / 2.0

    for candidate in near_best_candidates:
        candidate["center_x"] = (
            candidate["top_left_x"]
            + candidate["template_width"] / 2.0
        )

        candidate["center_y"] = (
            candidate["top_left_y"]
            + candidate["template_height"] / 2.0
        )

        candidate["distance_to_search_center"] = float(
            np.sqrt(
                (candidate["center_x"] - search_center_x) ** 2
                + (candidate["center_y"] - search_center_y) ** 2
            )
        )

    # Required problem-statement tie-break:
    # choose image-center-nearest candidate only among nearly equal matches.
    selected = min(
        near_best_candidates,
        key=lambda candidate: candidate[
            "distance_to_search_center"
        ]
    )

    sorted_scores = sorted(
        [candidate["score"] for candidate in all_candidates],
        reverse=True
    )

    if len(sorted_scores) >= 2:
        ambiguity_ratio = float(
            sorted_scores[1] / max(sorted_scores[0], 1e-12)
        )
    else:
        ambiguity_ratio = 0.0

    return {
        "x": int(round(selected["center_x"])),
        "y": int(round(selected["center_y"])),
        "best_score": round(float(best_score), 6),
        "ambiguity_ratio": round(ambiguity_ratio, 6),
        "near_best_candidate_count": len(
            near_best_candidates
        ),
        "selected_scale_multiplier": selected[
            "scale_multiplier"
        ],
        "selected_angle_degrees": selected[
            "angle_degrees"
        ],
        "template_width": selected["template_width"],
        "template_height": selected["template_height"]
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Locate a high-magnification reference image "
            "inside a lower-magnification search image."
        )
    )

    parser.add_argument(
        "--reference",
        required=True,
        help="Path to reference image."
    )

    parser.add_argument(
        "--search",
        required=True,
        help="Path to search image."
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output file path."
    )

    args = parser.parse_args()

    reference = read_grayscale(args.reference)
    search = read_grayscale(args.search)

    prediction = localize(reference, search)

    print(json.dumps(prediction, indent=2))

    if args.output:
        with open(
            args.output,
            "w",
            encoding="utf-8"
        ) as output_file:
            json.dump(
                prediction,
                output_file,
                indent=2
            )

        print()
        print("Prediction JSON saved to:")
        print(args.output)


if __name__ == "__main__":
    main()
