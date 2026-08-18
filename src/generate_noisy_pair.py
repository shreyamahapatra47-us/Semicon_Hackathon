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

    # Horizontal word-line-like structures
    for y in range(40, image_size, horizontal_pitch):
        cv2.line(
            image,
            (0, y),
            (image_size - 1, y),
            color=155,
            thickness=line_width
        )

    # Vertical bit-line-like structures
    for x in range(35, image_size, vertical_pitch):
        cv2.line(
            image,
            (x, 0),
            (x, image_size - 1),
            color=120,
            thickness=line_width
        )

    # Via/contact-like structures at intersections
    for y in range(40, image_size, horizontal_pitch):
        for x in range(35, image_size, vertical_pitch):
            cv2.circle(
                image,
                (x, y),
                via_radius,
                color=230,
                thickness=-1
            )

    # Coarser structures that remain visible after 10x downsampling
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


def add_edge_brightening(image, strength=0.35):
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

    gradient_magnitude = cv2.magnitude(gradient_x, gradient_y)

    gradient_magnitude = cv2.normalize(
        gradient_magnitude,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    enhanced = image_float + strength * gradient_magnitude

    return np.clip(enhanced, 0, 255).astype(np.uint8)


def add_independent_noise(image, gaussian_sigma, poisson_strength, rng):
    image_float = image.astype(np.float32)

    # Poisson-like electron/shot noise
    scaled_image = np.clip(
        image_float * poisson_strength,
        0,
        None
    )

    poisson_image = rng.poisson(scaled_image).astype(np.float32)
    poisson_image = poisson_image / poisson_strength

    # Gaussian-like electronics/read noise
    gaussian_noise = rng.normal(
        loc=0,
        scale=gaussian_sigma,
        size=image.shape
    ).astype(np.float32)

    noisy_image = poisson_image + gaussian_noise

    return np.clip(noisy_image, 0, 255).astype(np.uint8)


def rotate_image(image, angle_degrees):
    height, width = image.shape
    center = (width / 2.0, height / 2.0)

    transformation = cv2.getRotationMatrix2D(
        center,
        angle_degrees,
        1.0
    )

    return cv2.warpAffine(
        image,
        transformation,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )


# Separate random generators ensure reference and search noise differ.
rng_reference = np.random.default_rng(123)
rng_search = np.random.default_rng(456)

layout = create_large_dram_layout()

# Target center in the high-resolution physical layout.
target_center_x_hr = 6150
target_center_y_hr = 5820

# Physical center of the large search crop.
search_center_x_hr = 6000
search_center_y_hr = 6000

reference_size_hr = 1000
search_size_hr = 10000

half_reference = reference_size_hr // 2
half_search = search_size_hr // 2

# High-magnification 1000x1000 reference crop.
reference_clean = layout[
    target_center_y_hr - half_reference:target_center_y_hr + half_reference,
    target_center_x_hr - half_reference:target_center_x_hr + half_reference
]

# 10,000x10,000 physical search crop, later reduced to 1000x1000.
search_high_resolution = layout[
    search_center_y_hr - half_search:search_center_y_hr + half_search,
    search_center_x_hr - half_search:search_center_x_hr + half_search
]

search_clean = cv2.resize(
    search_high_resolution,
    (1000, 1000),
    interpolation=cv2.INTER_AREA
)

# Reference: relatively clean high-magnification image.
reference = add_edge_brightening(reference_clean, strength=0.20)

reference = cv2.GaussianBlur(
    reference,
    (3, 3),
    0.5
)

reference = add_independent_noise(
    reference,
    gaussian_sigma=3.0,
    poisson_strength=2.0,
    rng=rng_reference
)

# Search: noisier low-magnification image.
search = add_edge_brightening(search_clean, strength=0.35)

search = cv2.GaussianBlur(
    search,
    (5, 5),
    1.0
)

rotation_angle_degrees = 0.5

search = rotate_image(
    search,
    angle_degrees=rotation_angle_degrees
)

search = add_independent_noise(
    search,
    gaussian_sigma=7.0,
    poisson_strength=1.2,
    rng=rng_search
)

# Target coordinate in the 1000x1000 search image before rotation.
target_x_before_rotation = (
    500 + (target_center_x_hr - search_center_x_hr) / 10.0
)

target_y_before_rotation = (
    500 + (target_center_y_hr - search_center_y_hr) / 10.0
)

# Apply the same rotation used on the search image to the coordinate.
rotation_center = (500.0, 500.0)

rotation_matrix = cv2.getRotationMatrix2D(
    rotation_center,
    rotation_angle_degrees,
    1.0
)

point_before_rotation = np.array(
    [
        target_x_before_rotation,
        target_y_before_rotation,
        1.0
    ],
    dtype=np.float32
)

point_after_rotation = rotation_matrix @ point_before_rotation

true_x_search = int(round(point_after_rotation[0]))
true_y_search = int(round(point_after_rotation[1]))

reference_path = os.path.join(
    REFERENCE_FOLDER,
    "pair_002.png"
)

search_path = os.path.join(
    SEARCH_FOLDER,
    "pair_002.png"
)

label_path = os.path.join(
    LABEL_FOLDER,
    "pair_002.txt"
)

cv2.imwrite(reference_path, reference)
cv2.imwrite(search_path, search)

with open(label_path, "w", encoding="utf-8") as file:
    file.write(f"x={true_x_search}\n")
    file.write(f"y={true_y_search}\n")

print("Noisy reference-search pair created.")
print("Reference image:", reference_path)
print("Search image:", search_path)
print("Rotation applied to search image:", rotation_angle_degrees, "degrees")
print("True target center in search:", (true_x_search, true_y_search))
print("Reference noise seed: 123")
print("Search noise seed: 456")

cv2.imshow("Noisy reference: pair_002", reference)
cv2.imshow("Noisy search: pair_002", search)

cv2.waitKey(0)
cv2.destroyAllWindows()
