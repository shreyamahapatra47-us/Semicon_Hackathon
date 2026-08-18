# Semicon_Hackathon
DRIFT-SENSE: Periodicity-Aware Semiconductor Navigation Recovery

PROJECT PURPOSE
This project locates a high-magnification semiconductor reference region inside a lower-magnification search image. The output is the predicted center coordinate (x, y) in the search image.

METHOD
The solution uses:
- Expected 10x magnification relation between reference and search images
- Multi-scale normalized cross-correlation
- Multi-angle template matching for small rotational mismatch
- Large-context templates to distinguish repeated semiconductor structures
- Periodicity-aware ambiguity detection
- Center-nearest tie-break only for effectively equal candidates

INSTALLATION
Use Python 3.11 or newer.

Windows Command Prompt:
py -3.11 -m pip install -r requirements.txt

RUN INFERENCE
Open Command Prompt in this project folder and run:

py -3.11 inference.py --reference PATH_TO_REFERENCE_IMAGE --search PATH_TO_SEARCH_IMAGE

Example:
py -3.11 inference.py --reference data\reference_context_large\large_000.png --search data\search_context_large\large_000.png --output results\prediction.json

OUTPUT
The inference script prints JSON with:
- x: predicted horizontal coordinate
- y: predicted vertical coordinate
- best_score: normalized correlation score
- ambiguity_ratio: second-best score divided by best score
- near_best_candidate_count: number of essentially tied candidate locations
- selected_scale_multiplier: selected scale
- selected_angle_degrees: selected rotation

SYNTHETIC DATA GENERATOR
The generator models DRAM-like periodic layouts using:
- Horizontal word-line-like structures
- Vertical bit-line-like structures
- Via/contact-like structures
- Coarse sub-array separators
- Local low-frequency contextual regions
- Sparse routing-density context
- Independent Poisson-like and Gaussian-like noise
- Edge brightening
- Blur/defocus
- Rotation from -1 to +1 degrees

EVALUATION
Large-context synthetic dataset:
- 30 reference-search pairs
- 96.67% accuracy within 5 pixels
- 96.67% accuracy within 10 pixels
- Median localization error: 0.0 pixels
- Mean runtime: 387.96 ms per image pair
- One retained periodic-ambiguity failure case

LIMITATION
A purely periodic semiconductor region can be mathematically ambiguous when multiple regions provide the same visible evidence. The system explicitly reports ambiguity and applies the center-nearest selection rule only when candidate scores are nearly identical.

IMPORTANT FILES
inference.py
requirements.txt
src\generate_context_dataset_large_reference.py
src\evaluate_large_context.py
results\large_context_evaluation.csv
results\large_context_success_case.png
results\large_context_failure_case.png
## Final deliverables

The final reproducible pipeline is:

- `generate_dataset.py`
- `inference.py`
- `evaluate_final_dataset.py`
- `data/final_dataset/`
- `results/final_dataset_evaluation.csv`

The `src/` directory and other data/result folders contain earlier development experiments and ablation studies. They are retained for transparency but are not the final reported method.
