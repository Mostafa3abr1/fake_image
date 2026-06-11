# Maintenance Matching System - Backend Pipeline

This system checks whether an uploaded maintenance image matches the Arabic description written by the user. It is built as a modular library ready to be integrated into a Python-based backend API (e.g., FastAPI, Flask, Django).

## Pipeline

1. **Stage 1 Image Classifier**: Classifies images into:
   - `plumbing` (سباكة)
   - `electricity` (كهرباء)
   - `carpentry` (نجارة)
   - `painting` (نقاشة)
   - `irrelevant` (غير متعلق بالصيانة)

2. **Stage 2 Arabic Text Classifier**: Classifies Arabic descriptions into:
   - `سباكة`
   - `كهرباء`
   - `نجارة`
   - `نقاشة`
   - `غير متعلق بالصيانة`

3. **Decision Engine**: Combines both predictions using a rule-based logic to yield:
   - `MATCH` (متطابق)
   - `MISMATCH` (غير متطابق)
   - `UNCERTAIN` (غير مؤكد)

## Installation

Install dependencies:
```bash
pip install -r requirements.txt
```

## Backend Usage

Import and initialize the `MaintenancePipeline` inside your backend application (ideally once at startup, as loading models takes a few seconds):

```python
from pipeline import MaintenancePipeline

# Initialize once (thread-safe for inference)
pipeline = MaintenancePipeline()

# 1. Full prediction (Image + Text)
result = pipeline.predict(
    image_input=image_bytes, # Can be raw bytes, a PIL Image, or a file path
    description="الحنفية بتسرب ميه"
)
# Returns:
# {
#     "decision": "MATCH",
#     "reason": "categories_match",
#     "explanation_ar": "تخصص الصورة (سباكة) مطابق لتخصص النص (سباكة)، لذلك القرار MATCH.",
#     "image_class": "plumbing",
#     "image_category_ar": "سباكة",
#     "image_confidence": 0.9491,
#     "text_category": "سباكة",
#     "text_decision": "CATEGORY_MATCH",
#     "text_confidence": 0.9707,
#     "image_details": { ... },
#     "text_details": { ... }
# }

# 2. Image-only classification
image_result = pipeline.classify_image(image_bytes)

# 3. Text-only classification
text_result = pipeline.classify_text("الحنفية بتسرب ميه")

# 4. Health check endpoint integration
status = pipeline.health_check()
```

## Running Verification

To test that everything loads and runs correctly:
```bash
python verify_setup.py
```

## File Structure

```text
fake_image/
│
├── config.py                  # Centralized settings & thresholds
├── text_utils.py              # Arabic text normalization & keyword detection
├── pipeline.py                # Main backend-ready pipeline class
├── requirements.txt           # Dependency requirements
├── README.md                  # This instructions file
├── verify_setup.py            # Test execution verification script
│
└── models/
    ├── stage1/
    │   ├── best_model.pt      # Vision model weights
    │   └── best_model_info.json # Metadata info for image model
    │
    └── stage2/
        └── stage2_text_bundle.joblib # Fused text model bundle
```

