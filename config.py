"""
Maintenance Matching System - Configuration
============================================
Central configuration for model paths, thresholds, and category mappings.
All magic numbers and paths are defined here for easy tuning and deployment.
"""

from pathlib import Path
from enum import Enum


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODELS_DIR = BASE_DIR / "models"
STAGE1_DIR = MODELS_DIR / "stage1"
STAGE2_BUNDLE_PATH = MODELS_DIR / "stage2" / "stage2_text_bundle.joblib"
MINILM_DIR = MODELS_DIR / "minilm_model"

STAGE1_INFO_FILE = STAGE1_DIR / "best_model_info.json"
STAGE1_WEIGHTS_FILE = STAGE1_DIR / "best_model.pt"


# ============================================================
# Decision Enums
# ============================================================

class Decision(str, Enum):
    """Final pipeline decision."""
    MATCH = "MATCH"
    MISMATCH = "MISMATCH"
    UNCERTAIN = "UNCERTAIN"


class TextDecision(str, Enum):
    """Stage 2 text sub-decision."""
    CATEGORY_MATCH = "CATEGORY_MATCH"
    IRRELEVANT_TEXT = "IRRELEVANT_TEXT"
    UNCERTAIN = "UNCERTAIN"


class KeywordDecision(str, Enum):
    """Keyword detector sub-decision."""
    KEYWORD_MATCH = "KEYWORD_MATCH"
    KEYWORD_AMBIGUOUS = "KEYWORD_AMBIGUOUS"
    NO_KEYWORD = "NO_KEYWORD"


# ============================================================
# Category Mappings
# ============================================================

# English image class -> Arabic maintenance category
IMAGE_CLASS_TO_AR = {
    "plumbing":    "سباكة",
    "electricity": "كهرباء",
    "carpentry":   "نجارة",
    "painting":    "نقاشة",
    "irrelevant":  "irrelevant",
}

# Human-readable descriptions
CLASS_DESCRIPTIONS = {
    "plumbing":    "سباكة / Plumbing",
    "electricity": "كهرباء / Electricity",
    "carpentry":   "نجارة / Carpentry",
    "painting":    "نقاشة / Painting",
    "irrelevant":  "صورة ليس لها علاقة بالأعطال / Irrelevant",
}

# The 4 positive (maintenance-related) text categories
POSITIVE_TEXT_CATEGORIES = ["سباكة", "كهرباء", "نجارة", "نقاشة"]

# The irrelevant text label used by the trained model
IRRELEVANT_TEXT_LABEL = "غير متعلق بالصيانة"


# ============================================================
# Decision Engine Thresholds
# ============================================================

IMAGE_CONFIDENCE_THRESHOLD = 0.60
TEXT_CONFIDENCE_THRESHOLD = 0.55
MISMATCH_UNCERTAIN_THRESHOLD = 0.65

# Stage 2 keyword-vs-model arbitration thresholds
KEYWORD_OVERRIDE_CONFIDENCE = 0.65
MODEL_LOW_CONFIDENCE = 0.55
KEYWORD_MODEL_CLOSE_MARGIN = 0.18
KEYWORD_MODEL_CONFLICT_MARGIN = 0.10


# ============================================================
# Image Pre-processing
# ============================================================

IMAGE_SIZE = 224
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]
