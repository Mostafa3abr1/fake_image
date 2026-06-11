"""
Maintenance Matching System - Text Processing Utilities
=======================================================
Arabic text normalization and keyword detection logic.
"""

import re
from typing import Optional

import numpy as np
import pandas as pd

from config import (
    POSITIVE_TEXT_CATEGORIES,
    KeywordDecision,
)


def normalize_arabic(text: Optional[str]) -> str:
    """
    Normalize Arabic text for consistent matching.
    
    Handles:
    - Hamza variants (أ إ آ -> ا)
    - Ta-marbuta (ة -> ه)
    - Alef-maksura (ى -> ي)
    - Waw-hamza / Ya-hamza (ؤ -> و, ئ -> ي)
    - Persian kaf (گ -> ك)
    - Diacritics removal (tashkeel)
    - Non-Arabic/non-alphanumeric characters removal
    - Whitespace normalization
    """
    text = "" if pd.isna(text) else str(text)
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    text = text.replace("ة", "ه").replace("ى", "ي")
    text = text.replace("ؤ", "و").replace("ئ", "ي")
    text = text.replace("گ", "ك")
    text = re.sub(r"[ًٌٍَُِّْـ]", "", text)
    text = re.sub(r"[^\u0600-\u06FFa-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def detect_keywords(text: str, normalized_keywords: dict) -> dict:
    """
    Detect maintenance-related keywords in normalized Arabic text.
    
    Scoring:
    - Multi-word keywords (>=2 words): 2.5 points
    - Long keywords (>=6 chars):       1.5 points
    - Medium keywords (>=4 chars):     1.2 points
    - Short keywords (<4 chars):       1.0 points
    
    Returns dict with decision, category, confidence, scores, hits.
    """
    norm = normalize_arabic(text)
    scores = {}
    hits = {}

    for cat, kws in normalized_keywords.items():
        score = 0.0
        cat_hits = []

        for kw in kws:
            if kw and kw in norm:
                cat_hits.append(kw)
                if len(kw.split()) >= 2:
                    score += 2.5
                elif len(kw) >= 6:
                    score += 1.5
                elif len(kw) >= 4:
                    score += 1.2
                else:
                    score += 1.0

        scores[cat] = score
        hits[cat] = cat_hits

    best_cat = max(scores, key=scores.get)
    best_score = scores[best_cat]
    total_score = sum(scores.values())

    sorted_scores = sorted(scores.values(), reverse=True)
    second_score = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
    margin = best_score - second_score

    if total_score == 0:
        return {
            "decision": KeywordDecision.NO_KEYWORD,
            "category": None,
            "confidence": 0.0,
            "scores": scores,
            "hits": hits,
            "best_score": 0.0,
            "second_score": 0.0,
            "margin": 0.0,
            "total_score": 0.0,
        }

    confidence = best_score / total_score
    decision = (
        KeywordDecision.KEYWORD_MATCH
        if best_score >= 1.0 and margin >= 0.5
        else KeywordDecision.KEYWORD_AMBIGUOUS
    )

    return {
        "decision": decision,
        "category": best_cat,
        "confidence": float(confidence),
        "scores": scores,
        "hits": hits,
        "best_score": float(best_score),
        "second_score": float(second_score),
        "margin": float(margin),
        "total_score": float(total_score),
    }


def keyword_feature_vector(text: str, normalized_keywords: dict) -> np.ndarray:
    """
    Build a numeric feature vector from keyword detection results.
    Used as additional features for the Stage 2 ML classifier.
    
    Output shape: (4*3 + 7,) = (19,)
    - 4 raw scores, 4 normalized scores, 4 hit counts
    - best_score, second_score, margin, total_score, confidence, has_keyword, is_no_keyword
    """
    r = detect_keywords(text, normalized_keywords)
    scores = r["scores"]

    raw_scores = [scores.get(c, 0.0) for c in POSITIVE_TEXT_CATEGORIES]
    total = sum(raw_scores)

    norm_scores = [s / total for s in raw_scores] if total > 0 else [0.0] * len(POSITIVE_TEXT_CATEGORIES)
    hit_counts = [len(r["hits"].get(c, [])) for c in POSITIVE_TEXT_CATEGORIES]

    has_keyword = 1.0 if total > 0 else 0.0
    is_no_keyword = 1.0 if total == 0 else 0.0

    features = (
        raw_scores
        + norm_scores
        + hit_counts
        + [
            r["best_score"],
            r["second_score"],
            r["margin"],
            r["total_score"],
            r["confidence"],
            has_keyword,
            is_no_keyword,
        ]
    )

    return np.array(features, dtype=np.float32)
