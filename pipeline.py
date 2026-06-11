"""
Maintenance Matching System - Pipeline
=======================================
Production-ready pipeline for backend integration.

Usage:
    from pipeline import MaintenancePipeline

    # Initialize once at app startup
    pipeline = MaintenancePipeline()

    # --- Option A: Full pipeline (image + text) ---
    result = pipeline.predict(image_bytes=b"...", description="الحنفية بتسرب ميه")
    # result["decision"]        -> "MATCH" | "MISMATCH" | "UNCERTAIN"
    # result["image_category"]  -> "plumbing"
    # result["text_category"]   -> "سباكة"

    # --- Option B: Image only ---
    img_result = pipeline.classify_image(image_bytes=b"...")

    # --- Option C: Text only ---
    txt_result = pipeline.classify_text("الحنفية بتسرب ميه")
"""

import io
import json
import logging
import tempfile
import warnings
from pathlib import Path
from typing import Optional, Union

import joblib
import numpy as np
import torch
import timm
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
from scipy.sparse import hstack, csr_matrix
from sentence_transformers import SentenceTransformer

from config import (
    BASE_DIR,
    STAGE1_DIR,
    STAGE1_INFO_FILE,
    STAGE1_WEIGHTS_FILE,
    STAGE2_BUNDLE_PATH,
    MINILM_DIR,
    IMAGE_CLASS_TO_AR,
    CLASS_DESCRIPTIONS,
    POSITIVE_TEXT_CATEGORIES,
    IRRELEVANT_TEXT_LABEL,
    IMAGE_SIZE,
    IMAGE_MEAN,
    IMAGE_STD,
    IMAGE_CONFIDENCE_THRESHOLD,
    TEXT_CONFIDENCE_THRESHOLD,
    MISMATCH_UNCERTAIN_THRESHOLD,
    KEYWORD_OVERRIDE_CONFIDENCE,
    MODEL_LOW_CONFIDENCE,
    KEYWORD_MODEL_CLOSE_MARGIN,
    KEYWORD_MODEL_CONFLICT_MARGIN,
    Decision,
    TextDecision,
    KeywordDecision,
)
from text_utils import normalize_arabic, detect_keywords, keyword_feature_vector

# Suppress sklearn version mismatch warnings in production
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

logger = logging.getLogger(__name__)


class MaintenancePipeline:
    """
    Production pipeline for the Maintenance Matching System.

    Provides three entry points:
    - predict()        -> full image + text matching
    - classify_image() -> image classification only
    - classify_text()  -> text classification only

    The pipeline is thread-safe for inference (no mutable state after init).
    """

    def __init__(self, base_dir: Optional[Union[str, Path]] = None):
        """
        Load all models into memory.

        Args:
            base_dir: Root directory containing the ``models/`` folder.
                      Defaults to the directory where this file lives.
        """
        self._base_dir = Path(base_dir) if base_dir else BASE_DIR
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("Initializing MaintenancePipeline (device=%s)", self._device)

        self._load_stage1()
        self._load_stage2()

        logger.info("MaintenancePipeline ready.")

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_stage1(self):
        """Load the Stage 1 image classification model."""
        info_path = self._base_dir / "models" / "stage1" / "best_model_info.json"
        if not info_path.exists():
            raise FileNotFoundError(f"Stage 1 info not found: {info_path}")

        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)

        self._stage1_type = info["type"]

        weights_path = Path(info.get("weights_path", ""))
        if not weights_path.exists() or not weights_path.is_absolute():
            weights_path = self._base_dir / "models" / "stage1" / "best_model.pt"

        if not weights_path.exists():
            raise FileNotFoundError(f"Stage 1 weights not found: {weights_path}")

        if self._stage1_type == "YOLO-cls":
            self._yolo_model = YOLO(str(weights_path))
            self._timm_model = None
            self._idx_to_class = None
            self._img_transform = None
            logger.info("Stage 1: YOLO-cls loaded")
        else:
            checkpoint = torch.load(weights_path, map_location=self._device, weights_only=False)
            arch = checkpoint.get("model_arch", info.get("model_arch", "resnet50"))
            self._idx_to_class = {int(k): v for k, v in checkpoint["idx_to_class"].items()}

            self._timm_model = timm.create_model(
                arch, pretrained=False, num_classes=len(self._idx_to_class)
            ).to(self._device)
            self._timm_model.load_state_dict(checkpoint["state_dict"])
            self._timm_model.eval()

            self._yolo_model = None
            self._img_transform = transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
            ])
            logger.info("Stage 1: timm/%s loaded (%d classes)", arch, len(self._idx_to_class))

    def _load_stage2(self):
        """Load the Stage 2 text classification bundle."""
        bundle_path = self._base_dir / "models" / "stage2" / "stage2_text_bundle.joblib"
        if not bundle_path.exists():
            raise FileNotFoundError(f"Stage 2 bundle not found: {bundle_path}")

        bundle = joblib.load(bundle_path)

        self._text_model = bundle["classifier"]
        self._label_encoder = bundle["label_encoder"]
        self._word_tfidf = bundle["word_tfidf"]
        self._char_tfidf = bundle["char_tfidf"]
        self._emb_scaler = bundle["embedding_scaler"]
        self._kw_scaler = bundle["keyword_scaler"]
        self._category_keywords = bundle["keywords"]
        self._stage2_info = bundle["info"]

        minilm_dir = self._base_dir / "models" / "minilm_model"
        if minilm_dir.exists():
            self._embedder = SentenceTransformer(str(minilm_dir))
        else:
            self._embedder = SentenceTransformer(bundle["embedding_model_name"])

        self._normalized_keywords = {
            cat: sorted(set(normalize_arabic(k) for k in kws if normalize_arabic(k)))
            for cat, kws in self._category_keywords.items()
        }

        logger.info("Stage 2: text classifier loaded (model=%s)", type(self._text_model).__name__)

    # ------------------------------------------------------------------
    # Image helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _bytes_to_pil(image_input: Union[bytes, Image.Image, str, Path]) -> Image.Image:
        """
        Convert various image inputs to a PIL Image.

        Accepts:
        - bytes / bytearray (raw image data)
        - PIL.Image.Image
        - str / Path (file path)
        """
        if isinstance(image_input, Image.Image):
            return image_input.convert("RGB")
        if isinstance(image_input, (bytes, bytearray)):
            return Image.open(io.BytesIO(image_input)).convert("RGB")
        if isinstance(image_input, (str, Path)):
            return Image.open(image_input).convert("RGB")
        raise TypeError(f"Unsupported image type: {type(image_input)}")

    # ------------------------------------------------------------------
    # Public API: Image Classification
    # ------------------------------------------------------------------

    def classify_image(self, image_input: Union[bytes, Image.Image, str, Path]) -> dict:
        """
        Classify a maintenance image.

        Args:
            image_input: Image as bytes, PIL Image, or file path.

        Returns:
            dict with keys:
            - image_class (str):       e.g. "plumbing"
            - image_category_ar (str): e.g. "سباكة"
            - confidence (float):      0.0 - 1.0
            - probabilities (list):    sorted list of {image_class, arabic_category, probability}
        """
        image = self._bytes_to_pil(image_input)

        if self._stage1_type == "YOLO-cls":
            return self._classify_image_yolo(image)
        return self._classify_image_timm(image)

    def _classify_image_yolo(self, image: Image.Image) -> dict:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            image.save(tmp.name)
            tmp_path = tmp.name

        result = self._yolo_model(tmp_path, verbose=False)[0]
        probs = result.probs.data.detach().cpu().numpy()
        names = result.names
        pred_idx = int(result.probs.top1)

        return self._format_image_result(
            pred_class=names[pred_idx],
            confidence=float(result.probs.top1conf),
            probs=probs,
            index_to_class=names,
        )

    def _classify_image_timm(self, image: Image.Image) -> dict:
        x = self._img_transform(image).unsqueeze(0).to(self._device)

        with torch.no_grad():
            logits = self._timm_model(x)
            probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

        pred_idx = int(np.argmax(probs))

        return self._format_image_result(
            pred_class=self._idx_to_class[pred_idx],
            confidence=float(probs[pred_idx]),
            probs=probs,
            index_to_class=self._idx_to_class,
        )

    @staticmethod
    def _format_image_result(pred_class: str, confidence: float, probs, index_to_class: dict) -> dict:
        prob_rows = [
            {
                "image_class": index_to_class[i],
                "arabic_category": IMAGE_CLASS_TO_AR.get(index_to_class[i], index_to_class[i]),
                "probability": round(float(p), 6),
            }
            for i, p in enumerate(probs)
        ]
        return {
            "image_class": pred_class,
            "image_category_ar": IMAGE_CLASS_TO_AR.get(pred_class, pred_class),
            "confidence": round(confidence, 6),
            "probabilities": sorted(prob_rows, key=lambda x: x["probability"], reverse=True),
        }

    # ------------------------------------------------------------------
    # Public API: Text Classification
    # ------------------------------------------------------------------

    def classify_text(self, text: str) -> dict:
        """
        Classify an Arabic maintenance description.

        Args:
            text: Arabic user description.

        Returns:
            dict with keys:
            - text_category (str|None): final category or None if uncertain
            - text_decision (str):      "CATEGORY_MATCH" | "IRRELEVANT_TEXT" | "UNCERTAIN"
            - text_reason (str):        human-readable reason code
            - ml_category (str):        raw ML model prediction
            - ml_confidence (float):    ML model confidence
            - keyword_category (str|None)
            - keyword_decision (str)
            - keyword_confidence (float)
            - keyword_hits (dict)
            - probabilities (list)
        """
        text_norm = normalize_arabic(text)

        # Build fused feature vector
        x_word = self._word_tfidf.transform([text_norm])
        x_char = self._char_tfidf.transform([text_norm])

        x_emb = self._embedder.encode([text_norm], normalize_embeddings=True, show_progress_bar=False)
        x_emb_scaled = self._emb_scaler.transform(x_emb)

        x_kw = np.vstack([keyword_feature_vector(text_norm, self._normalized_keywords)])
        x_kw_scaled = self._kw_scaler.transform(x_kw)

        x_all = hstack([x_word, x_char, csr_matrix(x_emb_scaled), csr_matrix(x_kw_scaled)]).tocsr()

        # ML prediction
        probs = self._text_model.predict_proba(x_all)[0]
        pred_idx = int(np.argmax(probs))
        ml_category = str(self._label_encoder.inverse_transform([pred_idx])[0])
        ml_confidence = float(probs[pred_idx])

        prob_rows = [
            {
                "text_category": str(self._label_encoder.inverse_transform([i])[0]),
                "probability": round(float(p), 6),
            }
            for i, p in enumerate(probs)
        ]

        # Keyword detection
        kw = detect_keywords(text_norm, self._normalized_keywords)

        # Arbitrate between ML model and keyword detector
        final_category, decision, reason = self._arbitrate_text(
            ml_category, ml_confidence, probs, kw
        )

        return {
            "text_category": final_category,
            "text_decision": decision,
            "text_reason": reason,
            "ml_category": ml_category,
            "ml_confidence": round(ml_confidence, 6),
            "keyword_category": kw["category"],
            "keyword_decision": kw["decision"],
            "keyword_confidence": round(kw["confidence"], 6),
            "keyword_hits": {k: v for k, v in kw["hits"].items() if len(v) > 0},
            "probabilities": sorted(prob_rows, key=lambda x: x["probability"], reverse=True),
        }

    def _arbitrate_text(self, ml_category, ml_confidence, probs, kw):
        """Resolve conflicts between ML model and keyword detector."""
        final_category = ml_category
        decision = (
            TextDecision.CATEGORY_MATCH
            if ml_category != IRRELEVANT_TEXT_LABEL
            else TextDecision.IRRELEVANT_TEXT
        )
        reason = "ml_model_selected"

        if kw["decision"] == KeywordDecision.KEYWORD_MATCH:
            kw_cat = kw["category"]
            kw_idx = int(self._label_encoder.transform([kw_cat])[0])
            kw_model_prob = float(probs[kw_idx])

            if kw_cat == ml_category:
                decision = TextDecision.CATEGORY_MATCH
                reason = "ml_and_keyword_agree"

            elif kw["confidence"] >= KEYWORD_OVERRIDE_CONFIDENCE and (
                ml_confidence < MODEL_LOW_CONFIDENCE
                or (ml_confidence - kw_model_prob) <= KEYWORD_MODEL_CLOSE_MARGIN
            ):
                final_category = kw_cat
                decision = TextDecision.CATEGORY_MATCH
                reason = "strong_keyword_overrode_model"

            elif abs(ml_confidence - kw_model_prob) <= KEYWORD_MODEL_CONFLICT_MARGIN:
                final_category = None
                decision = TextDecision.UNCERTAIN
                reason = "keyword_model_conflict_close_scores"

        else:
            if ml_category == IRRELEVANT_TEXT_LABEL:
                final_category = IRRELEVANT_TEXT_LABEL
                decision = TextDecision.IRRELEVANT_TEXT
                reason = "no_keyword_model_irrelevant"

            elif kw["decision"] == KeywordDecision.NO_KEYWORD and ml_confidence < MODEL_LOW_CONFIDENCE:
                final_category = None
                decision = TextDecision.UNCERTAIN
                reason = "no_keyword_low_model_confidence"

        return final_category, decision, reason

    # ------------------------------------------------------------------
    # Public API: Full Pipeline (Image + Text)
    # ------------------------------------------------------------------

    def predict(
        self,
        image_input: Union[bytes, Image.Image, str, Path],
        description: str,
    ) -> dict:
        """
        Run the full matching pipeline.

        Args:
            image_input: Image as bytes, PIL Image, or file path.
            description: Arabic user description of the maintenance issue.

        Returns:
            dict with keys:
            - decision (str):          "MATCH" | "MISMATCH" | "UNCERTAIN"
            - reason (str):            machine-readable reason code
            - explanation_ar (str):    Arabic explanation
            - image_class (str):       predicted image class
            - image_category_ar (str): Arabic image category
            - image_confidence (float)
            - text_category (str|None)
            - text_decision (str)
            - text_confidence (float)
            - image_details (dict):    full Stage 1 output
            - text_details (dict):     full Stage 2 output
        """
        image_result = self.classify_image(image_input)
        text_result = self.classify_text(description)
        engine_result = self._decide(image_result, text_result)

        return {
            # Top-level summary for backend consumption
            "decision": engine_result["final_decision"],
            "reason": engine_result["reason"],
            "explanation_ar": engine_result["explanation_ar"],

            # Key fields flattened for convenience
            "image_class": image_result["image_class"],
            "image_category_ar": image_result["image_category_ar"],
            "image_confidence": image_result["confidence"],
            "text_category": text_result["text_category"],
            "text_decision": text_result["text_decision"],
            "text_confidence": text_result["ml_confidence"],

            # Full details for debugging / logging
            "image_details": image_result,
            "text_details": text_result,
        }

    def _decide(self, image_result: dict, text_result: dict) -> dict:
        """Decision engine: compare image and text predictions."""
        img_class = image_result["image_class"]
        img_cat_ar = image_result["image_category_ar"]
        img_conf = image_result["confidence"]

        txt_cat = text_result["text_category"]
        txt_decision = text_result["text_decision"]
        txt_conf = text_result["ml_confidence"]

        # Rule 1: Image is irrelevant
        if img_class == "irrelevant":
            return self._decision(
                Decision.MISMATCH, "image_is_irrelevant",
                "الصورة ليس لها علاقة بالأعطال أو أعمال الصيانة، لذلك لا يمكن مطابقتها مع وصف المستخدم."
            )

        # Rule 2: Text is irrelevant
        if txt_cat == IRRELEVANT_TEXT_LABEL or txt_decision == TextDecision.IRRELEVANT_TEXT:
            return self._decision(
                Decision.MISMATCH, "text_is_irrelevant",
                "وصف المستخدم غير متعلق بالصيانة، لذلك القرار MISMATCH."
            )

        # Rule 3: Text is uncertain
        if txt_cat is None or txt_decision == TextDecision.UNCERTAIN:
            return self._decision(
                Decision.UNCERTAIN, "text_category_uncertain",
                "النظام غير واثق من تخصص النص، لذلك القرار UNCERTAIN."
            )

        # Rule 4: Low image confidence
        if img_conf < IMAGE_CONFIDENCE_THRESHOLD:
            return self._decision(
                Decision.UNCERTAIN, "low_image_confidence",
                f"ثقة موديل الصورة منخفضة ({img_conf:.3f})، لذلك القرار UNCERTAIN."
            )

        # Rule 5: Low text confidence
        if txt_conf is not None and txt_conf < TEXT_CONFIDENCE_THRESHOLD:
            return self._decision(
                Decision.UNCERTAIN, "low_text_confidence",
                f"ثقة موديل النص منخفضة ({txt_conf:.3f})، لذلك القرار UNCERTAIN."
            )

        # Rule 6: Categories match
        if img_cat_ar == txt_cat:
            return self._decision(
                Decision.MATCH, "categories_match",
                f"تخصص الصورة ({img_cat_ar}) مطابق لتخصص النص ({txt_cat})، لذلك القرار MATCH."
            )

        # Rule 7: Categories differ but confidence not high enough
        if img_conf < MISMATCH_UNCERTAIN_THRESHOLD or (
            txt_conf is not None and txt_conf < MISMATCH_UNCERTAIN_THRESHOLD
        ):
            return self._decision(
                Decision.UNCERTAIN, "category_mismatch_low_confidence",
                f"تخصص الصورة ({img_cat_ar}) مختلف عن تخصص النص ({txt_cat})، "
                f"لكن الثقة ليست عالية كفاية، لذلك القرار UNCERTAIN."
            )

        # Rule 8: Clear mismatch
        return self._decision(
            Decision.MISMATCH, "categories_do_not_match",
            f"تخصص الصورة ({img_cat_ar}) مختلف عن تخصص النص ({txt_cat})، لذلك القرار MISMATCH."
        )

    @staticmethod
    def _decision(decision: Decision, reason: str, explanation_ar: str) -> dict:
        return {
            "final_decision": decision.value,
            "reason": reason,
            "explanation_ar": explanation_ar,
        }

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def health_check(self) -> dict:
        """
        Quick health check to verify all models are loaded.
        Useful for backend /health endpoints.
        """
        return {
            "status": "ok",
            "device": self._device,
            "stage1_type": self._stage1_type,
            "stage2_model": type(self._text_model).__name__,
            "categories": list(IMAGE_CLASS_TO_AR.keys()),
        }
