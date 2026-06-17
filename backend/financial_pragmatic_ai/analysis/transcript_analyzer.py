from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import logging
from pathlib import Path
import re
from collections import Counter
import threading

import torch
from huggingface_hub import hf_hub_download

from financial_pragmatic_ai.analysis.financial_signal_engine import (
    compute_risk_score,
    derive_signal,
)
from financial_pragmatic_ai.analysis.transcript_parser import parse_transcript
from financial_pragmatic_ai.models.conversation_attention_model import (
    INDEX_TO_SIGNAL,
    load_conversation_attention_model,
)
from financial_pragmatic_ai.models.financial_pragmatic_transformer_v2 import (
    FinancialPragmaticTransformer,
)
from financial_pragmatic_ai.models.finbert_intent_model import FinBERTIntentModel


# ---------------------------------------------------------------------------
# Models are loaded from HuggingFace Hub.
# No local model files required.
# Primary:  SarcoNarco/finbert_intent_v3
# Fallback: SarcoNarco/financial-models / pragmatic_transformer_trained.pt
# ---------------------------------------------------------------------------

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
CONVERSATION_ATTENTION_PATH = MODELS_DIR / "conversation_attention.pt"

_HF_FALLBACK_REPO = "SarcoNarco/financial-models"
_HF_FALLBACK_FILENAME = "pragmatic_transformer_trained.pt"
_FALLBACK_LOAD_TIMEOUT_SECONDS = 20

logger = logging.getLogger(__name__)

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

SPEAKER_ENCODING = {
    "CEO": [1.0, 0.0, 0.0],
    "CFO": [0.0, 1.0, 0.0],
    "ANALYST": [0.0, 0.0, 1.0],
}


def _speaker_vector(speaker: str) -> torch.Tensor:
    values = SPEAKER_ENCODING.get(speaker.upper(), [0.0, 0.0, 0.0])
    return torch.tensor(values, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Post-prediction intent correction
# ---------------------------------------------------------------------------

_HAS_PCT = re.compile(r"\d+(?:\.\d+)?\s*%|\bpercent\b", re.IGNORECASE)
_GROWTH_WORDS = re.compile(
    r"\b(growth|increase|increased|rose|grew|grow|up|gain|gains|raised|record|expansion|improved)\b",
    re.IGNORECASE,
)
_DECLINE_WORDS = re.compile(
    r"\b(decline|declined|decrease|decreased|fell|fall|down|drop|dropped|lower|loss)\b",
    re.IGNORECASE,
)
_COST_WORDS = re.compile(
    r"\b(cost|margin|pressure|inflation|headwind|compression|uncertainty|challenging)\b",
    re.IGNORECASE,
)
_QUESTION = re.compile(r"\?\s*$")

# Confidence thresholds
_HIGH_CONFIDENCE = 0.90   # don't override numeric rules above this
_LOW_CONFIDENCE  = 0.60   # only apply soft cost rule below this


def _correct_intent(text: str, intent: str, confidence: float) -> str:
    """
    Apply deterministic post-processing rules after model prediction.

    Rules (applied in order, earliest match wins):
      1. Question text  → never override, return as-is.
      2. % + growth word + confidence ≤ HIGH  → EXPANSION
      3. % + decline word + confidence ≤ HIGH → COST_PRESSURE
      4. cost/margin keyword + confidence < LOW → COST_PRESSURE
    """
    # Safety: never override questions
    if _QUESTION.search(text.strip()):
        return intent

    has_pct = bool(_HAS_PCT.search(text))

    # Rule 1 — numeric growth signal
    if has_pct and _GROWTH_WORDS.search(text) and confidence <= _HIGH_CONFIDENCE:
        new_intent = "EXPANSION"
        if new_intent != intent:
            logger.debug("Corrected intent %s -> %s text=%s", intent, new_intent, text[:60])
        return new_intent

    # Rule 2 — numeric decline signal
    if has_pct and _DECLINE_WORDS.search(text) and confidence <= _HIGH_CONFIDENCE:
        new_intent = "COST_PRESSURE"
        if new_intent != intent:
            logger.debug("Corrected intent %s -> %s text=%s", intent, new_intent, text[:60])
        return new_intent

    # Rule 3 — soft cost bias (low-confidence model only)
    if _COST_WORDS.search(text) and confidence < _LOW_CONFIDENCE:
        new_intent = "COST_PRESSURE"
        if new_intent != intent:
            logger.debug("Corrected intent %s -> %s text=%s", intent, new_intent, text[:60])
        return new_intent

    return intent


def smooth_intents(results, window=5):
    smoothed = []

    for i in range(len(results)):
        window_slice = results[max(0, i - window): i + window + 1]

        weights = {}
        for j, r in enumerate(window_slice):
            distance = abs(i - (max(0, i - window) + j))
            weight = 1 / (distance + 1)
            weights[r["intent"]] = weights.get(r["intent"], 0) + weight

        dominant = max(weights, key=weights.get)

        smoothed.append({
            **results[i],
            "intent": dominant
        })

    return smoothed


def _load_fallback_model_weights():
    """
    Download fallback model weights from HuggingFace Hub.
    Always loads on CPU for low-memory production safety.
    """
    logger.info(
        "Fallback load start repo=%s file=%s",
        _HF_FALLBACK_REPO,
        _HF_FALLBACK_FILENAME,
    )
    try:
        model_path = hf_hub_download(
            repo_id=_HF_FALLBACK_REPO,
            filename=_HF_FALLBACK_FILENAME,
        )
        load_attempts = (
            {"map_location": "cpu", "weights_only": True, "mmap": True},
            {"map_location": "cpu", "weights_only": True},
            {"map_location": "cpu", "mmap": True},
            {"map_location": "cpu"},
        )

        last_error = None
        for kwargs in load_attempts:
            try:
                return torch.load(model_path, **kwargs)
            except TypeError as exc:
                last_error = exc
                continue
            except Exception as exc:
                last_error = exc
                break

        if last_error is not None:
            logger.error("Fallback weights load failed after retries.", exc_info=True)
        return None
    except Exception as exc:
        logger.error("Failed to download fallback model weights.", exc_info=True)
        return None


class TranscriptAnalyzer:

    def __init__(self):
        self.intent_model = None
        self.fallback_intent_model = None
        self._fallback_device = torch.device("cpu")
        self._fallback_load_attempted = False
        self._fallback_load_failed = False
        self._fallback_load_lock = threading.Lock()
        self._fallback_load_timeout_seconds = _FALLBACK_LOAD_TIMEOUT_SECONDS
        self._last_fallback_used = False
        self._last_embeddings = []

        try:
            self.intent_model = FinBERTIntentModel(device=device)
            logger.info(
                "FinBERT intent model ready with num_labels=%s",
                self.intent_model.model.config.num_labels,
            )
        except Exception as exc:
            logger.error("FinBERT intent model failed to load.", exc_info=True)
            self.intent_model = None

        # Fallback model is lazy-loaded on demand to avoid startup OOM/restart loops.
        logger.info("Fallback pragmatic model lazy-loading enabled.")

        self.conversation_model = load_conversation_attention_model(
            model_path=CONVERSATION_ATTENTION_PATH,
            input_size=771,
            device=device,
        )
        if self.conversation_model is None:
            logger.warning("Conversation attention model not found. Using rule-based fallback.")

    def _ensure_fallback_model_loaded(self) -> bool:
        """Lazily load fallback model exactly once; never raise to caller."""
        if self.fallback_intent_model is not None:
            return True
        if self._fallback_load_failed:
            return False

        with self._fallback_load_lock:
            if self.fallback_intent_model is not None:
                return True
            if self._fallback_load_failed:
                return False
            if self._fallback_load_attempted:
                return False

            self._fallback_load_attempted = True
            logger.info(
                "Attempting lazy load of fallback pragmatic model (timeout=%ss).",
                self._fallback_load_timeout_seconds,
            )

            executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fallback_loader")
            future = executor.submit(_load_fallback_model_weights)
            try:
                state_dict = future.result(timeout=self._fallback_load_timeout_seconds)
            except FuturesTimeoutError:
                self._fallback_load_failed = True
                self.fallback_intent_model = None
                logger.error(
                    "Fallback model loading timed out after %ss; fallback disabled.",
                    self._fallback_load_timeout_seconds,
                )
                return False
            except Exception:
                self._fallback_load_failed = True
                self.fallback_intent_model = None
                logger.error("Fallback model loading failed unexpectedly; fallback disabled.", exc_info=True)
                return False
            finally:
                executor.shutdown(wait=False, cancel_futures=True)

            if state_dict is None:
                self._fallback_load_failed = True
                self.fallback_intent_model = None
                logger.warning("Fallback model weights unavailable; fallback disabled for this process.")
                return False

            try:
                model = FinancialPragmaticTransformer()
                model.load_state_dict(state_dict)
                model.to(self._fallback_device)
                model.eval()
                self.fallback_intent_model = model
                logger.info("Fallback pragmatic model loaded successfully (CPU).")
                return True
            except Exception:
                self._fallback_load_failed = True
                self.fallback_intent_model = None
                logger.error("Fallback model initialization failed; fallback disabled.", exc_info=True)
                return False

    @staticmethod
    def _clean_text(text: str) -> str:
        return re.sub(r"\s+", " ", str(text or "")).strip()

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        cleaned = TranscriptAnalyzer._clean_text(text)
        if not cleaned:
            return []

        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", cleaned)
            if sentence.strip()
        ]

        if len(sentences) <= 1:
            comma_split = [
                part.strip()
                for part in re.split(r"[;:]\s+|,\s+(?=[A-Z])", cleaned)
                if part.strip()
            ]
            if len(comma_split) > 1:
                sentences = comma_split

        if len(sentences) <= 1:
            words = cleaned.split()
            if len(words) > 45:
                sentences = [
                    " ".join(words[index : index + 24])
                    for index in range(0, len(words), 24)
                ]

        return sentences if sentences else [cleaned]

    @staticmethod
    def _chunk_sentences(sentences: list[str]) -> list[str]:
        if not sentences:
            return []

        n_sentences = len(sentences)
        if n_sentences <= 3:
            chunk_size = 1
        elif n_sentences <= 9:
            chunk_size = 2
        else:
            chunk_size = 3

        chunks = []
        for index in range(0, n_sentences, chunk_size):
            chunk = " ".join(sentences[index : index + chunk_size]).strip()
            if chunk:
                chunks.append(chunk)
        return chunks

    def _segment_with_speaker_cues(self, raw_text: str) -> list[dict]:
        text = self._clean_text(raw_text)
        if not text:
            return []

        pattern = re.compile(r"(CEO|CFO|ANALYST|EXECUTIVE|OPERATOR)\s*:\s*", re.IGNORECASE)
        parts = pattern.split(text)
        if len(parts) < 3:
            return []

        segments: list[dict] = []
        preface = parts[0].strip()
        if preface:
            for chunk in self._chunk_sentences(self._split_sentences(preface)):
                segments.append({"speaker": "EXECUTIVE", "text": chunk})

        for idx in range(1, len(parts), 2):
            if idx + 1 >= len(parts):
                break
            speaker = parts[idx].upper()
            block_text = parts[idx + 1].strip()
            if not block_text:
                continue
            block_sentences = self._split_sentences(block_text)
            for chunk in self._chunk_sentences(block_sentences):
                segments.append({"speaker": speaker, "text": chunk})

        return segments

    def _build_segments(self, raw_text: str) -> list[dict]:
        segments = self._segment_with_speaker_cues(raw_text)

        if not segments:
            parsed = parse_transcript(raw_text)
            for parsed_segment in parsed:
                speaker = str(parsed_segment.get("speaker", "EXECUTIVE")).upper()
                text = str(parsed_segment.get("text", "")).strip()
                if not text:
                    continue

                sentences = self._split_sentences(text)
                for chunk in self._chunk_sentences(sentences):
                    segments.append({"speaker": speaker, "text": chunk})

        segments = [
            segment
            for segment in segments
            if segment["text"].strip() and len(segment["text"].strip()) >= 10
        ]

        if len(segments) <= 1:
            text = self._clean_text(raw_text)
            base_sentences = self._split_sentences(text)
            fallback_chunks = self._chunk_sentences(base_sentences)
            if len(fallback_chunks) > 1:
                segments = [{"speaker": "EXECUTIVE", "text": chunk} for chunk in fallback_chunks]
            elif fallback_chunks:
                segments = [{"speaker": "EXECUTIVE", "text": fallback_chunks[0]}]

        return segments

    def predict_intent(self, text, speaker):
        if self.intent_model is not None:
            try:
                output = self.intent_model.predict(text)
                raw_intent = output["intent"]
                confidence = float(output.get("confidence", 0.0))
                intent = _correct_intent(text, raw_intent, confidence)
                cls_embedding = output["embedding"].float()
                final_embedding = torch.cat([cls_embedding, _speaker_vector(speaker)], dim=-1)
                return {
                    "intent": intent,
                    "logits": output["logits"],
                    "embedding": final_embedding,
                    "confidence": confidence,
                    "fallback_used": False,
                }
            except Exception:
                logger.error("FinBERT intent inference failed; trying fallback.", exc_info=True)

        if self.fallback_intent_model is not None or self._ensure_fallback_model_loaded():
            try:
                raw_intent = self.fallback_intent_model.predict(
                    text,
                    speaker=speaker,
                    target_device=self._fallback_device,
                )
                fallback_used = True
            except Exception:
                self._fallback_load_failed = True
                self.fallback_intent_model = None
                logger.error("Fallback intent inference failed; fallback disabled.", exc_info=True)
                raw_intent = "GENERAL_UPDATE"
                fallback_used = False
        else:
            raw_intent = "GENERAL_UPDATE"
            fallback_used = False

        # Fallback path: confidence unknown — apply correction with neutral confidence
        intent = _correct_intent(text, raw_intent, confidence=0.5)
        fallback_embedding = torch.cat(
            [torch.zeros(768, dtype=torch.float32), _speaker_vector(speaker)],
            dim=-1,
        )
        return {
            "intent": intent,
            "logits": torch.zeros(4, dtype=torch.float32),
            "embedding": fallback_embedding,
            "confidence": 0.5,
            "fallback_used": fallback_used,
        }

    def predict_conversation_signal(self, intents):
        if (
            self.conversation_model is not None
            and len(intents) > 0
            and len(self._last_embeddings) == len(intents)
        ):
            sequence = torch.stack(self._last_embeddings).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = self.conversation_model(sequence)
            pred = int(torch.argmax(logits, dim=-1).item())
            return INDEX_TO_SIGNAL[pred]

        score = compute_risk_score(intents)
        return derive_signal(score)

    def analyze(self, raw_text):
        logger.info("Transcript analysis request started.")
        segments = self._build_segments(raw_text)
        logger.debug("Parsed %s segments.", len(segments))
        results = []
        embeddings = []
        fallback_used = False

        for seg in segments:
            prediction = self.predict_intent(seg["text"], seg["speaker"])
            intent = prediction["intent"]
            logger.debug("Model intent=%s speaker=%s", intent, seg["speaker"])
            fallback_used = fallback_used or bool(prediction.get("fallback_used", False))

            results.append({
                "speaker": seg["speaker"],
                "text": seg["text"],
                "intent": intent,
            })
            embeddings.append(prediction["embedding"])

        # results = smooth_intents(results)
        self._last_embeddings = embeddings[: len(results)]
        self._last_fallback_used = fallback_used

        logger.debug("Sample output:")
        for result in results[:5]:
            logger.debug("%s", result)
        intent_distribution = Counter(result["intent"] for result in results)
        logger.debug("Intent distribution=%s", dict(intent_distribution))

        return results

    @property
    def last_fallback_used(self) -> bool:
        return bool(self._last_fallback_used)
