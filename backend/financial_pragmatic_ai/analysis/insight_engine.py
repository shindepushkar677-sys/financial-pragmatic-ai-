"""
insight_engine.py — Driver extraction + compression pipeline.

Pipeline per segment:
  raw text
    → normalise whitespace
    → split sentences
    → pick best sentence (number > keyword > first)
    → strip generic prefix
    → compress to core phrase ([metric] [direction%] OR noun phrase)
    → quality gate (must have financial signal)
    → semantic dedup across bucket
    → cap at limit
"""

import re

# ---------------------------------------------------------------------------
# Keyword sets
# ---------------------------------------------------------------------------

GROWTH_KEYWORDS = frozenset([
    "growth", "revenue", "demand", "expansion", "momentum",
    "record", "improved", "increase", "raised", "guidance",
    "margin", "income", "sales", "profit",
])

RISK_KEYWORDS = frozenset([
    "pressure", "decline", "risk", "loss", "drop",
    "uncertainty", "headwind", "compression", "cost",
    "inflation", "slowdown", "challenging", "variability",
    "softness", "weakness",
])

# Words that carry no meaning for dedup comparison
_STOPWORDS = frozenset([
    "a", "an", "the", "our", "we", "us", "i", "is", "was", "were", "be",
    "been", "are", "have", "has", "had", "this", "that", "these", "those",
    "with", "at", "to", "in", "of", "by", "on", "for", "and", "or", "but",
    "as", "it", "its", "from", "which", "more", "also", "very", "strong",
    "significant", "continued", "continue", "remains", "remain",
])

# Direction words → +  or  –
_UP_VERBS = re.compile(
    r"\b(grew|grown|increased|rose|improved|expanded|surged|accelerated|"
    r"raised|higher|gain|up|outperform)\b",
    re.IGNORECASE,
)
_DOWN_VERBS = re.compile(
    r"\b(declined|fell|dropped|decreased|compressed|contracted|lower|"
    r"down|loss|pressure|headwind|softness|weakness)\b",
    re.IGNORECASE,
)

_PCT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")

# Generic sentence-opening prefixes to strip
_PREFIX_RE = re.compile(
    r"^(the company (said|noted|highlighted|reported|stated|saw)|"
    r"management (noted|stated|said|highlighted|indicated) that|"
    r"we (believe|think|expect|continue to|are (seeing|experiencing))|"
    r"i (think|believe)|"
    r"this (growth|performance|result|was)|"
    r"as (we|i) (mentioned|noted|said)|"
    r"looking ahead[,]?|going forward[,]?)\s*",
    re.IGNORECASE,
)

# Words to drop from the compressed phrase
_FILLER_RE = re.compile(
    r"\b(very|quite|really|truly|largely|primarily|general|overall|"
    r"strong|significant|continued|approximately|roughly|slightly)\b\s*",
    re.IGNORECASE,
)

# Minimum financial signal to pass quality gate
_SIGNAL_RE = re.compile(
    r"\b(\d+(\.\d+)?%?|\bpercent\b|growth|revenue|cost|margin|income|"
    r"sales|profit|decline|increase|decrease|pressure|headwind|"
    r"expansion|raised|guidance|record|loss)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def _normalise(raw: str) -> str:
    return " ".join(raw.split())


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.;!])\s+", text)
    return [p.strip().rstrip(".,;:") for p in parts if len(p.strip()) > 8]


def _strip_prefix(text: str) -> str:
    return _PREFIX_RE.sub("", text).strip()


def _best_sentence(sentences: list[str]) -> str | None:
    """Prefer sentence with a % number, then any signal keyword, then first."""
    for s in sentences:
        if _PCT_RE.search(s):
            return s
    for s in sentences:
        if _SIGNAL_RE.search(s):
            return s
    return sentences[0] if sentences else None


# ---------------------------------------------------------------------------
# Core compression  →  [metric] [±value%]  or  noun phrase
# ---------------------------------------------------------------------------

def _compress(text: str) -> str | None:
    """
    Try to produce a compact phrase:
      • If a % exists  →  "[2-3 metric words] [+/-][N%]"
      • Otherwise      →  "[2-4 word noun phrase around keyword]"
    """
    # --- Path 1: has a percentage ---
    pct_match = _PCT_RE.search(text)
    if pct_match:
        pct_val = pct_match.group(1)

        # Direction
        if _DOWN_VERBS.search(text):
            sign = "-"
        elif _UP_VERBS.search(text):
            sign = "+"
        else:
            sign = ""

        # Metric = up to 3 meaningful words immediately before the number
        before = text[: pct_match.start()].strip()
        words = [w for w in before.split() if w.lower() not in _STOPWORDS]
        metric_words = words[-3:] if words else []

        # Strip filler from metric words too
        metric = " ".join(metric_words)
        metric = _FILLER_RE.sub("", metric).strip()

        if metric:
            phrase = f"{metric.lower()} {sign}{pct_val}%".strip()
            return phrase

    # --- Path 2: no percentage — extract noun phrase ---
    # Find the first financial keyword and take ±2 words around it
    all_kw = GROWTH_KEYWORDS | RISK_KEYWORDS
    words = text.split()
    for idx, w in enumerate(words):
        if w.lower().rstrip(".,;") in all_kw:
            start = max(0, idx - 1)
            end = min(len(words), idx + 3)
            phrase = " ".join(words[start:end]).rstrip(".,;:")
            phrase = _FILLER_RE.sub("", phrase).strip()
            if phrase:
                return phrase.lower()

    return None


# ---------------------------------------------------------------------------
# Semantic dedup — remove drivers that share ≥ 2 significant words
# ---------------------------------------------------------------------------

def _sig_words(text: str) -> frozenset[str]:
    return frozenset(
        w.lower().rstrip(".,;:%+")
        for w in text.split()
        if w.lower().rstrip(".,;:%+") not in _STOPWORDS and len(w) > 2
    )


def _semantic_dedup(drivers: list[str]) -> list[str]:
    kept: list[str] = []
    kept_sig: list[frozenset[str]] = []

    for d in drivers:
        sig = _sig_words(d)
        # Check overlap with every already-kept driver
        if any(len(sig & ks) >= 2 for ks in kept_sig):
            continue
        kept.append(d)
        kept_sig.append(sig)

    return kept


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_key_drivers(results, limit: int = 4) -> dict[str, list[str]]:
    growth_raw: list[str] = []
    risk_raw: list[str] = []

    for r in results:
        raw = r.get("text", "")
        intent = r.get("intent", "")
        word_count = len(raw.split())
        text_lower = raw.lower()

        if word_count < 5:
            continue

        # Full pipeline
        normalised = _normalise(raw)
        sentences = _split_sentences(normalised)
        best = _best_sentence(sentences)
        if not best:
            continue

        best = _strip_prefix(best)
        phrase = _compress(best)
        if not phrase:
            continue

        # Quality gate
        if not _SIGNAL_RE.search(phrase):
            continue

        if intent == "EXPANSION" and any(kw in text_lower for kw in GROWTH_KEYWORDS):
            growth_raw.append(phrase)
        elif intent == "COST_PRESSURE" and any(kw in text_lower for kw in RISK_KEYWORDS):
            risk_raw.append(phrase)

    growth = _semantic_dedup(growth_raw)[:limit]
    risk = _semantic_dedup(risk_raw)[:limit]

    print("CLEAN GROWTH DRIVERS:", growth)
    print("CLEAN RISK DRIVERS:", risk)

    return {
        "growth_drivers": growth,
        "risk_drivers": risk,
    }
