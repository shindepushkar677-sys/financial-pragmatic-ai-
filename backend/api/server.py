import os
import logging
os.environ.setdefault("HF_HOME", "/tmp/hf_cache")

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from api.schemas import CompareRequest, TranscriptRequest
from financial_pragmatic_ai.analysis.earnings_call_analyzer import EarningsCallAnalyzer
from financial_pragmatic_ai.analysis.financial_signal_engine import (
    compute_confidence,
    compute_intent_distribution,
    compute_risk_score,
    compute_signal_distribution,
    compute_signal_std,
    detect_volatility,
    derive_signal,
    generate_insight,
)
from financial_pragmatic_ai.analysis.insight_engine import extract_key_drivers
from financial_pragmatic_ai.analysis.market_predictor import predict_market_outlook

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


app = FastAPI(title="Financial Pragmatic AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

analyzer = None


def _run_analysis(transcript: str):
    global analyzer
    if analyzer is None:
        analyzer = EarningsCallAnalyzer()

    result = analyzer.analyze(transcript)
    segments = result["segments"]
    fallback_used = bool(result.get("fallback_used", False))
    if len(segments) == 0:
        return {"error": "Could not parse transcript"}

    # --- Raw driver extraction removed ---
    # Path B (insight_engine.extract_key_drivers) is the single source of truth.

    score = compute_risk_score(segments)
    signal = derive_signal(score)
    confidence = compute_confidence(segments)
    volatility = detect_volatility(segments)
    volatility_std = round(compute_signal_std(segments), 4)
    intent_distribution = compute_intent_distribution(segments)
    signal_distribution = compute_signal_distribution(segments)
    logger.info("intent_distribution=%s", intent_distribution)
    logger.info("signal_distribution=%s", signal_distribution)
    market = predict_market_outlook(
        signal=signal,
        risk_score=score,
        volatility=volatility,
        intent_distribution=intent_distribution,
    )
    insight = generate_insight(score, segments)
    drivers = extract_key_drivers(segments)

    logger.info("growth_drivers=%s", drivers["growth_drivers"])
    logger.info("risk_drivers=%s", drivers["risk_drivers"])

    _INTENT_VALUE = {
        "EXPANSION": 1,
        "COST_PRESSURE": -1,
        "GENERAL_UPDATE": 0,
        "STRATEGIC_PROBING": 0,
    }
    timeline = [
        {
            "step": i,
            "value": _INTENT_VALUE.get(seg["intent"], 0),
            "intent": seg["intent"],
            "label": seg["text"][:60],
        }
        for i, seg in enumerate(segments)
    ]
    logger.info("timeline_length=%s fallback_used=%s", len(timeline), fallback_used)

    return {
        "score": score,
        "signal": signal,
        "prediction": market["prediction"],
        "prediction_explanation": market["explanation"],
        "confidence": confidence,
        "volatility": volatility,
        "volatility_std": volatility_std,
        "intent_distribution": intent_distribution,
        "insight": insight,
        "segments": segments,
        "growth_drivers": drivers["growth_drivers"],
        "risk_drivers": drivers["risk_drivers"],
        "drivers": drivers,
        "timeline": timeline,
        "fallback_used": fallback_used,
    }


@app.post("/analyze")
def analyze_transcript(request: TranscriptRequest):
    return _run_analysis(request.transcript)


@app.post("/upload")
async def upload_transcript(
    file: UploadFile = File(...),
):
    content = await file.read()
    filename = file.filename.lower()

    if filename.endswith(".pdf"):
        import io
        import pdfplumber

        try:
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                pages = []
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        pages.append(text)

            text = "\n".join(pages)
        except Exception as exc:
            return {"error": f"PDF parsing failed: {str(exc)}"}
    elif filename.endswith(".txt"):
        try:
            text = content.decode("utf-8")
        except Exception:
            return {"error": "TXT file must be UTF-8 encoded"}
    else:
        return {"error": "Only .txt and .pdf files are supported"}

    text = text.replace("\n\n", "\n")
    text = text.strip()
    return _run_analysis(text)


@app.post("/compare")
async def compare_transcripts(request: CompareRequest):
    if not request.transcript_1 or not request.transcript_2:
        raise HTTPException(
            status_code=400,
            detail="Provide transcript_1 and transcript_2",
        )

    first = _run_analysis(request.transcript_1)
    if "error" in first:
        return first
    second = _run_analysis(request.transcript_2)
    if "error" in second:
        return second

    risk_delta = round(float(second["score"]) - float(first["score"]), 2)
    confidence_delta = round(float(second["confidence"]) - float(first["confidence"]), 2)
    signal_changed = first["signal"] != second["signal"]

    if risk_delta > 0:
        comparison_text = f"Risk increased by {abs(risk_delta):.2f}% compared to previous call."
        trend = "UP"
    elif risk_delta < 0:
        comparison_text = f"Risk decreased by {abs(risk_delta):.2f}% compared to previous call."
        trend = "DOWN"
    else:
        comparison_text = "Risk is unchanged compared to previous call."
        trend = "FLAT"

    return {
        "transcript_1": first,
        "transcript_2": second,
        "signal_difference": {
            "from": first["signal"],
            "to": second["signal"],
            "changed": signal_changed,
        },
        "risk_delta": risk_delta,
        "confidence_delta": confidence_delta,
        "trend": trend,
        "comparison": comparison_text,
    }
