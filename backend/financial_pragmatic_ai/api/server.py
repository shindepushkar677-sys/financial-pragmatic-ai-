from fastapi import FastAPI
from pydantic import BaseModel

from financial_pragmatic_ai.analysis.earnings_call_analyzer import EarningsCallAnalyzer
app = FastAPI()

analyzer = EarningsCallAnalyzer()

class TranscriptRequest(BaseModel):
    transcript: str


@app.post("/analyze")
def analyze_transcript(request: TranscriptRequest):
    result = analyzer.analyze(request.transcript)
    signal = result["aggregation"]["dominant_signal"]
    insight = result["insight"]

    return {
        "segments": result["segments"],
        "signal": signal,
        "insight": insight,
    }
