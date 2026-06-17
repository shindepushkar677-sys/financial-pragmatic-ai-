"""Decision engine that combines event, intent, and pragmatic signals."""

from pathlib import Path
from typing import Dict, List

import torch

from financial_pragmatic_ai.inference.signal_extractor import FinancialSignalExtractor
from financial_pragmatic_ai.models.intent_classifier import ExecutiveIntentClassifier
from financial_pragmatic_ai.utils.pragmatic_analyzer import PragmaticAnalyzer
from financial_pragmatic_ai.utils.transcript_parser import TranscriptParser


class FinancialDecisionEngine:
    """Combine extracted signals to produce structured transcript analysis."""

    def __init__(self) -> None:
        self.parser = TranscriptParser()
        self.signal_extractor = FinancialSignalExtractor()
        self.pragmatic_analyzer = PragmaticAnalyzer()

        self.device = self.signal_extractor.device
        self.intent_classifier = ExecutiveIntentClassifier().to(self.device)

        project_root = Path(__file__).resolve().parents[2]
        model_path = (
            project_root
            / "financial_pragmatic_ai"
            / "models"
            / "intent_classifier.pt"
        )
        if not model_path.exists():
            raise FileNotFoundError(f"Trained classifier not found at: {model_path}")

        state_dict = torch.load(model_path, map_location=self.device)
        self.intent_classifier.load_state_dict(state_dict)
        self.intent_classifier.eval()

    @staticmethod
    def _compute_risk_score(
        events: List[str], hedging: bool, risk_signal: bool
    ) -> float:
        score = 0.0
        if risk_signal:
            score += 0.4
        if hedging:
            score += 0.2
        if any("MARGIN_COMPRESSION" in event for event in events):
            score += 0.3
        return max(0.0, min(1.0, score))

    def analyze_transcript(self, text: str) -> list[dict]:
        """
        Parse transcript text and produce segment-level financial analysis.

        Returns:
            List of {
                "speaker", "text", "intent", "events",
                "hedging", "confidence", "risk_signal", "risk_score"
            }
        """
        parsed_segments = self.parser.parse_transcript(text)
        extracted = self.signal_extractor.extract_signals(parsed_segments)
        results: List[Dict[str, object]] = []

        for segment_signal in extracted:
            segment_text = str(segment_signal["text"])
            events = list(segment_signal["events"])

            pragmatic = self.pragmatic_analyzer.analyze(segment_text)
            cls_embedding = self.signal_extractor._extract_cls_embedding(segment_text)
            cls_embedding = cls_embedding.unsqueeze(0)
            intent = self.intent_classifier.predict(cls_embedding)

            risk_score = self._compute_risk_score(
                events=events,
                hedging=bool(pragmatic["hedging"]),
                risk_signal=bool(pragmatic["risk_signal"]),
            )

            results.append(
                {
                    "speaker": str(segment_signal["speaker"]),
                    "text": segment_text,
                    "intent": intent,
                    "events": events,
                    "hedging": bool(pragmatic["hedging"]),
                    "confidence": bool(pragmatic["confidence"]),
                    "risk_signal": bool(pragmatic["risk_signal"]),
                    "risk_score": risk_score,
                }
            )

        return results


if __name__ == "__main__":
    sample_transcript = """
    CEO: We remain cautiously optimistic but expect margin compression next quarter.
    CFO: Revenue growth remains strong.
    """

    engine = FinancialDecisionEngine()
    analysis = engine.analyze_transcript(sample_transcript)

    for item in analysis:
        print(item)
