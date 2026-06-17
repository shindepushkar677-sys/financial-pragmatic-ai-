"""Extract structured financial signals from transcript segments."""

from typing import Dict, List

import torch

from financial_pragmatic_ai.models.finbert_base import load_finbert
from financial_pragmatic_ai.utils.device import get_torch_device
from financial_pragmatic_ai.utils.financial_event_tokenizer import (
    FinancialEventTokenizer,
)
from financial_pragmatic_ai.utils.transcript_parser import TranscriptParser


class FinancialSignalExtractor:
    """Run event tokenization and FinBERT embedding extraction on segments."""

    def __init__(self) -> None:
        self.device = get_torch_device()
        self.tokenizer, self.model = load_finbert(device=self.device.type)
        self.event_tokenizer = FinancialEventTokenizer()

    def _extract_cls_embedding(self, text: str) -> torch.Tensor:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # CLS token embedding for the first (and only) sequence in the batch.
        return outputs.last_hidden_state[:, 0, :].squeeze(0)

    def extract_signals(self, segments: list[dict]) -> list[dict]:
        """
        Extract structured financial signals from parsed transcript segments.

        Args:
            segments: List like [{"speaker": "...", "text": "..."}, ...]

        Returns:
            List of dictionaries with speaker, transformed text, detected events,
            and embedding shape.
        """
        results: List[Dict[str, object]] = []

        for segment in segments:
            speaker = str(segment.get("speaker", "Unknown")).strip() or "Unknown"
            raw_text = str(segment.get("text", "")).strip()
            if not raw_text:
                continue

            processed_text = self.event_tokenizer.replace_events(raw_text)
            events = self.event_tokenizer.detect_events(raw_text)
            cls_embedding = self._extract_cls_embedding(processed_text)

            results.append(
                {
                    "speaker": speaker,
                    "text": processed_text,
                    "events": events,
                    "embedding_shape": tuple(cls_embedding.shape),
                }
            )

        return results


if __name__ == "__main__":
    sample_transcript = """
    CEO: We expect margin compression next quarter.
    CFO: Revenue growth remains strong.
    """

    parser = TranscriptParser()
    parsed_segments = parser.parse_transcript(sample_transcript)

    extractor = FinancialSignalExtractor()
    signals = extractor.extract_signals(parsed_segments)

    for signal in signals:
        print(signal)
