"""Detect pragmatic financial signals in executive statements."""


HEDGING_PHRASES = [
    "cautiously optimistic",
    "some uncertainty",
    "potential challenges",
    "monitoring the situation",
    "may experience pressure",
]

CONFIDENCE_PHRASES = [
    "strong performance",
    "robust growth",
    "high confidence",
    "excellent results",
]

RISK_PHRASES = [
    "headwinds",
    "margin pressure",
    "uncertain environment",
    "challenging conditions",
]


class PragmaticAnalyzer:
    """Simple phrase-based pragmatic signal detector."""

    def _detect_phrase(self, text: str, phrase_list: list[str]) -> bool:
        text_lower = text.lower()
        return any(phrase.lower() in text_lower for phrase in phrase_list)

    def analyze(self, text: str) -> dict:
        return {
            "hedging": self._detect_phrase(text, HEDGING_PHRASES),
            "confidence": self._detect_phrase(text, CONFIDENCE_PHRASES),
            "risk_signal": self._detect_phrase(text, RISK_PHRASES),
        }


if __name__ == "__main__":
    analyzer = PragmaticAnalyzer()

    test_sentences = [
        "We remain cautiously optimistic about next quarter.",
        "We expect strong performance this year.",
    ]

    for sentence in test_sentences:
        print(f"Text: {sentence}")
        print(f"Signals: {analyzer.analyze(sentence)}")
