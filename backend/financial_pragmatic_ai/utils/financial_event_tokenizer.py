"""Detect and replace important financial phrases with event tokens."""

import re
from typing import Dict, List, Pattern, Tuple


class FinancialEventTokenizer:
    """Replace known financial events in text with special tokens."""

    DEFAULT_EVENT_MAP: Dict[str, str] = {
        "margin compression": "<MARGIN_COMPRESSION>",
        "share buyback": "<SHARE_BUYBACK>",
        "cost restructuring": "<COST_RESTRUCTURE>",
        "guidance downgrade": "<GUIDANCE_DOWNGRADE>",
        "supply chain disruption": "<SUPPLY_CHAIN_DISRUPTION>",
        "revenue growth": "<REVENUE_GROWTH>",
        "market expansion": "<MARKET_EXPANSION>",
    }

    def __init__(self, event_map: Dict[str, str] | None = None) -> None:
        self.event_map: Dict[str, str] = event_map or self.DEFAULT_EVENT_MAP.copy()
        self._compiled: List[Tuple[Pattern[str], str]] = self._compile_patterns(
            self.event_map
        )

    @staticmethod
    def _compile_patterns(event_map: Dict[str, str]) -> List[Tuple[Pattern[str], str]]:
        # Sort longest phrase first to reduce ambiguity during replacement.
        items = sorted(event_map.items(), key=lambda item: len(item[0]), reverse=True)
        compiled: List[Tuple[Pattern[str], str]] = []
        for phrase, token in items:
            escaped = re.escape(phrase).replace(r"\ ", r"\s+")
            pattern = re.compile(rf"(?<!\w){escaped}(?!\w)", re.IGNORECASE)
            compiled.append((pattern, token))
        return compiled

    def replace_events(self, text: str) -> str:
        """Replace detected financial phrases with event tokens."""
        updated = text
        for pattern, token in self._compiled:
            updated = pattern.sub(token, updated)
        return updated

    def detect_events(self, text: str) -> list[str]:
        """Return detected event tokens in order of first appearance."""
        found: List[Tuple[int, str]] = []
        for pattern, token in self._compiled:
            for match in pattern.finditer(text):
                found.append((match.start(), token))

        found.sort(key=lambda item: item[0])
        tokens: list[str] = []
        seen = set()
        for _, token in found:
            if token not in seen:
                seen.add(token)
                tokens.append(token)
        return tokens


if __name__ == "__main__":
    sample = (
        "The company expects margin compression due to supply chain disruption, "
        "but still sees revenue growth from market expansion."
    )

    tokenizer = FinancialEventTokenizer()
    processed = tokenizer.replace_events(sample)
    detected = tokenizer.detect_events(sample)

    print("Original:", sample)
    print("Processed:", processed)
    print("Detected:", detected)
