"""Utilities for parsing raw financial transcript text."""

import re
from typing import Dict, List


class TranscriptParser:
    """Parse transcript text into speaker-attributed segments."""

    SPEAKER_PATTERN = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 &.\-']*)\s*:\s*(.*)$")

    def parse_transcript(self, text: str) -> List[Dict[str, str]]:
        """
        Parse raw transcript text into speaker segments.

        Args:
            text: Raw transcript text.

        Returns:
            List of {"speaker": str, "text": str} segments.
        """
        segments: List[Dict[str, str]] = []
        current_speaker = ""
        current_lines: List[str] = []

        def flush_current() -> None:
            nonlocal current_speaker, current_lines
            if not current_lines:
                return
            combined = re.sub(r"\s+", " ", " ".join(current_lines)).strip()
            if combined:
                segments.append({"speaker": current_speaker, "text": combined})
            current_lines = []

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            match = self.SPEAKER_PATTERN.match(line)
            if match:
                flush_current()
                current_speaker = match.group(1).strip()
                first_line = match.group(2).strip()
                if first_line:
                    current_lines.append(first_line)
                continue

            # Continuation line for the current speaker; fallback to Unknown.
            if not current_speaker:
                current_speaker = "Unknown"
            current_lines.append(line)

        flush_current()
        return segments


def load_transcript(file_path: str) -> str:
    """Read and return a transcript text file."""
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


if __name__ == "__main__":
    sample_text = """
    Operator: Good morning everyone.

    CEO: Thank you operator. We had a strong quarter.
    Revenue increased significantly.

    CFO: Margins declined slightly due to supply costs.

    Analyst: Could you elaborate on restructuring?
    """

    parser = TranscriptParser()
    parsed = parser.parse_transcript(sample_text)
    for item in parsed:
        print(item)
