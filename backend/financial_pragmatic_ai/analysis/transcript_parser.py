import re


COMMON_EXECUTIVE_KEYWORDS = [
    "revenue", "growth", "strategy", "market", "expansion", "expand",
    "performance", "guidance"
]

CFO_KEYWORDS = [
    "margin", "cost", "expense", "ebitda", "operating income",
    "cash flow", "balance sheet"
]

ANALYST_KEYWORDS = [
    "question", "how", "what", "why", "could you", "can you"
]

SPEAKER_CUE_PATTERN = re.compile(r"(?=(?:Analyst|Executive|CFO|CEO|Operator)\s*:)", re.IGNORECASE)
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")


def clean_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_speaker_blocks(text: str):
    """
    Extract blocks using NAME: pattern
    Works for:
    Doug McMillon:
    John Smith:
    Operator:
    """
    pattern = (
        r"([A-Z][a-z]+(?:\s[A-Z][a-z]+)*|[A-Z]{2,}(?:\s[A-Z]{2,})*):\s*(.+?)"
        r"(?=(?:[A-Z][a-z]+(?:\s[A-Z][a-z]+)*|[A-Z]{2,}(?:\s[A-Z]{2,})*):|$)"
    )

    matches = re.findall(pattern, text, re.DOTALL)

    blocks = []

    for name, content in matches:
        content = content.strip()

        if len(content) < 40:
            continue

        blocks.append({
            "name": name.strip(),
            "text": content
        })

    return blocks


def infer_role(name: str, text: str):
    text_lower = text.lower()
    name_lower = name.lower()

    if "operator" in name_lower:
        return "OPERATOR"

    if "analyst" in name_lower:
        return "ANALYST"

    # Analyst detection (questions dominate Q&A)
    if any(q in text_lower for q in ANALYST_KEYWORDS):
        return "ANALYST"

    # CFO detection (financial-heavy language)
    if any(k in text_lower for k in CFO_KEYWORDS):
        return "CFO"

    # CEO / Exec detection (strategy-heavy language)
    if any(k in text_lower for k in COMMON_EXECUTIVE_KEYWORDS):
        return "CEO"

    return "EXECUTIVE"


def _chunk_by_sentences(text: str, chunk_size: int = 2):
    sentences = [sentence.strip() for sentence in SENTENCE_SPLIT_PATTERN.split(text) if sentence.strip()]
    if not sentences:
        return []

    chunks = []
    for i in range(0, len(sentences), chunk_size):
        chunk = " ".join(sentences[i : i + chunk_size]).strip()
        if len(chunk) >= 30:
            chunks.append(chunk)
    return chunks


def _split_block(name: str, text: str):
    parts = [part.strip() for part in SPEAKER_CUE_PATTERN.split(text) if part.strip()]
    results = []

    if len(parts) <= 1:
        chunks = _chunk_by_sentences(text, chunk_size=2)
        if len(chunks) > 1:
            return [{"name": name, "text": chunk} for chunk in chunks]
        return [{"name": name, "text": text.strip()}]

    for part in parts:
        matched = re.match(r"^(Analyst|Executive|CFO|CEO|Operator)\s*:\s*(.*)$", part, re.IGNORECASE | re.DOTALL)
        if matched:
            local_name = matched.group(1).upper()
            content = matched.group(2).strip()
        else:
            local_name = name
            content = part

        if not content:
            continue

        chunks = _chunk_by_sentences(content, chunk_size=2)
        if len(chunks) > 1:
            for chunk in chunks:
                results.append({"name": local_name, "text": chunk})
        else:
            if len(content) >= 30:
                results.append({"name": local_name, "text": content})

    return results


def fallback_chunking(text: str):
    """
    If no speaker blocks found -> split into semantic chunks
    """
    sentences = re.split(r"[.!?]", text)

    chunks = []
    buffer = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        buffer += sentence + ". "

        if len(buffer) > 120:
            chunks.append({
                "name": "UNKNOWN",
                "text": buffer.strip()
            })
            buffer = ""

    if len(buffer.strip()) > 30:
        chunks.append({
            "name": "UNKNOWN",
            "text": buffer.strip()
        })

    return chunks


def parse_transcript(text: str):
    text = clean_text(text)

    blocks = extract_speaker_blocks(text)

    if len(blocks) == 0:
        blocks = fallback_chunking(text)
    else:
        expanded_blocks = []
        for block in blocks:
            expanded_blocks.extend(_split_block(block["name"], block["text"]))
        if expanded_blocks:
            blocks = expanded_blocks

    if len(blocks) == 1:
        block = blocks[0]
        fallback_chunks = _chunk_by_sentences(block["text"], chunk_size=2)
        if len(fallback_chunks) > 1:
            blocks = [{"name": block["name"], "text": chunk} for chunk in fallback_chunks]

    segments = []

    for block in blocks:
        role = infer_role(block["name"], block["text"])

        segments.append({
            "speaker": role,
            "name": block["name"],
            "text": block["text"]
        })

    print(f"[DEBUG] Parsed {len(segments)} segments")

    return segments
