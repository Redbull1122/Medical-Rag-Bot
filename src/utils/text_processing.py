import re
from typing import List


def normalize_text(text: str) -> str:
    """
    Text normalization: lower case, removal of extra spaces.
    """
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _split_sentences(text: str) -> List[str]:
    """
    Lightweight sentence splitter without external deps.
    Splits on ., !, ? followed by whitespace. Keeps short abbreviations together heuristically.
    """
    if not text:
        return []

    # Protect common abbreviations by temporarily replacing periods
    abbreviations = [
        r"e\.g\.", r"i\.e\.", r"mr\.", r"mrs\.", r"dr\.", r"vs\.", r"prof\.", r"inc\.", r"etc\."
    ]
    placeholder = "<DOT>"
    protected = text
    for abbr in abbreviations:
        protected = re.sub(abbr, lambda m: m.group(0).replace('.', placeholder), protected, flags=re.IGNORECASE)

    # Split on sentence boundaries
    parts = re.split(r"(?<=[.!?])\s+", protected)
    sentences = [p.replace(placeholder, ".").strip() for p in parts if p.strip()]
    return sentences


def extract_key_sentences(text: str, max_sentences: int = 3) -> List[str]:
    """
    Extracts up to max_sentences sentences using a lightweight splitter.
    """
    sentences = _split_sentences(text)
    return sentences[:max_sentences]