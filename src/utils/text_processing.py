import re
from typing import List
import nltk
import ssl

# SSL context for NLTK downloads
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download NLTK data
try:
    nltk.download("wordnet", quiet=True)
except Exception as e:
    print(f"Warning: Could not download wordnet: {e}")

from nltk.tokenize import sent_tokenize


def normalize_text(text: str) -> str:
    """
    Text normalization: lower case, removal of extra spaces.
    """
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_key_sentences(text: str, max_sentences: int = 3) -> List[str]:
    """
    Extracts key sentences from the text.
    """
    try:
        sentences = sent_tokenize(text)
        return sentences[:max_sentences]
    except Exception as e:
        print(f"Warning: Could not tokenize sentences: {e}")
        # Fallback: просто розділяємо по крапках
        return text.split('.')[:max_sentences]