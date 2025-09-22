FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel \
 && python3 -m pip install --no-cache-dir -r requirements.txt \
 && python3 - <<'PY'
import importlib, sys
try:
    importlib.import_module('nltk')
    print('nltk import OK')
except Exception as e:
    print('nltk import FAILED:', e)
    sys.exit(1)
PY

COPY . .

# Default port for Streamlit
ENV PORT=8501
ENV PYTHONPATH=/app
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHERUSAGESTATS=false
ENV STREAMLIT_SERVER_ENABLECORS=false

# Pre-download NLTK data needed at runtime (punkt for sent_tokenize, wordnet for synonyms)
ENV NLTK_DATA=/usr/local/share/nltk_data
RUN python3 - <<'PY'
import nltk
packages = ["punkt", "wordnet"]
for p in packages:
    try:
        nltk.download(p, quiet=True)
        print(f"Downloaded NLTK package: {p}")
    except Exception as e:
        print(f"Warning: failed to download {p}: {e}")
PY

EXPOSE 8501

CMD ["sh", "-c", "streamlit run main.py --server.port=$PORT --server.address=0.0.0.0"]