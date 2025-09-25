FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel \
 && python3 -m pip install --no-cache-dir -r requirements.txt

COPY . .

# Default port for Streamlit
ENV PORT=8501
ENV PYTHONPATH=/app
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHERUSAGESTATS=false
ENV STREAMLIT_SERVER_ENABLECORS=false
ENV DNNL_MAX_CPU_ISA=GENERIC



EXPOSE 8501

CMD ["sh", "-c", "streamlit run main.py --server.port=$PORT --server.address=0.0.0.0"]