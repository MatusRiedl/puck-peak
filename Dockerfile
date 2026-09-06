FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Background cache warmer. Defaults to off in nhl/cache_warmer.py so local and
# test runs stay quiet; production must opt in explicitly or the first visitor
# after every container restart pays the full cold-fetch cost (~11s on the
# records.nhl.com all-time table) inside their own page load.
ENV PUCKPEAK_CACHE_WARMER_ENABLED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

RUN mkdir -p /app/.cache/nhl_api

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3).read(); sys.exit(0)" || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
