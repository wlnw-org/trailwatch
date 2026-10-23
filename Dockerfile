# Multi-arch: builds on AWS Graviton (arm64) and x86_64.
# WITH_MODELS=1 (default) bakes in SpeciesNet and MegaDetector; WITH_MODELS=0 gives a small image with the test stub.
FROM python:3.12-slim
ARG WITH_MODELS=1
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HOME=/home/app
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* && useradd -r -m -d /home/app -u 10001 app
COPY requirements.txt requirements-speciesnet.txt ./
RUN if [ "$WITH_MODELS" = "1" ]; then \
      pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-speciesnet.txt; \
    else pip install --no-cache-dir -r requirements.txt; fi
USER app
# Download the model weights at build time so the first request is fast.
RUN if [ "$WITH_MODELS" = "1" ]; then python -c "import speciesnet as s; s.SpeciesNetClassifier(s.DEFAULT_MODEL); s.SpeciesNetDetector(s.DEFAULT_MODEL)"; fi
COPY --chown=app trailwatch ./trailwatch
COPY --chown=app samples ./samples
ENV TRAILWATCH_CLASSIFIER=speciesnet
EXPOSE 8080
HEALTHCHECK --start-period=120s CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8080/healthz')"
CMD ["uvicorn", "trailwatch.api:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
