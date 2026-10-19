# Multi-arch: builds on AWS Graviton (arm64) and x86_64.
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY trailwatch ./trailwatch
RUN useradd -r -u 10001 app
USER app
EXPOSE 8080
HEALTHCHECK CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8080/healthz')"
CMD ["uvicorn", "trailwatch.api:app", "--host", "0.0.0.0", "--port", "8080"]
