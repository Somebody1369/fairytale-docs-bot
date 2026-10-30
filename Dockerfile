FROM python:3.12-slim

# Tesseract reads the scanned PDFs in the archive (Ukrainian), antiword reads the old Word .doc format.
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-ukr antiword \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY bot ./bot

CMD ["python", "-m", "bot"]
