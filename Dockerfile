FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies (THIS IS CRITICAL FOR OCR)
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-nep \
    libtesseract-dev \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean

WORKDIR /app

COPY . /app

# Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 10000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "10000"]
