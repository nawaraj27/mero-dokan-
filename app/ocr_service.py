from __future__ import annotations

import io
import logging
import os
import sys
import shutil
import subprocess
from pathlib import Path
from typing import Iterable

import cv2
import fitz
import numpy as np
import pytesseract
from PIL import Image, ImageOps
from pytesseract import Output


logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"
}
SUPPORTED_EXTENSIONS = SUPPORTED_IMAGE_EXTENSIONS | {".pdf"}


class OCRProcessingError(Exception):
    pass


class UnsupportedFileTypeError(Exception):
    pass


class OCRProcessor:
    def __init__(self) -> None:
        self.is_available = False
        self.ocr_languages = os.getenv("OCR_LANGUAGES", "eng+nep")

        # =========================================================
        # SAFE TESSERACT SETUP (RENDER + LOCAL SAFE)
        # =========================================================

        # 1. ENV override
        tesseract_path = os.getenv("TESSERACT_CMD")

        # 2. system path
        if not tesseract_path:
            tesseract_path = shutil.which("tesseract")

        # 3. Render default fallback
        if not tesseract_path:
            tesseract_path = "/usr/bin/tesseract"

        # 4. Windows fallback
        if sys.platform.startswith("win"):
            tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

        # verify binary exists
        if not tesseract_path or not Path(tesseract_path).exists():
            logger.warning("Tesseract binary not found at: %s", tesseract_path)
            return

        # verify binary works
        try:
            subprocess.run(
                [tesseract_path, "--version"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except Exception as exc:
            logger.warning("Tesseract binary not working: %s", exc)
            return

        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        logger.info("Tesseract working at: %s", tesseract_path)

        # verify pytesseract connection
        try:
            pytesseract.get_tesseract_version()
            self.is_available = True
        except Exception as exc:
            logger.warning("Tesseract not available: %s", exc)
            return

        # language validation
        try:
            self._validate_languages()
        except Exception as exc:
            logger.warning("Language validation failed: %s", exc)

        logger.info("OCR ready with languages: %s", self.ocr_languages)

    # =========================================================
    # PUBLIC API
    # =========================================================
    def extract_text(self, filename: str, file_bytes: bytes) -> tuple[str, float, str]:
        if not self.is_available:
            raise OCRProcessingError("Tesseract not available")

        extension = Path(filename).suffix.lower()

        if extension not in SUPPORTED_EXTENSIONS:
            raise UnsupportedFileTypeError(f"Unsupported file type: {extension}")

        if extension == ".pdf":
            pages = self._pdf_to_images(file_bytes)
            results = [self._ocr_page(img) for img in pages]

            texts = [t for t, _ in results if t]
            confidences = [c for _, c in results if c > 0]

            return (
                "\n\n".join(texts).strip(),
                self._average(confidences),
                "pdf",
            )

        image = self._load_image(file_bytes)
        text, confidence = self._ocr_page(image)
        return text, confidence, "image"

    # =========================================================
    # IMAGE HANDLING
    # =========================================================
    def _load_image(self, file_bytes: bytes) -> Image.Image:
        try:
            image = Image.open(io.BytesIO(file_bytes))
            image = ImageOps.exif_transpose(image)
            return image.convert("RGB")
        except Exception as exc:
            raise OCRProcessingError("Invalid image file") from exc

    def _pdf_to_images(self, file_bytes: bytes) -> Iterable[Image.Image]:
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
        except Exception as exc:
            raise OCRProcessingError("Invalid PDF file") from exc

        if doc.page_count == 0:
            raise OCRProcessingError("PDF has no pages")

        zoom = fitz.Matrix(2.0, 2.0)
        images: list[Image.Image] = []

        for page in doc:
            pix = page.get_pixmap(matrix=zoom, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)

        doc.close()
        return images

    # =========================================================
    # OCR CORE
    # =========================================================
    def _ocr_page(self, image: Image.Image) -> tuple[str, float]:
        processed = self._preprocess(image)
        config = "--oem 1 --psm 6"

        try:
            text = pytesseract.image_to_string(
                processed,
                config=config,
                lang=self.ocr_languages
            )

            details = pytesseract.image_to_data(
                processed,
                config=config,
                lang=self.ocr_languages,
                output_type=Output.DICT,
            )
        except Exception as exc:
            raise OCRProcessingError(f"OCR failed: {exc}") from exc

        clean_text = self._clean_text(text)
        confidence = self._extract_confidence(details)

        return clean_text, confidence

    # =========================================================
    # LANGUAGE VALIDATION
    # =========================================================
    def _validate_languages(self) -> None:
        requested = [l.strip() for l in self.ocr_languages.split("+") if l.strip()]

        try:
            available = set(pytesseract.get_languages(config=""))
        except Exception:
            available = {"eng"}

        missing = [l for l in requested if l not in available]

        if missing:
            logger.warning("Missing language packs: %s", missing)

    # =========================================================
    # IMAGE PREPROCESSING
    # =========================================================
    def _preprocess(self, image: Image.Image) -> np.ndarray:
        rgb = np.array(image.convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        h, w = bgr.shape[:2]
        longest = max(h, w)

        if longest < 1800:
            scale = min(3.0, 1800 / max(longest, 1))
            bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        denoise = cv2.fastNlMeansDenoising(gray, None, 15, 7, 21)
        contrast = cv2.createCLAHE(2.0, (8, 8)).apply(denoise)
        _, thresh = cv2.threshold(
            contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        return thresh

    # =========================================================
    # TEXT CLEANING
    # =========================================================
    def _clean_text(self, text: str) -> str:
        return "\n".join(line.rstrip() for line in text.splitlines()).strip()

    # =========================================================
    # CONFIDENCE
    # =========================================================
    def _extract_confidence(self, details: dict) -> float:
        values = []

        for v in details.get("conf", []):
            try:
                f = float(v)
                # pytesseract uses -1 for invalid confidence
                if f > 0:
                    values.append(f)
            except:
                continue

        return self._average(values)

    def _average(self, values: list[float]) -> float:
        if not values:
            return 0.0
        return round(sum(values) / len(values), 2)
