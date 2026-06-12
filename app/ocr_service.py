from __future__ import annotations

import io
import logging
import os
import sys
from pathlib import Path
from typing import Iterable

import cv2
import fitz
import numpy as np
import pytesseract
from PIL import Image, ImageOps
from pytesseract import Output


logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
SUPPORTED_EXTENSIONS = SUPPORTED_IMAGE_EXTENSIONS | {".pdf"}


class OCRProcessingError(Exception):
    pass


class UnsupportedFileTypeError(Exception):
    pass


class OCRProcessor:
    def __init__(self) -> None:
        self.is_available = False
        self.ocr_languages = "eng"
        
        # Set Tesseract path based on platform
        if sys.platform.startswith('win'):
            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        else:
            # Standard fallback path for Linux cloud environments
            pytesseract.pytesseract.tesseract_cmd = 'tesseract'

        try:
            pytesseract.get_tesseract_version()
            self.is_available = True
            self.ocr_languages = os.getenv("OCR_LANGUAGES", "eng+nep").strip() or "eng"
            self._validate_languages()
            logger.info("OCR processor initialized successfully with languages: %s", self.ocr_languages)
        except (pytesseract.TesseractNotFoundError, OCRProcessingError) as exc:
            logger.warning(
                "Tesseract OCR is not available: %s. OCR functionality will be disabled. "
                "Install Tesseract or set TESSERACT_CMD environment variable to enable it.",
                str(exc),
            )

    def extract_text(self, filename: str, file_bytes: bytes) -> tuple[str, float, str]:
        if not self.is_available:
            raise OCRProcessingError("OCR processor is not available. Tesseract is not installed on this system.")
        
        extension = Path(filename).suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise UnsupportedFileTypeError(f"Unsupported file type. Supported formats: {supported}")

        if extension == ".pdf":
            page_results = [self._ocr_page(image) for image in self._pdf_to_images(file_bytes)]
            texts = [text for text, _ in page_results if text]
            confidences = [confidence for _, confidence in page_results if confidence > 0]
            return "\n\n".join(texts).strip(), self._average(confidences), "pdf"

        image = self._load_image(file_bytes)
        text, confidence = self._ocr_page(image)
        return text, confidence, "image"

    def _load_image(self, file_bytes: bytes) -> Image.Image:
        try:
            image = Image.open(io.BytesIO(file_bytes))
            image = ImageOps.exif_transpose(image)
            return image.convert("RGB")
        except Exception as exc:
            raise OCRProcessingError("The uploaded image could not be opened.") from exc

    def _pdf_to_images(self, file_bytes: bytes) -> Iterable[Image.Image]:
        try:
            document = fitz.open(stream=file_bytes, filetype="pdf")
        except Exception as exc:
            raise OCRProcessingError("The uploaded PDF could not be opened.") from exc

        if document.page_count == 0:
            raise OCRProcessingError("The uploaded PDF does not contain any pages.")

        images: list[Image.Image] = []
        zoom_matrix = fitz.Matrix(2.0, 2.0)

        for page in document:
            pixmap = page.get_pixmap(matrix=zoom_matrix, alpha=False)
            image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
            images.append(image)

        document.close()
        return images

    def _ocr_page(self, image: Image.Image) -> tuple[str, float]:
        processed = self._preprocess_image(image)
        config = "--oem 1 --psm 6"

        try:
            raw_text = pytesseract.image_to_string(processed, config=config, lang=self.ocr_languages)
            details = pytesseract.image_to_data(
                processed,
                config=config,
                lang=self.ocr_languages,
                output_type=Output.DICT,
            )
        except pytesseract.TesseractError as exc:
            raise OCRProcessingError(f"OCR processing failed: {exc}") from exc

        text = self._clean_text(raw_text)
        confidence = self._extract_confidence(details)
        return text, confidence

    def _validate_languages(self) -> None:
        requested_languages = [lang.strip() for lang in self.ocr_languages.split("+") if lang.strip()]
        if not requested_languages:
            raise OCRProcessingError("No OCR languages are configured.")

        try:
            available_languages = set(pytesseract.get_languages(config=""))
        except pytesseract.TesseractError as exc:
            raise OCRProcessingError(f"Unable to read installed Tesseract languages: {exc}") from exc

        missing_languages = [lang for lang in requested_languages if lang not in available_languages]
        if missing_languages:
            missing = ", ".join(missing_languages)
            raise OCRProcessingError(
                "Missing Tesseract language data: "
                f"{missing}. Install the required traineddata files or change OCR_LANGUAGES."
            )

    def _preprocess_image(self, image: Image.Image) -> np.ndarray:
        rgb_image = np.array(image.convert("RGB"))
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        height, width = bgr_image.shape[:2]

        longest_edge = max(height, width)
        if longest_edge < 1800:
            scale = min(3.0, 1800 / max(longest_edge, 1))
            bgr_image = cv2.resize(bgr_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        grayscale = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(grayscale, None, 15, 7, 21)
        contrasted = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(denoised)
        _, thresholded = cv2.threshold(contrasted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return thresholded

    def _clean_text(self, text: str) -> str:
        lines = [line.rstrip() for line in text.splitlines()]
        return "\n".join(lines).strip()

    def _extract_confidence(self, details: dict) -> float:
        confidences: list[float] = []
        for item in details.get("conf", []):
            try:
                value = float(item)
            except (TypeError, ValueError):
                continue
            if value >= 0:
                confidences.append(value)
        return self._average(confidences)

    def _average(self, values: list[float]) -> float:
        if not values:
            return 0.0
        return round(sum(values) / len(values), 2)
