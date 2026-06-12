from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.ocr_service import OCRProcessingError, OCRProcessor, UnsupportedFileTypeError
from app.schemas import OCRSuccessResponse


BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("ocr-app")

app = FastAPI(title="Mero dokan", version="1.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

ocr_processor: OCRProcessor | None = None


@app.on_event("startup")
async def startup_event() -> None:
    global ocr_processor
    try:
        ocr_processor = OCRProcessor()
        if ocr_processor.is_available:
            logger.info("OCR processor initialized successfully.")
        else:
            logger.warning("OCR processor initialized but Tesseract is not available. OCR endpoints will return errors.")
    except Exception as exc:
        logger.error("Failed to initialize OCR processor: %s", exc)
        ocr_processor = OCRProcessor()  # Initialize with graceful fallback


@app.get("/")
async def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/api/ocr", response_model=OCRSuccessResponse)
async def extract_ocr(file: UploadFile = File(default=None)) -> JSONResponse:
    if file is None:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "No file uploaded."},
        )

    filename = file.filename or ""
    if not filename.strip():
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "Uploaded file must have a valid filename."},
        )

    try:
        file_bytes = await file.read()
        if not file_bytes:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "Uploaded file is empty."},
            )

        if len(file_bytes) > MAX_FILE_SIZE_BYTES:
            return JSONResponse(
                status_code=413,
                content={
                    "status": "error",
                    "message": "Uploaded file is too large. Maximum supported size is 25 MB.",
                },
            )

        if ocr_processor is None:
            raise OCRProcessingError("OCR processor is not available.")

        extracted_text, confidence, file_type = ocr_processor.extract_text(filename, file_bytes)
        response = OCRSuccessResponse(
            status="success",
            filename=filename,
            extracted_text=extracted_text,
            confidence=confidence,
            file_type=file_type,
        )
        return JSONResponse(status_code=200, content=response.model_dump())

    except UnsupportedFileTypeError as exc:
        return JSONResponse(status_code=415, content={"status": "error", "message": str(exc)})
    except OCRProcessingError as exc:
        logger.exception("OCR processing error: %s", exc)
        return JSONResponse(status_code=500, content={"status": "error", "message": str(exc)})
    except Exception as exc:
        logger.exception("Unexpected server error: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": "Unexpected server error during OCR processing."},
        )
    finally:
        if file is not None:
            await file.close()
