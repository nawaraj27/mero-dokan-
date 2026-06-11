# Local OCR Web Application

This project is a Python-only local OCR web application with:

- `FastAPI` backend API
- real text extraction using `Tesseract OCR`
- OCR configured for `English + Nepali` by default
- support for `JPG`, `JPEG`, `PNG`, `WEBP`, `BMP`, `TIFF`, and multi-page `PDF`
- PDF-to-image conversion using `PyMuPDF`
- image preprocessing for better OCR accuracy:
  - grayscale conversion
  - noise removal
  - contrast enhancement
  - thresholding
  - resizing for readability

## API response format

Successful OCR requests return:

```json
{
  "status": "success",
  "filename": "file_name",
  "extracted_text": "REAL OCR OUTPUT TEXT HERE",
  "confidence": 0.0,
  "file_type": "image/pdf"
}
```

Error responses return:

```json
{
  "status": "error",
  "message": "Meaningful error message"
}
```

## 1. Install Tesseract

This app performs real OCR and requires a local Tesseract installation.

### Windows

1. Install Tesseract OCR on your machine.
2. Make sure `tesseract.exe` is available in `PATH`.
3. Install the Tesseract language data for `eng` and `nep`.
4. If it is not in `PATH`, set an environment variable named `TESSERACT_CMD` to the full path of `tesseract.exe`.

Example:

```powershell
$env:TESSERACT_CMD='C:\Program Files\Tesseract-OCR\tesseract.exe'
$env:OCR_LANGUAGES='eng+nep'
```

## 2. Create a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 3. Install Python dependencies

```powershell
pip install -r requirements.txt
```

## 4. Run the application

```powershell
python run.py
```

Then open:

```text
http://127.0.0.1:8000
```

## API endpoints

### `GET /`

Returns the local web interface.

### `GET /health`

Health check endpoint.

### `POST /api/ocr`

Upload one supported file using `multipart/form-data` with field name `file`.

Example:

```powershell
curl -X POST http://127.0.0.1:8000/api/ocr `
  -F "file=@sample.png"
```

## Notes

- Multi-page PDFs are processed page by page, then combined into a single OCR text output.
- OCR uses `eng+nep` by default. You can change this with the `OCR_LANGUAGES` environment variable.
- If the OCR engine runs successfully but cannot detect text, `extracted_text` may be empty and `confidence` may be `0.0`.
- No Docker is used in this project.
