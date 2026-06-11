from pydantic import BaseModel


class OCRSuccessResponse(BaseModel):
    status: str
    filename: str
    extracted_text: str
    confidence: float
    file_type: str


class ErrorResponse(BaseModel):
    status: str
    message: str
