from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    status: str
    filename: str | None
    format: str
    width: int
    height: int
    size_bytes: int
    confidence_threshold: float