from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    features: list[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="Four numerical features used by the ticket classifier.",
    )


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    model_name: str
    model_alias: str
