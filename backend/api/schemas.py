from pydantic import BaseModel, Field


class TranscriptRequest(BaseModel):
    transcript: str


class CompareRequest(BaseModel):
    transcript_1: str = Field(min_length=1)
    transcript_2: str = Field(min_length=1)
