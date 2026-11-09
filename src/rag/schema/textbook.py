from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class TextBookBase(BaseModel):
    title: str = Field(
        min_length=4, max_length=100, description="Title of the textbook"
    )
    path: str = Field(min_length=4, max_length=255, description="Path of the textbook")
    edition: int | str | None = Field(
        default=None, description="Year or edition of syllabus"
    )
    grade: int | None = Field(default=None, description="Grade of textbook")
    subject: str | None = Field(default=None, description="Subject of textbook")
    medium: str = Field(default="english", description="Language medium")


class TextBookCreate(TextBookBase):
    pass


class TextBookDatasetCreate(BaseModel):
    title: str = Field(
        min_length=4, max_length=100, description="Title of the textbook"
    )
    edition: int | str | None = Field(
        default=None, description="Year or edition of syllabus"
    )
    grade: int | None = Field(default=None, description="Grade of textbook")
    subject: str | None = Field(default=None, description="Subject of textbook")
    medium: str = Field(default="english", description="Language medium")
    pages: list[dict] = Field(..., description="Pre-extracted page dicts from local run")


class TextBookResponse(TextBookBase):
    textbook_id: UUID
    created_at: datetime
    collection_name: str
    total_pages: int

    model_config = ConfigDict(
        from_attributes=True
    )  # It allows Pydantic to read data from SQLAlchemy objects
