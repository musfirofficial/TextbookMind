from pydantic import BaseModel, Field


class TextBookBase(BaseModel):
    title: str = Field(
        min_length=4, max_length=100, description="Title of the textbook"
    )
    path: str = Field(min_length=4, max_length=255, description="Path of the textbook")
    collection_name: str = Field(
        min_length=4, max_length=100, description="Collection name of the textbook"
    )
    total_pages: int = Field(gt=0, description="Total number of pages in the textbook")


class TextBookCreate(TextBookBase):
    pass
