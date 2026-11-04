import uuid
from datetime import datetime

from sqlalchemy import DateTime, Integer, String
from sqlalchemy import Uuid as UUID
from sqlalchemy.orm import Mapped, mapped_column

from .session import Base

local_tz = datetime.now().astimezone().tzinfo


class TextBook(Base):
    __tablename__ = "textbooks"

    textbook_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    path: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    collection_name: Mapped[str] = mapped_column(String(255), nullable=False)
    total_pages: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(local_tz)
    )
