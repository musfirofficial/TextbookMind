import os
from collections.abc import AsyncGenerator

from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from ..rag.core.config import ROOT_DIR

# 1. Load environment variables from .env file


load_dotenv(ROOT_DIR / ".env")

DEFAULT_DB_PATH = ROOT_DIR / "data" / "textbookmind.db"
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite+aiosqlite:///{DEFAULT_DB_PATH}")

# 2. Create an asynchronous engine
engine = create_async_engine(
    DATABASE_URL,
    echo=True,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
)

# 3. Create an asynchronous session maker
async_session = async_sessionmaker(
    bind=engine, class_=AsyncSession, expire_on_commit=False
)


# 4. Define a base class for declarative models
class Base(DeclarativeBase):
    pass


# 5. Create a function to get an asynchronous session
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session() as session:
        yield session


# 6. Create a function to initialize the database
async def init_db():
    """Creates all database tables defined in Base models if they don't exist."""
    import db.models  # noqa: F401 - Ensure models are registered in Base.metadata

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
