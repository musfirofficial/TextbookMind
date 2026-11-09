import asyncio
import json
import re
from pathlib import Path
from typing import Annotated

from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from rag.crud.textbook import (
    create_textbook as create_textbook_crud,
    get_textbook_by_id as get_textbook_by_id_crud,
    get_textbook_by_title as get_textbook_by_title_crud,
    get_textbooks as get_textbooks_crud,
)
from rag.db.session import get_session
from rag.exceptions import ConfigError, IngestionError, QuotaExceededError
from rag.ingest import (
    index_and_persist_to_chroma,
    index_pages_to_chroma,
    save_extracted_dataset,
)
from rag.schema.textbook import (
    TextBookCreate,
    TextBookDatasetCreate,
    TextBookResponse,
)

ROOT_DIR = Path(__file__).resolve().parents[4]
router = APIRouter(prefix="/textbook", tags=["Text Book"])


async def _process_and_save_textbook(
    pdf_path: Path,
    title: str,
    grade: int | None,
    subject: str | None,
    medium: str,
    edition: int | str | None,
    db: AsyncSession,
) -> TextBookResponse:
    """Shared ingestion coordinator for both JSON path and file upload endpoints."""
    existing = await get_textbook_by_title_crud(db=db, title=title)
    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"A textbook with title '{title}' already exists in the database.",
        )

    try:
        # Run heavy CPU ingestion in a separate thread so asyncio loop doesn't freeze
        collection_name, total_pages = await asyncio.to_thread(
            index_and_persist_to_chroma,
            file=str(pdf_path),
            grade=grade,
            subject=subject,
            medium=medium,
            edition=edition,
            auto_ingest=True,
        )

        textbook_in = TextBookCreate(
            title=title,
            path=str(pdf_path),
            grade=grade,
            subject=subject,
            medium=medium,
            edition=edition,
        )

        new_textbook = await create_textbook_crud(
            db=db,
            textbook_in=textbook_in,
            collection_name=collection_name,
            total_pages=total_pages,
        )
        return new_textbook

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except QuotaExceededError as e:
        raise HTTPException(status_code=429, detail=f"API Quota Exceeded: {e}")
    except ConfigError as e:
        raise HTTPException(status_code=500, detail=f"Server Configuration Error: {e}")
    except IngestionError as e:
        raise HTTPException(status_code=400, detail=f"Ingestion Error: {e}")
    except IntegrityError:
        raise HTTPException(
            status_code=409,
            detail="A textbook with this title or path already exists in the database.",
        )
    except SQLAlchemyError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Database error: {e!s}",
        )


# ---------------------------------------------------------------------------
# Endpoint 1: Upload a Textbook PDF (Admin Panel / Multipart Form)
# ---------------------------------------------------------------------------
@router.post("/upload", response_model=TextBookResponse, status_code=201)
async def upload_textbook(
    db: Annotated[AsyncSession, Depends(get_session)],
    file: UploadFile = File(..., description="PDF textbook file to upload"),
    title: str = Form(..., min_length=4, max_length=100, description="Book title"),
    grade: int | None = Form(None, description="Grade level (e.g. 10 or 11)"),
    subject: str | None = Form(None, description="Subject (e.g. history, science)"),
    medium: str = Form("english", description="Medium of instruction"),
    edition: int | str | None = Form(None, description="Syllabus edition / year"),
):
    """
    Used by the Admin Panel:
    Accepts a direct PDF file upload, saves it into managed `data/books/`,
    and triggers automated ingestion and indexing.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    # Save uploaded PDF to managed storage folder
    books_dir = ROOT_DIR / "data" / "books"
    books_dir.mkdir(parents=True, exist_ok=True)
    destination = books_dir / file.filename

    content = await file.read()
    with open(destination, "wb") as f:
        f.write(content)

    return await _process_and_save_textbook(
        pdf_path=destination,
        title=title,
        grade=grade,
        subject=subject,
        medium=medium,
        edition=edition,
        db=db,
    )


# ---------------------------------------------------------------------------
# Endpoint 2: Ingest from Existing File Path (Scripting / JSON Body)
# ---------------------------------------------------------------------------
@router.post("/", response_model=TextBookResponse, status_code=201)
async def create_textbook_from_path(
    textbook_in: TextBookCreate, db: Annotated[AsyncSession, Depends(get_session)]
):
    """
    Used for local scripts, CLI, or already existing files on disk.
    Accepts a JSON payload with file path and metadata.
    """
    pdf_path = Path(textbook_in.path)
    if not pdf_path.is_absolute():
        candidate = ROOT_DIR / pdf_path
        if candidate.exists():
            pdf_path = candidate
        else:
            pdf_path = pdf_path.resolve()

    if not pdf_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"File not found at path: {textbook_in.path}",
        )

    return await _process_and_save_textbook(
        pdf_path=pdf_path,
        title=textbook_in.title,
        grade=textbook_in.grade,
        subject=textbook_in.subject,
        medium=textbook_in.medium,
        edition=textbook_in.edition,
        db=db,
    )


# ---------------------------------------------------------------------------
# Endpoint 3: Ingest Pre-Extracted JSON Dataset Payload (Local -> Cloud)
# ---------------------------------------------------------------------------
@router.post("/dataset", response_model=TextBookResponse, status_code=201)
async def create_textbook_from_dataset(
    dataset_in: TextBookDatasetCreate,
    db: Annotated[AsyncSession, Depends(get_session)],
):
    """
    Accepts pre-extracted JSON page documents directly.
    Indexes into ChromaDB in ~2 seconds and registers the textbook record in the database.
    Zero Gemini calls on server, zero gateway timeouts!
    """
    if not dataset_in.pages:
        raise HTTPException(status_code=400, detail="Pages list cannot be empty.")

    existing = await get_textbook_by_title_crud(db=db, title=dataset_in.title)
    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"A textbook with title '{dataset_in.title}' already exists in the database.",
        )

    clean_stem = re.sub(r"[^\w\-]", "_", dataset_in.title)
    extracted_path = f"data/extracted/{clean_stem}.json"

    # Save to data/extracted permanently
    save_extracted_dataset(extracted_path, dataset_in.pages)

    try:
        collection_name, total_pages = await asyncio.to_thread(
            index_pages_to_chroma,
            pages=dataset_in.pages,
            grade=dataset_in.grade,
            subject=dataset_in.subject,
            medium=dataset_in.medium,
            edition=dataset_in.edition,
            fallback_stem=clean_stem,
        )

        textbook_in = TextBookCreate(
            title=dataset_in.title,
            path=extracted_path,
            grade=dataset_in.grade,
            subject=dataset_in.subject,
            medium=dataset_in.medium,
            edition=dataset_in.edition,
        )

        new_textbook = await create_textbook_crud(
            db=db,
            textbook_in=textbook_in,
            collection_name=collection_name,
            total_pages=total_pages,
        )
        return new_textbook

    except IntegrityError:
        raise HTTPException(
            status_code=409,
            detail="A textbook with this title or path already exists in the database.",
        )
    except SQLAlchemyError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Database error: {e!s}",
        )


# ---------------------------------------------------------------------------
# Endpoint 4: Upload a Pre-Extracted JSON Dataset File (Admin Panel)
# ---------------------------------------------------------------------------
@router.post("/dataset/upload", response_model=TextBookResponse, status_code=201)
async def upload_textbook_dataset(
    db: Annotated[AsyncSession, Depends(get_session)],
    file: UploadFile = File(..., description="Pre-extracted JSON dataset file"),
    title: str = Form(..., min_length=4, max_length=100, description="Book title"),
    grade: int | None = Form(None, description="Grade level"),
    subject: str | None = Form(None, description="Subject"),
    medium: str = Form("english", description="Medium"),
    edition: int | str | None = Form(None, description="Syllabus edition"),
):
    """
    Accepts a pre-extracted JSON file upload (e.g. history_G-11_E.json).
    Indexes into ChromaDB in ~2 seconds and registers the textbook in the database.
    """
    if not file.filename or not file.filename.lower().endswith(".json"):
        raise HTTPException(
            status_code=400,
            detail="Only JSON dataset files are supported for this endpoint.",
        )

    existing = await get_textbook_by_title_crud(db=db, title=title)
    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"A textbook with title '{title}' already exists in the database.",
        )

    content = await file.read()
    try:
        pages = json.loads(content.decode("utf-8"))
        if not isinstance(pages, list) or len(pages) == 0:
            raise ValueError("JSON must contain a non-empty array of page objects.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON dataset file: {e}")

    # Save to data/extracted permanently
    extracted_dir = ROOT_DIR / "data" / "extracted"
    extracted_dir.mkdir(parents=True, exist_ok=True)
    destination = extracted_dir / file.filename
    with open(destination, "wb") as f:
        f.write(content)

    clean_stem = Path(file.filename).stem

    try:
        collection_name, total_pages = await asyncio.to_thread(
            index_pages_to_chroma,
            pages=pages,
            grade=grade,
            subject=subject,
            medium=medium,
            edition=edition,
            fallback_stem=clean_stem,
        )

        textbook_in = TextBookCreate(
            title=title,
            path=str(destination),
            grade=grade,
            subject=subject,
            medium=medium,
            edition=edition,
        )

        new_textbook = await create_textbook_crud(
            db=db,
            textbook_in=textbook_in,
            collection_name=collection_name,
            total_pages=total_pages,
        )
        return new_textbook

    except IntegrityError:
        raise HTTPException(
            status_code=409,
            detail="A textbook with this title or path already exists in the database.",
        )
    except SQLAlchemyError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Database error: {e!s}",
        )


# ---------------------------------------------------------------------------
# Endpoint 5: List All Textbooks
# ---------------------------------------------------------------------------
@router.get("/", response_model=list[TextBookResponse])
async def list_textbooks(
    db: Annotated[AsyncSession, Depends(get_session)],
    skip: int = Query(0, ge=0, description="Number of textbooks to skip"),
    limit: int = Query(100, ge=1, le=500, description="Max textbooks to return"),
):
    """List all textbooks registered in the database."""
    return await get_textbooks_crud(db=db, skip=skip, limit=limit)


# ---------------------------------------------------------------------------
# Endpoint 6: Get Textbook by ID
# ---------------------------------------------------------------------------
@router.get("/{textbook_id}", response_model=TextBookResponse)
async def get_textbook(
    textbook_id: UUID,
    db: Annotated[AsyncSession, Depends(get_session)],
):
    """Retrieve a single textbook by its unique UUID."""
    textbook = await get_textbook_by_id_crud(db=db, textbook_id=textbook_id)
    if not textbook:
        raise HTTPException(status_code=404, detail="Textbook not found.")
    return textbook

