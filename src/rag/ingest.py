import json
import re
from pathlib import Path

import chromadb
import pymupdf
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore

from rag.config import get_gemini_api_key
from rag.exceptions import (
    ConfigError,
    IngestionError,
    QuotaExceededError,
    VisionAPIError,
)

from .config import init_llama_index
from .parser import (
    TOC_EXTRA_PAGES,
    classify_page,
    detect_repeating_headers_footers,
    extract_digital_page_markdown,
    extract_printed_page_number,
    extract_table_of_contents,
    find_watermark_xrefs,
    map_page_to_chapter,
)
from .vision import transcribe_page_with_gemini

DEFAULT_CHROMA_DIR = str(Path(__file__).resolve().parents[2] / "data" / "chroma_db")
CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"


def get_checkpoint_path(file_path: str) -> Path:
    """Returns the JSON checkpoint cache path for a given textbook file."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    clean_stem = re.sub(r"[^\w\-]", "_", Path(file_path).stem)
    return CACHE_DIR / f"{clean_stem}_checkpoint.json"


def load_checkpoint(cache_path: Path) -> dict[int, dict]:
    """Loads checkpoint cache. Returns dict of {pdf_page: page_dict}."""
    if not cache_path.exists():
        return {}
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            pages = data.get("pages", [])
            return {p["pdf_page"]: p for p in pages if "pdf_page" in p}
    except Exception as e:
        print(f"[WARNING] Could not read checkpoint cache: {e}. Starting fresh.")
        return {}


def save_checkpoint(
    cache_path: Path, pages: list[dict], file_path: str, total_pages: int
):
    """Saves current extracted pages to checkpoint cache."""
    try:
        data = {
            "document_name": Path(file_path).name,
            "total_pages": total_pages,
            "completed_count": len(pages),
            "pages": pages,
        }
        temp_path = cache_path.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        temp_path.replace(cache_path)
    except Exception as e:
        print(f"[WARNING] Failed to write checkpoint cache: {e}")


def remove_checkpoint(cache_path: Path):
    """Removes checkpoint cache upon 100% successful ingestion."""
    try:
        if cache_path.exists():
            cache_path.unlink()
            print(f"[CACHE] Ingestion complete. Cleaned up checkpoint: {cache_path.name}")
    except Exception as e:
        print(f"[WARNING] Could not delete checkpoint {cache_path}: {e}")


EXTRACTED_DIR = Path(__file__).resolve().parents[2] / "data" / "extracted"


def get_extracted_path(file_path: str) -> Path:
    """Returns the permanent extracted dataset path (data/extracted/<clean_stem>.json)."""
    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    clean_stem = re.sub(r"[^\w\-]", "_", Path(file_path).stem)
    return EXTRACTED_DIR / f"{clean_stem}.json"


def save_extracted_dataset(file_path: str, pages: list[dict]) -> Path:
    """Saves the final extracted textbook pages permanently to data/extracted/<clean_stem>.json."""
    out_path = get_extracted_path(file_path)
    temp_path = out_path.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)
    temp_path.replace(out_path)
    print(f"\n[DATASET] Saved permanent textbook dataset ({len(pages)} pages) to: {out_path}")
    return out_path


def load_extracted_dataset(file_path: str) -> list[dict] | None:
    """Loads permanent extracted dataset if it already exists."""
    out_path = get_extracted_path(file_path)
    if out_path.exists():
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                pages = json.load(f)
                if isinstance(pages, list) and len(pages) > 0:
                    return pages
        except Exception as e:
            print(f"[WARNING] Could not load extracted dataset from {out_path}: {e}")
    return None


def parse_and_prepare_textbook(
    file_path: str,
    grade: int | None = None,
    subject: str | None = None,
    medium: str = "english",
    edition: int | str | None = None,
    auto_ingest: bool = False,
    include_images: bool = True,
    toc_extra_pages: int = TOC_EXTRA_PAGES,
    force_reextract: bool = False,
) -> list[dict]:
    """
    Main Pre-Chunking Ingestion Coordinator:
    1. Checks if a permanent extracted dataset exists (loads in 0.05s if present).
    2. Otherwise pre-scans document for repeating watermarks & headers/footers.
    3. Extracts Table of Contents (TOC) via Gemini Vision on rendered TOC pages.
    4. Classifies pages into Text-Only vs Visual (tables, large images, vector charts).
       If include_images is False, only pages with tables or vector drawings are sent to Gemini.
    5. Displays a summary dashboard with TOC and page counts.
    6. Pauses for user confirmation ('Proceed') if auto_ingest is False.
    7. Extracts pages in sequential order (resumable via checkpoint cache).
    8. Attaches complete metadata and saves permanently to data/extracted/<stem>.json.
    """
    path_obj = Path(file_path).resolve()
    if not path_obj.exists():
        raise FileNotFoundError(f"Textbook file not found at path: {file_path}")

    # Check if a completed permanent dataset already exists
    if not force_reextract:
        saved_dataset = load_extracted_dataset(str(path_obj))
        if saved_dataset:
            print("\n" + "=" * 65)
            print(f"  DATASET FOUND: {get_extracted_path(str(path_obj)).name}")
            print("=" * 65)
            print(
                f"[DATASET] Loaded {len(saved_dataset)} pages instantly from disk ($0 API cost, 0s).\n"
            )
            return saved_dataset

    try:
        doc = pymupdf.open(str(path_obj))
    except Exception as exc:
        raise IngestionError(f"Failed to open PDF '{file_path}': {exc}") from exc

    total_pages = len(doc)
    if total_pages == 0:
        doc.close()
        raise ValueError(f"PDF '{file_path}' has 0 pages.")

    print("\n" + "=" * 65)
    print(f"  PRE-SCANNING TEXTBOOK: {path_obj.name}")
    print("=" * 65)

    # ------------------------------------------------------------------
    # Step 1: Pre-Scan Watermarks & Repeating Headers/Footers
    # ------------------------------------------------------------------
    watermark_xrefs = find_watermark_xrefs(doc)
    detect_repeating_headers_footers(doc)

    # ------------------------------------------------------------------
    # Step 2: Extract Table of Contents (Bookmarks -> Groq Fallback)
    # ------------------------------------------------------------------
    print("Extracting Table of Contents (TOC)...")
    toc, toc_start_page = extract_table_of_contents(doc, extra_pages=toc_extra_pages)

    # ------------------------------------------------------------------
    # Step 3: Classify Pages (Text-Only vs Visual)
    # ------------------------------------------------------------------
    page_classifications = []
    for p_idx, page in enumerate(doc):
        # Pages before the Table of Contents are front matter and skipped from Gemini Vision
        is_front_matter = (toc_start_page is not None and p_idx < toc_start_page)
        if is_front_matter:
            cls_info = {
                "is_visual": False,
                "has_tables": False,
                "has_images": False,
                "has_drawings": False,
                "is_front_matter": True,
            }
        else:
            cls_info = classify_page(
                page, watermark_xrefs=watermark_xrefs, include_images=include_images
            )
            cls_info["is_front_matter"] = False
        page_classifications.append(cls_info)

    text_only_count = sum(1 for c in page_classifications if not c["is_visual"])
    visual_count = total_pages - text_only_count
    front_matter_count = sum(1 for c in page_classifications if c.get("is_front_matter"))
    table_count = sum(1 for c in page_classifications if c["has_tables"])
    image_count = sum(1 for c in page_classifications if c["has_images"])
    drawing_count = sum(1 for c in page_classifications if c["has_drawings"])

    text_pct = (text_only_count / total_pages) * 100
    visual_pct = (visual_count / total_pages) * 100

    # ------------------------------------------------------------------
    # Step 4: Display Pre-Scan Summary & TOC Dashboard
    # ------------------------------------------------------------------
    print("\n" + "-" * 65)
    print("  TABLE OF CONTENTS")
    print("-" * 65)
    if toc:
        for ch in toc:
            ch_num = ch.get("chapter_number", "?")
            title = ch.get("title", "Untitled")
            s_page = ch.get("start_page", "?")
            e_page = ch.get("end_page", "?")
            print(f"  [Chapter {ch_num:>2}] {title:<42} (Pages {s_page} - {e_page})")
    else:
        print("  (No structured Table of Contents detected in first 25 pages)")

    print("\n" + "-" * 65)
    print("  PAGE CLASSIFICATION SUMMARY")
    print("-" * 65)
    print(f"  Textbook File:       {path_obj.name}")
    print(
        f"  Metadata:            Grade: {grade or 'N/A'} | Subject: {subject or 'N/A'} | Medium: {medium} | Edition: {edition or 'N/A'}"
    )
    print(f"  Total Pages:         {total_pages}")
    if not include_images:
        print("  Image Routing:       Disabled (Only tables & vector drawings sent to Gemini)")
    if front_matter_count > 0:
        print(f"  Front-Matter Pages:  {front_matter_count} (Pages 1 - {front_matter_count}) -> Skipped Gemini ($0)")
    print(
        f"  Text-Only Pages:     {text_only_count} ({text_pct:.1f}%) -> Local PyMuPDF ($0)"
    )
    print(
        f"  Visual Pages:        {visual_count} ({visual_pct:.1f}%) -> Gemini Flash Lite"
    )
    print(f"    - With Tables:     {table_count}")
    print(f"    - With Diagrams:   {image_count}")
    print(f"    - With Vector Art: {drawing_count}")
    print(f"  Watermarks Excluded: {len(watermark_xrefs)} repeating asset XREFs")
    print("-" * 65)

    # ------------------------------------------------------------------
    # Step 5: Approval Gate (Proceed / Cancel)
    # ------------------------------------------------------------------
    if not auto_ingest:
        user_choice = input(
            "\nType 'Proceed' to begin extraction (or 'n' to cancel): "
        ).strip()
        if user_choice.lower() not in ["proceed", "y", "yes"]:
            print("\n[INFO] Ingestion cancelled by user. Exiting safely.\n")
            doc.close()
            return []
    else:
        print("\n[AUTO INGEST] auto_ingest=True: Proceeding automatically...")

    # Early check: Ensure Gemini API key is available before starting extraction if visual pages exist
    if visual_count > 0:
        get_gemini_api_key()

    # ------------------------------------------------------------------
    # Step 6: Ordered Page Extraction with Checkpoint & Fallback
    # ------------------------------------------------------------------
    cache_path = get_checkpoint_path(str(path_obj))
    cached_pages = load_checkpoint(cache_path)
    if cached_pages:
        print(
            f"\n[RESUME] Found existing checkpoint cache ({len(cached_pages)}/{total_pages} pages completed). Resuming..."
        )

    print(f"\nExtracting {total_pages} pages in sequential order...")
    extracted_pages = []

    # Calculate page offset from first printed book page if available
    first_printed_page = None
    first_printed_pdf_idx = None
    # 1. Search for Book Page 1 specifically
    for p_idx in range(min(35, total_pages)):
        bp = extract_printed_page_number(doc[p_idx])
        if bp == 1:
            first_printed_page = bp
            first_printed_pdf_idx = p_idx + 1
            break

    # 2. If Book Page 1 wasn't found (e.g. unnumbered intro), find first low number after TOC
    if first_printed_page is None:
        start_scan = (toc_start_page or 0)
        for p_idx in range(start_scan, min(start_scan + 25, total_pages)):
            bp = extract_printed_page_number(doc[p_idx])
            if bp is not None and 0 < bp <= 15:
                first_printed_page = bp
                first_printed_pdf_idx = p_idx + 1
                break

    page_offset = (
        (first_printed_pdf_idx - first_printed_page)
        if (first_printed_page and first_printed_pdf_idx)
        else 0
    )
    if page_offset:
        print(f"[PAGE OFFSET] Detected Book Page {first_printed_page} at PDF Page {first_printed_pdf_idx} (Offset: {page_offset})")

    for p_idx, page in enumerate(doc):
        pdf_page = p_idx + 1
        classification = page_classifications[p_idx]

        # 1. Resume from Checkpoint if already extracted
        if pdf_page in cached_pages:
            cached_record = cached_pages[pdf_page]
            extracted_pages.append(cached_record)
            ch_num = cached_record.get("chapter_number")
            ch_label = f"Ch.{ch_num}" if ch_num else "General"
            b_page = cached_record.get("book_page") or "N/A"
            method = cached_record.get("extraction_method", "cached")
            print(
                f"  [{pdf_page:>3}/{total_pages}] [{method:<17}] {ch_label:<8} (Book Page: {b_page}) [CACHED]"
            )
            continue

        book_page = extract_printed_page_number(page)
        # Infer book page from offset if unprinted (e.g. full-page diagrams/illustrations)
        if (
            book_page is None
            and page_offset
            and pdf_page > page_offset
            and not classification.get("is_front_matter", False)
        ):
            book_page = pdf_page - page_offset

        # Map to chapter
        if classification.get("is_front_matter", False):
            chapter_num = None
            chapter_title = "Front Matter"
        else:
            chapter_num, chapter_title = map_page_to_chapter(
                book_page=book_page,
                pdf_page=pdf_page,
                toc=toc,
                page_offset=page_offset,
            )

        # Route extraction path:
        # Visual pages after the TOC go to Gemini Vision; front matter and text-only go to digital
        if classification["is_visual"] and not classification.get("is_front_matter", False):
            # Render page to PNG bytes for Gemini Vision
            try:
                pix = page.get_pixmap(dpi=150)
                image_bytes = pix.tobytes("png")
                page_text = transcribe_page_with_gemini(image_bytes, page_num=pdf_page)
                extraction_method = "gemini_vision"
            except (QuotaExceededError, ConfigError):
                doc.close()
                save_checkpoint(cache_path, extracted_pages, str(path_obj), total_pages)
                raise
            except VisionAPIError as ve:
                print(
                    f"\n[FALLBACK] Page {pdf_page}: Gemini vision extraction failed after retries: {ve}. "
                    f"Falling back to local digital extraction."
                )
                page_text = extract_digital_page_markdown(doc, page)
                extraction_method = "digital_fallback"
            except Exception as e:
                print(
                    f"\n[FALLBACK] Page {pdf_page}: Unexpected error during vision extraction: {e}. "
                    f"Falling back to local digital extraction."
                )
                page_text = extract_digital_page_markdown(doc, page)
                extraction_method = "digital_fallback"
        else:
            page_text = extract_digital_page_markdown(doc, page)
            extraction_method = "digital"

        # If include_images is False, raster images are not routed to Gemini Vision for transcription;
        # only vector drawings (diagrams, flowcharts, charts) represent transcribed figures.
        has_figures = (
            classification["has_drawings"]
            if not include_images
            else (classification["has_images"] or classification["has_drawings"])
        )

        page_record = {
            "document_name": path_obj.name,
            "grade": grade,
            "subject": subject,
            "medium": medium,
            "edition": edition,
            "pdf_page": pdf_page,
            "book_page": book_page,
            "chapter_number": chapter_num,
            "chapter_title": chapter_title,
            "has_figures": has_figures,
            "has_tables": classification["has_tables"],
            "extraction_method": extraction_method,
            "text": page_text,
        }
        extracted_pages.append(page_record)

        # Save progress to checkpoint cache
        save_checkpoint(cache_path, extracted_pages, str(path_obj), total_pages)

        if classification.get("is_front_matter", False):
            ch_label = "Front Matter"
        elif chapter_num:
            ch_label = f"Ch.{chapter_num}"
        else:
            ch_label = "General"

        print(
            f"  [{pdf_page:>3}/{total_pages}] [{extraction_method:<17}] {ch_label:<12} (Book Page: {book_page or 'N/A'})"
        )

    doc.close()

    # Once 100% completed successfully, delete the checkpoint cache and save permanent dataset
    remove_checkpoint(cache_path)
    save_extracted_dataset(str(path_obj), extracted_pages)
    print(f"\n[SUCCESS] Successfully extracted {len(extracted_pages)} pages.\n")
    return extracted_pages


def index_pages_to_chroma(
    pages: list[dict],
    grade: int | None = None,
    subject: str | None = None,
    medium: str = "english",
    edition: int | str | None = None,
    fallback_stem: str = "textbook",
    persist_dir: str = DEFAULT_CHROMA_DIR,
) -> tuple[str, int]:
    """
    Indexes an already-extracted list of page dicts directly into ChromaDB.
    Returns (collection_name, total_pages).
    """
    if not pages:
        print("[INFO] No pages to index.")
        return "", 0

    init_llama_index()

    collection_name = (
        f"grade-{grade}-{subject}-{medium}" if (grade and subject) else fallback_stem
    )

    # Initialize ChromaDB
    chroma_client = chromadb.PersistentClient(path=persist_dir)
    chroma_collection = chroma_client.get_or_create_collection(name=collection_name)

    # Wire ChromaDB to LlamaIndex
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Wrap Pages into LlamaIndex Documents with complete metadata
    documents = [
        Document(
            text=page["text"],
            metadata={
                "file_name": page.get("document_name", fallback_stem),
                "grade": page.get("grade", grade),
                "subject": page.get("subject", subject),
                "medium": page.get("medium", medium),
                "edition": page.get("edition", edition),
                "pdf_page": page["pdf_page"],
                "book_page": page.get("book_page"),
                "chapter_number": page.get("chapter_number"),
                "chapter_title": page.get("chapter_title"),
                "has_figures": page.get("has_figures", False),
                "has_tables": page.get("has_tables", False),
                "extraction_method": page.get("extraction_method", "dataset"),
            },
            excluded_embed_metadata_keys=["extraction_method"],
            excluded_llm_metadata_keys=["extraction_method"],
        )
        for page in pages
        if page.get("text", "").strip() and page.get("chapter_title") != "Front Matter"
    ]

    # Safe Chunking within 512 embedding limit
    node_parser = SentenceSplitter(
        chunk_size=450,
        chunk_overlap=80,
        paragraph_separator="\n\n",
    )

    # Build Vector Store
    index = VectorStoreIndex.from_documents(
        documents=documents,
        storage_context=storage_context,
        transformations=[node_parser],
        show_progress=True,
    )

    print(
        f"Successfully indexed {len(documents)} pages into collection '{collection_name}' at '{persist_dir}'."
    )
    return collection_name, len(pages)


def index_and_persist_to_chroma(
    file: str,
    grade: int | None = None,
    subject: str | None = None,
    medium: str = "english",
    edition: int | str | None = None,
    auto_ingest: bool = True,
    include_images: bool = True,
    persist_dir: str = DEFAULT_CHROMA_DIR,
) -> tuple[str, int]:
    """
    Ingests textbook PDF (or cached dataset) and indexes into ChromaDB.
    Returns (collection_name, total_pages).
    """
    pages = parse_and_prepare_textbook(
        file_path=file,
        grade=grade,
        subject=subject,
        medium=medium,
        edition=edition,
        auto_ingest=auto_ingest,
        include_images=include_images,
    )

    file_stem = Path(file).stem.replace(" ", "-")
    return index_pages_to_chroma(
        pages=pages,
        grade=grade,
        subject=subject,
        medium=medium,
        edition=edition,
        fallback_stem=file_stem,
        persist_dir=persist_dir,
    )
