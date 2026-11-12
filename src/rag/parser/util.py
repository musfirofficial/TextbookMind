import base64
import json
import re
import time
import unicodedata
from collections import Counter
from pathlib import Path

import httpx
import pymupdf
import pymupdf4llm

from rag.core.config import GEMINI_API_URL, ROOT_DIR, get_gemini_api_key
from rag.prompts import gemini_toc_prompt

# ==============================================================================
# Configurable Parser Parameters (Easily tunable)
# ==============================================================================

MAX_TOC_SEARCH_PAGES = 25  # Max early pages to scan for Contents
TOC_EXTRA_PAGES = 2  # Extra pages to capture after "Contents" header (3 pages total)
WATERMARK_REPEAT_THRESHOLD = 5  # XREFs appearing on >= 5 pages are watermarks
MIN_IMAGE_DIM = 250  # Filters out tiny icons, bullets, and 1px lines
MIN_VECTOR_DIM = 250  # Filters out 1px line dividers, catches charts/diagrams
HEADER_MARGIN = 40  # Points from page top to inspect for running headers
FOOTER_MARGIN = 40  # Points from page bottom to inspect for footers


# ==============================================================================
# Find Repeating Watermarks
# ==============================================================================


def find_watermark_xrefs(
    doc: pymupdf.Document, threshold: int = WATERMARK_REPEAT_THRESHOLD
) -> set[int]:
    """
    Identifies image XREFs that appear repeatedly across pages.
    In PDFs, watermarks and background logos reuse the exact same XREF.
    """
    xref_counts = Counter()
    for page in doc:
        try:
            for img in page.get_images():
                xref = img[0]
                xref_counts[xref] += 1
        except Exception:
            continue

    watermark_xrefs = {
        xref for xref, count in xref_counts.items() if count >= threshold
    }
    return watermark_xrefs  # as a set


# ==============================================================================
# Find Table of Contents
# ==============================================================================


def find_toc_start_page(
    doc: pymupdf.Document, max_search_pages: int = MAX_TOC_SEARCH_PAGES
) -> int | None:
    """
    Locates the 0-indexed physical page where the Table of Contents begins.
    1. Checks for prominent TOC keywords (e.g. 'contents', 'table of contents', 'chapters', 'index').
    2. Fallback: Checks for chapter/page number patterns, excluding writer credits.
    """
    keywords = ["table of contents", "contents", "chapters", "content", "index"]
    for i in range(min(max_search_pages, len(doc))):
        try:
            t = doc[i].get_text().lower()
            for line in t.split("\n"):
                line_str = line.strip()
                if line_str in keywords:
                    return i
                if any(k in line_str for k in ["table of contents", "contents"]):
                    return i
        except Exception:
            continue

    # Fallback: scan for chapter entries with page numbers (avoiding contributor credit pages)
    for i in range(min(max_search_pages, len(doc))):
        try:
            t = doc[i].get_text()
            t_lower = t.lower()
            if "panel of writers" in t_lower or "panel of editors" in t_lower:
                continue
            numbered_with_pages = re.findall(
                r"\d{1,2}\.?\s+[A-Za-z][^\n]+\s*\n?\s*\d{1,3}", t
            )
            if len(numbered_with_pages) >= 3:
                # Check if previous page has vector drawings / blank text (e.g. Science Part 2 TOC page 1)
                if (
                    i > 0
                    and len(doc[i - 1].get_drawings()) > 100
                    and len(doc[i - 1].get_text().strip()) < 50
                ):
                    return i - 1
                return i
        except Exception:
            continue

    return None


# ==============================================================================
# Extract Contents From Found Table of Contents
# ==============================================================================


def extract_table_of_contents(
    doc: pymupdf.Document,
    max_search_pages: int = MAX_TOC_SEARCH_PAGES,
    extra_pages: int = TOC_EXTRA_PAGES,
    max_retries: int = 3,
) -> tuple[list[dict], int | None]:
    """
    Extracts the Table of Contents into a structured list of chapters, along with
    the physical 0-indexed page index where the TOC starts.
    Directly renders the detected TOC pages (up to extra_pages + 1) as images and sends
    them to Gemini Vision to extract a clean JSON list of chapters.
    Returns:
        (chapters_list, toc_start_page_idx)
    """
    toc_start_page = find_toc_start_page(doc, max_search_pages)
    if toc_start_page is None:
        print("[INFO] No Table of Contents detected in early pages.")
        return [], None

    # Render TOC pages (up to 1 + extra_pages) as 100 DPI JPEGs (~60-80KB each)
    end_scan = min(toc_start_page + 1 + extra_pages, len(doc))
    prompt_text = gemini_toc_prompt()
    parts = [{"text": prompt_text}]

    rendered_pages = 0
    for p_idx in range(toc_start_page, end_scan):
        # Stop early if we reach the start of textbook lesson body
        if p_idx > toc_start_page:
            try:
                early_text = doc[p_idx].get_text()[:250].lower()
                if any(
                    kw in early_text
                    for kw in [
                        "chapter 1",
                        "unit 1",
                        "unit 13",
                        "13.1",
                        "1.1",
                        "lesson 1",
                    ]
                ):
                    break
            except Exception:
                pass

        try:
            pix = doc[p_idx].get_pixmap(dpi=100)
            img_bytes = pix.tobytes("jpeg", jpg_quality=75)
            parts.append(
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": base64.b64encode(img_bytes).decode("utf-8"),
                    }
                }
            )
            rendered_pages += 1
        except Exception as e:
            print(f"[WARNING] Could not render TOC page {p_idx} for Gemini: {e}")

    if rendered_pages == 0:
        return [], toc_start_page

    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 4096,
        },
    }

    try:
        api_key = get_gemini_api_key()
    except Exception as e:
        print(f"[WARNING] Gemini API key not available for TOC extraction: {e}")
        return [], toc_start_page

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }

    content = ""
    for attempt in range(max_retries):
        try:
            with httpx.Client(http2=True, timeout=60.0) as client:
                response = client.post(
                    GEMINI_API_URL,
                    json=payload,
                    headers=headers,
                )

            if response.status_code == 200:
                data = response.json()
                candidates = data.get("candidates", [])
                if candidates:
                    parts_out = candidates[0].get("content", {}).get("parts", [])
                    content = "".join(p.get("text", "") for p in parts_out).strip()
                break
            else:
                if attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1))
                else:
                    print(
                        f"[WARNING] Gemini TOC extraction failed with HTTP {response.status_code}: {response.text[:200]}"
                    )
        except Exception as exc:
            if attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))
            else:
                print(f"[WARNING] Gemini TOC extraction connection error: {exc}")

    if not content:
        return [], toc_start_page

    # Parse JSON from model output
    chapters = []
    try:
        match = re.search(r"\[[\s\S]*\]", content)
        if match:
            chapters = json.loads(match.group(0))
        else:
            cleaned = content.strip()
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
            chapters = json.loads(cleaned)
    except Exception as e:
        print(f"[WARNING] Could not parse Gemini TOC JSON response: {e}")
        return [], toc_start_page

    if not isinstance(chapters, list):
        return [], toc_start_page

    # Validate and clean up chapters
    valid_chapters = []
    for idx, item in enumerate(chapters):
        if not isinstance(item, dict):
            continue
        try:
            ch_num = int(item.get("chapter_number", idx + 1))
        except (ValueError, TypeError):
            ch_num = idx + 1

        title = str(item.get("title", f"Chapter {ch_num}")).strip()

        try:
            start_p = int(item["start_page"])
        except (KeyError, ValueError, TypeError):
            continue

        end_p = item.get("end_page")
        if end_p is not None:
            try:
                end_p = int(end_p)
            except (ValueError, TypeError):
                end_p = None

        valid_chapters.append(
            {
                "chapter_number": ch_num,
                "title": title,
                "start_page": start_p,
                "end_page": end_p,
            }
        )

    # Sort chapters by start_page
    valid_chapters.sort(key=lambda c: c["start_page"])

    # Fill in missing end_page
    for i, ch in enumerate(valid_chapters):
        if ch["end_page"] is None or ch["end_page"] < ch["start_page"]:
            if i + 1 < len(valid_chapters):
                ch["end_page"] = max(
                    ch["start_page"], valid_chapters[i + 1]["start_page"] - 1
                )
            else:
                ch["end_page"] = ch["start_page"] + 20

    return valid_chapters, toc_start_page


# ==============================================================================
# Page Classification
# ==============================================================================


def classify_page(
    page: pymupdf.Page,
    watermark_xrefs: set[int],
    min_image_dim: int = MIN_IMAGE_DIM,
    min_vector_dim: int = MIN_VECTOR_DIM,
    include_images: bool = True,
) -> dict:
    """
    Classifies a page into 'text_only' vs 'visual'.
    - If include_images is True: pages with tables, large images, or vector charts are visual.
    - If include_images is False: ONLY pages with tables or vector charts are visual
      (embedded raster images do NOT trigger visual/Gemini routing).
    Defensively catches any image or vector errors so corrupted pages never crash the pipeline.
    """
    has_tables = False
    has_images = False
    has_drawings = False

    # 1. Check for Tables
    try:
        tables = page.find_tables()
        if tables and len(tables.tables) > 0:
            has_tables = True
    except Exception:
        has_tables = False

    # 2. Check for Non-Watermark Images
    try:
        images = page.get_images()
        for img in images:
            xref = img[0]
            if xref in watermark_xrefs:
                continue
            pixel_w = img[2]
            pixel_h = img[3]
            if pixel_w >= min_image_dim and pixel_h >= min_image_dim:
                has_images = True
                break
    except Exception:
        has_images = False

    # 3. Check for Large Vector Drawings (charts, flowcharts, geometry)
    try:
        drawings = page.get_drawings()
        for d in drawings:
            rect = d.get("rect")
            if rect:
                w = rect.width
                h = rect.height
                # Ignore 1px lines, dividers, and tiny icons
                if w >= min_vector_dim and h >= min_vector_dim:
                    items = d.get("items", [])
                    # Ignore single colored background boxes/banners behind headings
                    if len(items) <= 1 and items and items[0][0] == "re":
                        continue
                    has_drawings = True
                    break
    except Exception:
        has_drawings = False

    # Visual routing: only include raster images if include_images is True
    is_visual = has_tables or has_drawings or (has_images if include_images else False)

    return {
        "is_visual": is_visual,
        "has_tables": has_tables,
        "has_images": has_images,
        "has_drawings": has_drawings,
    }


# ==============================================================================
# Unicode Normalization
# ==============================================================================


def clean_text(text: str) -> str:
    """Applies Unicode normalization (NFKD) and cleans ghost spaces & soft hyphens."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = text.replace("\xa0", " ").replace("\xad", "")
    return text.strip()


# ==============================================================================
# Extract Book page Number
# ==============================================================================


def extract_printed_page_number(page: pymupdf.Page) -> int | None:
    """
    Extracts the printed book page number from the page:
    1. Checks PDF page labels (page.get_label()).
    2. Scans footer blocks (bottom 65pt) and header blocks (top 65pt).
    3. Matches pure numbers ('8', '- 6 -', '— 12 —'), 'For free distribution 8',
       '6   For free distribution', 'Page 8', and short header/footer lines.
    """
    # 1. Check built-in page label
    label = page.get_label()
    if label and label.strip().isdigit():
        return int(label.strip())

    # 2. Check text blocks in footer and header margins
    try:
        blocks = page.get_text("blocks")
        if not blocks:
            return None

        p_height = page.rect.height
        # Collect blocks located in the footer margin (bottom 65pt)
        footer_blocks = [b for b in blocks if b[3] >= p_height - 65]
        # Collect header blocks if footer has nothing (top 65pt)
        header_blocks = [b for b in blocks if b[1] <= 65]

        # Prioritize footer blocks (where curriculum books put page numbers)
        for block_group in [footer_blocks, header_blocks]:
            for b in block_group:
                text = b[4].strip()
                lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
                for line in lines:
                    # Case A: Pure number or hyphenated: '8', '- 6 -', '— 12 —'
                    m = re.search(r"^[-–—\s]*(\d+)[-–—\s]*$", line)
                    if m and 0 < int(m.group(1)) < 2000:
                        return int(m.group(1))

                    # Case B: Prefixed: 'For free distribution 8', 'Page 8', 'Pg. 8'
                    m = re.search(
                        r"(?:for free distribution|page|pg\.?)\s*(\d+)\b",
                        line,
                        re.IGNORECASE,
                    )
                    if m and 0 < int(m.group(1)) < 2000:
                        return int(m.group(1))

                    # Case C: Suffixed: '6   For free distribution', '6 Page'
                    m = re.search(
                        r"^\s*(\d+)\s*(?:for free distribution|page|pg\.?)\b",
                        line,
                        re.IGNORECASE,
                    )
                    if m and 0 < int(m.group(1)) < 2000:
                        return int(m.group(1))

                    # Case D: Short line (<= 35 chars) ending with a number: e.g. 'Science 8'
                    if len(line) <= 35:
                        m = re.search(r"(?:^|\s)(\d+)\s*$", line)
                        if m and 0 < int(m.group(1)) < 2000:
                            return int(m.group(1))
    except Exception:
        pass

    return None


# ==============================================================================
# Conver text only pages to Markdown
# ==============================================================================


def extract_digital_page_markdown(doc: pymupdf.Document, page: pymupdf.Page) -> str:
    """
    Extracts text-only pages cleanly to Markdown using pymupdf4llm,
    with defensive fallback to page.get_text('blocks').
    """
    try:
        md = pymupdf4llm.to_markdown(doc, pages=[page.number])
        if md and md.strip():
            return clean_text(md)
    except Exception:
        pass

    # Fallback to standard blocks
    try:
        blocks = page.get_text("blocks")
        text = "\n\n".join(b[4].strip() for b in blocks if b[4].strip())
        return clean_text(text)
    except Exception:
        return ""


# ==============================================================================
# Map pages to chapter with TOC
# ==============================================================================


def map_page_to_chapter(
    book_page: int | None,
    pdf_page: int,
    toc: list[dict],
    page_offset: int = 0,
) -> tuple[int | None, str | None]:
    """
    Maps a page to its corresponding chapter from the Table of Contents.
    Uses book_page if available; otherwise uses (pdf_page - page_offset).
    Returns (None, None) if the page is before Chapter 1 or unmapped.
    """
    if not toc:
        return None, None

    if book_page is not None:
        effective_page = book_page
    else:
        effective_page = pdf_page - page_offset
        if effective_page < 1:
            return None, None

    for item in toc:
        start_p = item.get("start_page", 0)
        end_p = item.get("end_page", 999999)
        if start_p <= effective_page <= end_p:
            return item.get("chapter_number"), item.get("title")

    return None, None


# ==============================================================================
# Create & Delete Checkpoint path
# ==============================================================================

CACHE_DIR = ROOT_DIR / "data" / "cache"


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
            print(
                f"[CACHE] Ingestion complete. Cleaned up checkpoint: {cache_path.name}"
            )
    except Exception as e:
        print(f"[WARNING] Could not delete checkpoint {cache_path}: {e}")


# ==============================================================================
# Create Extracted file path
# ==============================================================================

EXTRACTED_DIR = ROOT_DIR / "data" / "extracted"


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
    print(
        f"\n[DATASET] Saved permanent textbook dataset ({len(pages)} pages) to: {out_path}"
    )
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
