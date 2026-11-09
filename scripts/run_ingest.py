import argparse
import json
import sys
from pathlib import Path

import httpx

from rag.ingest import (
    get_extracted_path,
    index_pages_to_chroma,
    parse_and_prepare_textbook,
)


def main():
    parser = argparse.ArgumentParser(
        description="TextbookMind Local Ingestion & Dataset Extraction CLI"
    )
    parser.add_argument(
        "--file",
        "-f",
        default="data/history G-11 E.pdf",
        help="Path to the PDF textbook file (default: data/history G-11 E.pdf)",
    )
    parser.add_argument(
        "--grade",
        "-g",
        type=int,
        default=11,
        help="Grade level of the textbook (default: 11)",
    )
    parser.add_argument(
        "--subject",
        "-s",
        default="history",
        help="Subject of the textbook (default: history)",
    )
    parser.add_argument(
        "--medium",
        "-m",
        default="English",
        help="Medium of instruction (default: english)",
    )
    parser.add_argument(
        "--edition",
        "-e",
        default="2014",
        help="Edition or syllabus year (default: 2014)",
    )
    parser.add_argument(
        "--auto-ingest",
        action="store_true",
        help="Skip interactive confirmation gate and proceed automatically",
    )
    parser.add_argument(
        "--include-images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to route pages with embedded raster images to Gemini Vision (default: True). If False, only pages with tables or vector drawings are sent to Gemini.",
    )
    parser.add_argument(
        "--index",
        action="store_true",
        help="Index the extracted pages directly into the local ChromaDB vector store",
    )
    parser.add_argument(
        "--toc-extra-pages",
        type=int,
        default=2,
        help="Extra pages after Contents header to send to Gemini Vision for TOC parsing (default: 2)",
    )
    parser.add_argument(
        "--force-reextract",
        action="store_true",
        help="Ignore existing extracted dataset and re-extract from PDF",
    )
    parser.add_argument(
        "--publish-to-api",
        help="Optional API URL (e.g. http://127.0.0.1:8000/textbook/dataset) to upload the dataset once extracted",
    )

    args = parser.parse_args()

    pdf_file = Path(args.file)
    if not pdf_file.exists():
        print(f"[ERROR] File not found: {args.file}")
        sys.exit(1)

    if pdf_file.suffix.lower() == ".json":
        print(f"\n[CLI] Loading existing JSON dataset: {pdf_file.name}")
        with open(pdf_file, encoding="utf-8") as f:
            pages = json.load(f)
        extracted_path = str(pdf_file.resolve())
    else:
        print(f"\n[CLI] Starting local extraction for: {pdf_file.name}")
        print(
            f"      Grade: {args.grade} | Subject: {args.subject} | Medium: {args.medium} | Edition: {args.edition}"
        )
        print(f"      Include Images: {args.include_images}\n")

        pages = parse_and_prepare_textbook(
            file_path=str(pdf_file),
            grade=args.grade,
            subject=args.subject,
            medium=args.medium,
            edition=args.edition,
            auto_ingest=args.auto_ingest,
            include_images=args.include_images,
            toc_extra_pages=args.toc_extra_pages,
            force_reextract=args.force_reextract,
        )
        extracted_path = get_extracted_path(str(pdf_file))

    if not pages:
        print("[INFO] No pages were extracted.")
        return

    print(f"\n[CLI SUCCESS] Extracted {len(pages)} pages.")
    print(f"              Saved dataset: {extracted_path}")

    # Optional: Index dataset directly into local ChromaDB
    if args.index:
        print("\n[CLI INDEX] Indexing dataset into local ChromaDB...")
        collection_name, total_indexed = index_pages_to_chroma(
            pages=pages,
            grade=args.grade,
            subject=args.subject,
            medium=args.medium,
            edition=args.edition,
            fallback_stem=pdf_file.stem.replace(" ", "-"),
        )
        print(
            f"[CLI INDEX SUCCESS] Indexed {total_indexed} pages into collection '{collection_name}'."
        )

    # Optional: Publish dataset to cloud/backend API
    if args.publish_to_api:
        api_url = args.publish_to_api
        print(f"\n[CLI PUBLISH] Publishing dataset to API: {api_url} ...")
        payload = {
            "title": f"Grade {args.grade} {args.subject.title()} ({args.medium.title()})",
            "grade": args.grade,
            "subject": args.subject,
            "medium": args.medium,
            "edition": args.edition,
            "pages": pages,
        }
        try:
            with httpx.Client(timeout=60.0) as client:
                res = client.post(api_url, json=payload)
                if res.status_code in (200, 201):
                    print(
                        f"[CLI PUBLISH SUCCESS] Ingested into backend! Response: {res.json()}"
                    )
                else:
                    print(
                        f"[CLI PUBLISH ERROR] Server returned HTTP {res.status_code}: {res.text}"
                    )
        except Exception as e:
            print(f"[CLI PUBLISH FAILED] Could not connect to API: {e}")


if __name__ == "__main__":
    main()
