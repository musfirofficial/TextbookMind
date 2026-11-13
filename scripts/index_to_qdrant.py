import argparse
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
ROOT_PATH = Path(__file__).resolve().parents[1]
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

from rag import (
    DEFAULT_COLLECTION_NAME,
    ROOT_DIR,
    create_chunks_from_dataset,
    get_qdrant_client,
    init_textbook_collection,
    upsert_chunks,
)

EXTRACTED_DIR = ROOT_DIR / "data" / "extracted"


def resolve_dataset_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.exists():
        return p
    # Try looking directly in data/extracted/
    p_in_extracted = EXTRACTED_DIR / path_str
    if p_in_extracted.exists():
        return p_in_extracted
    if not path_str.endswith(".json"):
        p_with_json = EXTRACTED_DIR / f"{path_str}.json"
        if p_with_json.exists():
            return p_with_json
    raise FileNotFoundError(f"Extracted dataset not found: '{path_str}'")


def main():
    parser = argparse.ArgumentParser(
        description="Textbook Ingestion Step 2: Embed (Dense + BM25) and store chunks into persistent Qdrant."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--dataset",
        "-d",
        type=str,
        help="Path or name of extracted dataset JSON (e.g. 'SCIENCE_G-10_P1_E.json').",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Embed and index all extracted dataset JSONs found in data/extracted/.",
    )

    parser.add_argument(
        "--collection",
        "-c",
        type=str,
        default=DEFAULT_COLLECTION_NAME,
        help=f"Qdrant collection name (default: '{DEFAULT_COLLECTION_NAME}').",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=64,
        help="Embedding batch size for FastEmbed (default: 64).",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate/clear collection before indexing (WARNING: removes existing points in this collection).",
    )

    args = parser.parse_args()

    # 1. Resolve which datasets to process
    target_files: list[Path] = []
    if args.all:
        target_files = sorted(EXTRACTED_DIR.glob("*.json"))
        if not target_files:
            print(f"[ERROR] No extracted dataset JSONs found in: {EXTRACTED_DIR}")
            sys.exit(1)
    else:
        try:
            target_files = [resolve_dataset_path(args.dataset)]
        except FileNotFoundError as err:
            print(f"[ERROR] {err}")
            sys.exit(1)

    print("\n" + "=" * 70)
    print("  QDRANT DUAL-VECTOR INDEXER (DENSE BGE-SMALL + SPARSE BM25)")
    print("=" * 70)
    print(f"Target Collection : '{args.collection}'")
    print(f"Total Datasets    : {len(target_files)}")
    print(f"Embedding Batch   : {args.batch_size}")
    if args.recreate:
        print("Collection Mode   : RECREATE (Wiping old collection)")
    print("-" * 70)

    # 2. Initialize persistent Qdrant client
    client = get_qdrant_client()
    init_textbook_collection(
        client, collection_name=args.collection, recreate=args.recreate
    )

    total_chunks_indexed = 0
    t_start_all = time.time()

    for idx, json_file in enumerate(target_files, 1):
        print(f"\n[{idx}/{len(target_files)}] Processing: {json_file.name}")
        t0 = time.time()

        # Chunk the dataset
        chunks = create_chunks_from_dataset(str(json_file))
        if not chunks:
            print(f"  [SKIP] No chunks produced for {json_file.name}")
            continue

        # Embed and upsert into Qdrant
        count = upsert_chunks(
            chunks=chunks,
            client=client,
            collection_name=args.collection,
            batch_size=args.batch_size,
        )
        total_chunks_indexed += count
        print(f"  -> Indexed {count} chunks in {time.time() - t0:.1f}s")

    col_info = client.get_collection(args.collection)
    total_in_col = col_info.points_count

    print("\n" + "=" * 70)
    print("✅ INDEXING COMPLETE")
    print(f"   • Datasets Processed  : {len(target_files)}")
    print(f"   • New Chunks Indexed  : {total_chunks_indexed}")
    print(f"   • Total Points in DB  : {total_in_col}")
    print(f"   • Elapsed Time        : {time.time() - t_start_all:.1f}s")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
