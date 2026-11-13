import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
ROOT_PATH = Path(__file__).resolve().parents[1]
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

from qdrant_client import models

from src.rag.core.config import DEFAULT_COLLECTION_NAME, ROOT_DIR
from src.rag.vector_store import (
    get_qdrant_client,
    hybrid_search,
)

BENCHMARK_FILE = ROOT_DIR / "eval" / "benchmark_queries.json"
RESULTS_FILE = ROOT_DIR / "eval" / "benchmark_results.json"


def normalize_page_list(pages: Any) -> list[int]:
    """Converts a single int or list of ints/strings into a clean list of ints."""
    if pages is None:
        return []
    if isinstance(pages, int):
        return [pages]
    if isinstance(pages, (list, tuple)):
        result = []
        for p in pages:
            try:
                result.append(int(p))
            except (ValueError, TypeError):
                continue
        return result
    try:
        return [int(pages)]
    except (ValueError, TypeError):
        return []


def run_evaluation(collection_name: str = DEFAULT_COLLECTION_NAME, top_k: int = 5):
    print("\n" + "=" * 75)
    print("      TEXTBOOKMIND RETRIEVAL BENCHMARK EVALUATOR (HIT RATE & MRR)")
    print("=" * 75)

    if not BENCHMARK_FILE.exists():
        print(f"[ERROR] Benchmark file not found at: {BENCHMARK_FILE}")
        return

    with open(BENCHMARK_FILE, "r", encoding="utf-8") as f:
        all_cases: list[dict] = json.load(f)

    if not all_cases:
        print(f"[ERROR] Benchmark file '{BENCHMARK_FILE.name}' is empty.")
        return

    # 1. Connect to persistent Qdrant
    client = get_qdrant_client()

    if not client.collection_exists(collection_name):
        print(f"\n[ERROR] Qdrant collection '{collection_name}' does not exist.")
        print("Please embed and store your textbooks first by running:")
        print("   python scripts/index_to_qdrant.py --all")
        print("   (or: python scripts/index_to_qdrant.py --dataset <filename>)\n")
        return

    col_info = client.get_collection(collection_name)
    if col_info.points_count == 0:
        print(f"\n[ERROR] Collection '{collection_name}' has 0 points.")
        print("Please index textbooks using 'python scripts/index_to_qdrant.py --all'\n")
        return

    print(f"Connected to Qdrant Collection: '{collection_name}' ({col_info.points_count} points in DB)")

    # 2. Check which documents exist in Qdrant via chunk_id stems
    all_docs = sorted(list(set(c.get("document_name", "").strip() for c in all_cases if c.get("document_name"))))
    
    # Fast discovery of active document stems from Qdrant chunk_ids
    sample_points, _ = client.scroll(
        collection_name=collection_name,
        limit=10000,
        with_payload=["chunk_id", "document_name"],
        with_vectors=False,
    )
    indexed_docs = set(
        p.payload["chunk_id"].rsplit("_p", 1)[0]
        for p in sample_points
        if "_p" in p.payload.get("chunk_id", "")
    )
    missing_docs = set(d for d in all_docs if d not in indexed_docs)

    print(f"\n[DB STATUS] Active Documents in Qdrant ({len(indexed_docs)}):")
    for d in sorted(indexed_docs):
        print(f"  ✅ {d}")

    if missing_docs:
        print(f"\n[NOTICE] {len(missing_docs)} Documents NOT yet indexed in Qdrant (Skipping their queries):")
        for d in sorted(missing_docs):
            print(f"  ⏳ {d}  (To index: python scripts/index_to_qdrant.py --dataset {d}.json)")

    # 3. Filter test queries to only active documents
    active_cases = [
        c for c in all_cases
        if Path(c.get("document_name", "")).stem in indexed_docs
    ]

    if not active_cases:
        print("\n[ERROR] None of the documents in benchmark_queries.json are indexed in Qdrant.")
        print("Please run: python scripts/index_to_qdrant.py --all\n")
        return

    total_active = len(active_cases)
    total_in_file = len(all_cases)
    print(f"\n[EVALUATION SCOPE] Running {total_active} active queries out of {total_in_file} in benchmark.")
    print(f"Scoring will be calculated out of {total_active} (100% = {total_active}/{total_active}).")
    print("-" * 75)

    # 4. Run Evaluation Queries
    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_5 = 0
    rr_scores = []
    latencies = []
    failures = []

    for idx, item in enumerate(active_cases, 1):
        query = item.get("query", "").strip()
        expected_doc = Path(item.get("document_name", "")).stem
        expected_kw = str(item.get("expected_keyword", "")).strip().lower()
        expected_ch = item.get("expected_chapter")
        expected_pages = normalize_page_list(item.get("expected_book_page"))
        query_filter = item.get("filter")

        t0 = time.time()
        results = hybrid_search(
            query=query,
            client=client,
            collection_name=collection_name,
            top_k=top_k,
            filter_conditions=query_filter,
        )
        latency = (time.time() - t0) * 1000.0
        latencies.append(latency)

        hit_rank = None
        for rank, point in enumerate(results, 1):
            p_payload = point.payload
            p_chunk_id = p_payload.get("chunk_id", "")
            p_doc = p_chunk_id.rsplit("_p", 1)[0] if "_p" in p_chunk_id else Path(p_payload.get("document_name", "")).stem
            p_ch = p_payload.get("chapter_number")
            p_page = p_payload.get("book_page")
            p_text = p_payload.get("text", "").lower()

            doc_ok = (not expected_doc) or (p_doc == expected_doc)
            ch_ok = (expected_ch is None) or (p_ch == expected_ch)
            page_ok = (not expected_pages) or (p_page in expected_pages)
            kw_ok = (not expected_kw) or (expected_kw in p_text)

            if doc_ok and ch_ok and page_ok and kw_ok:
                hit_rank = rank
                break

        # Tally metrics
        if hit_rank == 1:
            hits_at_1 += 1
        if hit_rank is not None and hit_rank <= 3:
            hits_at_3 += 1
        if hit_rank is not None and hit_rank <= 5:
            hits_at_5 += 1

        reciprocal_rank = 1.0 / hit_rank if hit_rank is not None else 0.0
        rr_scores.append(reciprocal_rank)

        # Status badge
        if hit_rank == 1:
            status = "🥇 Rank 1"
        elif hit_rank in (2, 3):
            status = f"✅ Rank {hit_rank}"
        elif hit_rank in (4, 5):
            status = f"⚠️ Rank {hit_rank}"
        else:
            status = "❌ MISS"
            failures.append({
                "index": idx,
                "query": query,
                "doc": expected_doc,
                "expected_chapter": expected_ch,
                "expected_pages": expected_pages,
                "expected_keyword": expected_kw,
                "retrieved_top_chunk": results[0].payload.get("chunk_id") if results else None,
                "retrieved_text": results[0].payload.get("text", "")[:150] if results else None,
            })

        print(f"[{idx:03d}/{total_active:03d}] {status:10} | {latency:5.1f}ms | \"{query[:50]}\"")

    hit_rate_1 = (hits_at_1 / total_active) * 100.0
    hit_rate_3 = (hits_at_3 / total_active) * 100.0
    hit_rate_5 = (hits_at_5 / total_active) * 100.0
    mrr_5 = sum(rr_scores) / total_active
    avg_latency = sum(latencies) / total_active

    # 5. Display Final Summary
    print("\n" + "=" * 75)
    print("                    FINAL BENCHMARK EVALUATION REPORT")
    print("=" * 75)
    print(f" Active Queries Tested   : {total_active} / {total_in_file}")
    print(f" Documents Evaluated     : {len(indexed_docs)} (of {len(all_docs)} total)")
    print(f" Average Latency         : {avg_latency:.2f} ms")
    print("-" * 75)
    print(f" 🎯 Hit Rate @ 1         : {hit_rate_1:.2f}%  ({hits_at_1}/{total_active})")
    print(f" 🎯 Hit Rate @ 3         : {hit_rate_3:.2f}%  ({hits_at_3}/{total_active})")
    print(f" 🎯 Hit Rate @ 5         : {hit_rate_5:.2f}%  ({hits_at_5}/{total_active})")
    print(f" 📈 Mean Reciprocal Rank : {mrr_5:.4f}")
    print("=" * 75)

    if failures:
        print(f"\n[MISS ANALYSIS] {len(failures)} queries missed target criteria in Top-5:")
        for f_item in failures[:5]:
            print(f"  • Q#{f_item['index']}: \"{f_item['query']}\"")
            print(f"    Target: Doc '{f_item['doc']}', Ch.{f_item['expected_chapter']}, Pages {f_item['expected_pages']}, Kw '{f_item['expected_keyword']}'")
            print(f"    Got   : ID '{f_item['retrieved_top_chunk']}'\n")

    summary_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "active_queries": total_active,
        "total_benchmark_queries": total_in_file,
        "indexed_documents": sorted(list(indexed_docs)),
        "hit_rate_at_1": round(hit_rate_1, 2),
        "hit_rate_at_3": round(hit_rate_3, 2),
        "hit_rate_at_5": round(hit_rate_5, 2),
        "mrr_at_5": round(mrr_5, 4),
        "avg_latency_ms": round(avg_latency, 2),
        "failures_count": len(failures),
    }
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print(f"[REPORT] Saved summary metrics to: {RESULTS_FILE}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Hybrid Search Retrieval with Hit Rate and MRR.")
    parser.add_argument("--collection", "-c", type=str, default=DEFAULT_COLLECTION_NAME, help="Qdrant collection name.")
    parser.add_argument("--top-k", "-k", type=int, default=5, help="Top-K retrieval limit (default: 5).")
    args = parser.parse_args()

    run_evaluation(collection_name=args.collection, top_k=args.top_k)
