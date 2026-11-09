# Standard Run (Page mistakes won't fail tests) - uv run python eval/run_harness.py
# Run with a Limit (e.g. first 20 questions) - uv run python eval/run_harness.py --limit 20
# Strict Mode (Enforce both page matching & keywords) - uv run python eval/run_harness.py --check-pages

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# Ensure src is in python path
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rag.engine import load_chat_engine

TEST_CASES_FILE = Path(__file__).resolve().parent / "test_case.json"

# ANSI Terminal Colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

# Phrases indicating proper refusal for unanswerable questions
REFUSAL_PHRASES = [
    "not enough information",
    "there is not enough information",
    "not provided",
    "not mentioned",
    "does not contain",
    "cannot answer",
    "no information",
    "insufficient information",
    "excerpt does not",
    "textbook does not",
]


def check_refusal(text: str) -> bool:
    lower = text.lower()
    return any(p in lower for p in REFUSAL_PHRASES)


def evaluate_case(
    engine: Any,
    tc: dict[str, Any],
    check_pages: bool = False,
    min_kw_ratio: float = 0.4,
) -> dict[str, Any]:
    tc_id = tc["id"]
    question = tc["question"]
    answerable = tc.get("answerable", True)
    expected_keywords = tc.get("expected_keywords", [])
    expected_pages = tc.get("expected_pages", [])

    # Measure latency & query engine (with automatic retry for rate limits)
    max_retries = 3
    response_text = ""
    response = None
    latency = 0.0

    for attempt in range(max_retries):
        start = time.perf_counter()
        try:
            response = engine.chat(question)
            latency = time.perf_counter() - start
            response_text = str(response).strip()
            break
        except Exception as e:
            latency = time.perf_counter() - start
            err_msg = str(e)
            # If hit by Groq rate limit / 429, wait and retry
            if (
                "rate_limit" in err_msg.lower()
                or "429" in err_msg
                or "too many requests" in err_msg.lower()
            ) and attempt < max_retries - 1:
                wait_sec = 3 * (attempt + 1)
                time.sleep(wait_sec)
                continue
            return {
                "id": tc_id,
                "question": question,
                "passed": False,
                "error": err_msg,
                "latency": latency,
            }

    # Extract retrieved pages from source nodes
    retrieved_pages = []
    if (
        response is not None
        and hasattr(response, "source_nodes")
        and response.source_nodes
    ):
        for node in response.source_nodes:
            p = node.metadata.get("page_number")
            if p is not None and str(p).isdigit() and int(p) not in retrieved_pages:
                retrieved_pages.append(int(p))

    # Evaluate Unanswerable questions
    if not answerable:
        refused = check_refusal(response_text)
        return {
            "id": tc_id,
            "question": question,
            "answerable": False,
            "passed": refused,
            "refused": refused,
            "retrieved_pages": retrieved_pages,
            "response": response_text[:200],
            "latency": latency,
        }

    # Evaluate Answerable questions
    exp_set = {int(p) for p in expected_pages if str(p).isdigit()}
    ret_set = set(retrieved_pages)
    page_match = bool(exp_set & ret_set) if exp_set else True

    response_lower = response_text.lower()
    matched_kw = [kw for kw in expected_keywords if kw.lower() in response_lower]
    kw_ratio = len(matched_kw) / len(expected_keywords) if expected_keywords else 1.0

    # Determine PASS:
    # If check_pages is True, both page match AND keywords must pass.
    # If check_pages is False (default), pass is based purely on keywords.
    kw_passed = kw_ratio >= min_kw_ratio
    passed = (page_match and kw_passed) if check_pages else kw_passed

    return {
        "id": tc_id,
        "question": question,
        "answerable": True,
        "passed": passed,
        "page_match": page_match,
        "expected_pages": expected_pages,
        "retrieved_pages": retrieved_pages,
        "matched_pages": list(exp_set & ret_set),
        "matched_keywords": matched_kw,
        "total_keywords": len(expected_keywords),
        "kw_ratio": kw_ratio,
        "response": response_text[:200],
        "latency": latency,
    }


def main():
    parser = argparse.ArgumentParser(description="Textbook RAG Evaluation Harness")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Number of test cases to run (default: 0 = run all tests).",
    )
    parser.add_argument(
        "--check-pages",
        action="store_true",
        default=False,
        help="Enforce page matching for tests to pass (default: False, ignores JSON page errors).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay in seconds between questions to prevent API rate limits (default: 0.5s).",
    )
    args = parser.parse_args()

    if not TEST_CASES_FILE.exists():
        print(f"{RED}Error: {TEST_CASES_FILE} not found.{RESET}")
        return

    with open(TEST_CASES_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Flatten all test cases into a single list
    tests = []
    for group in data:
        collection = group.get("collection", "Full")
        for tc in group.get("test_cases", []):
            tc["_collection"] = collection
            tests.append(tc)

    if args.limit and args.limit > 0:
        tests = tests[: args.limit]

    total = len(tests)
    print(f"\n{BOLD}{CYAN}=== Textbook RAG Test Runner ==={RESET}")
    print(f"Total Tests:   {BOLD}{total}{RESET}")
    page_mode_text = (
        f"{YELLOW}ENFORCED (tests fail if pages don't match){RESET}"
        if args.check_pages
        else f"{DIM}INFORMATIONAL ONLY (JSON page errors won't fail tests){RESET}"
    )
    print(f"Page Check:    {page_mode_text}\n")

    current_engine = None
    current_collection = None
    results = []
    total_start = time.perf_counter()

    for idx, tc in enumerate(tests, start=1):
        collection = tc["_collection"]

        # Load/switch engine if needed
        if current_engine is None or current_collection != collection:
            current_engine = load_chat_engine(collection_name=collection)
            current_collection = collection

        # Reset chat memory for independent test evaluation
        current_engine.reset()

        # Run test
        res = evaluate_case(current_engine, tc, check_pages=args.check_pages)
        results.append(res)

        badge = f"{GREEN}[PASS]{RESET}" if res["passed"] else f"{RED}[FAIL]{RESET}"
        latency = res.get("latency", 0.0)
        print(
            f"[{idx}/{total}] {badge} {BOLD}{res['id']}{RESET} ({latency:.2f}s) - {res['question']}"
        )

        # Handle API / execution errors safely
        if "error" in res:
            print(f"      {RED}API Error:{RESET} {res['error']}")
        elif not res.get("answerable", True):
            note = (
                f"{GREEN}Refused correctly{RESET}"
                if res.get("refused", False)
                else f"{RED}Hallucinated (did not refuse){RESET}"
            )
            print(f"      {DIM}Anti-Hallucination:{RESET} {note}")
        else:
            page_match = res.get("page_match", False)
            p_color = GREEN if page_match else (YELLOW if not args.check_pages else RED)
            p_text = f"{p_color}Pages: {res.get('retrieved_pages', [])} (Expected: {res.get('expected_pages', [])}){RESET}"
            kw_count = len(res.get("matched_keywords", []))
            kw_total = res.get("total_keywords", 0)
            kw_ratio = res.get("kw_ratio", 0.0)
            kw_text = f"Keywords: {kw_count}/{kw_total} ({kw_ratio * 100:.0f}%)"
            print(f"      {p_text} | {kw_text}")

        if not res["passed"] and "error" not in res:
            print(
                f"      {DIM}Answer preview:{RESET} {res.get('response', '')[:120]}..."
            )

        print()

        # Small delay between questions to avoid hitting Groq rate limits
        if args.delay > 0 and idx < total:
            time.sleep(args.delay)

    duration = time.perf_counter() - total_start
    passed_count = sum(1 for r in results if r["passed"])
    error_count = sum(1 for r in results if "error" in r)

    # Metrics
    answerable_res = [
        r for r in results if r.get("answerable", True) and "error" not in r
    ]
    page_match_rate = (
        (
            sum(1 for r in answerable_res if r.get("page_match", False))
            / len(answerable_res)
            * 100
        )
        if answerable_res
        else 0.0
    )
    avg_latency = sum(r.get("latency", 0.0) for r in results) / total if total else 0.0

    print("=" * 55)
    score_color = GREEN if passed_count == total else YELLOW
    print(
        f"{BOLD}Scorecard: {score_color}{passed_count}/{total} Passed ({(passed_count / total) * 100:.1f}%){RESET}"
    )
    print(f"Page Retrieval Accuracy: {page_match_rate:.1f}%")
    if error_count > 0:
        print(f"{RED}API Errors / Timeouts:   {error_count}{RESET}")
    print(f"Average Latency:         {avg_latency:.2f}s per query")
    print(f"Total Time:              {duration:.1f}s")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
