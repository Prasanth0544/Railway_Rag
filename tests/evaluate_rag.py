"""
evaluate_rag.py — Automated RAG Quality Evaluation

Measures:
  1. Intent Accuracy: Does the classifier get the right intent?
  2. Retrieval Recall: Are relevant docs retrieved?
  3. Answer Keyword Hit Rate: Does the answer contain expected keywords?
  4. Hallucination Detection: Does the answer contain train numbers NOT in context?

Usage:
  python tests/evaluate_rag.py
  python tests/evaluate_rag.py --dataset tests/test_queries.json --verbose
"""

import json
import os
import re
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))


def load_test_dataset(path: str = "tests/test_queries.json") -> list[dict]:
    """Load the test query dataset."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_intent(test_cases: list[dict], verbose: bool = False) -> dict:
    """Evaluate intent classifier accuracy."""
    from app.intent import classify_intent

    correct = 0
    total = 0
    failures = []

    for tc in test_cases:
        question = tc["question"]
        expected_intent = tc["intent"]
        result = classify_intent(question)
        actual_intent = result["intent"]
        total += 1

        if actual_intent == expected_intent:
            correct += 1
            if verbose:
                print(f"  ✅ [{expected_intent}] {question}")
        else:
            failures.append({
                "question": question,
                "expected": expected_intent,
                "actual": actual_intent,
                "reasons": result.get("reasons", []),
            })
            if verbose:
                print(f"  ❌ [{expected_intent}→{actual_intent}] {question}")
                print(f"      Reasons: {result.get('reasons', [])}")

    accuracy = correct / total if total > 0 else 0
    return {
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "total": total,
        "failures": failures,
    }


def evaluate_retrieval(test_cases: list[dict], verbose: bool = False) -> dict:
    """Evaluate retrieval quality — checks if retrieved docs contain expected keywords."""
    from app.retriever import get_unified_retriever

    retriever = get_unified_retriever(top_k=10)

    total = 0
    hit_count = 0
    keyword_hits = 0
    keyword_total = 0
    details = []

    # Only test STATIC/HYBRID queries (not LIVE or OUT_OF_DOMAIN)
    static_cases = [tc for tc in test_cases if tc["intent"] in ("STATIC", "HYBRID")]

    for tc in static_cases:
        question = tc["question"]
        expected_kws = tc.get("expected_keywords", [])
        if not expected_kws:
            continue

        total += 1
        docs = retriever.retrieve(question)
        combined_text = " ".join(d.page_content for d in docs).lower()

        # Check how many expected keywords appear in retrieved context
        hits = [kw for kw in expected_kws if kw.lower() in combined_text]
        misses = [kw for kw in expected_kws if kw.lower() not in combined_text]

        keyword_hits += len(hits)
        keyword_total += len(expected_kws)

        if len(hits) >= len(expected_kws) * 0.5:  # At least 50% keyword hit
            hit_count += 1

        if verbose:
            status = "✅" if len(hits) >= len(expected_kws) * 0.5 else "❌"
            print(f"  {status} [{len(hits)}/{len(expected_kws)} kw] {question} ({len(docs)} docs)")
            if misses:
                print(f"      Missing: {misses}")

        details.append({
            "question": question,
            "docs_retrieved": len(docs),
            "keyword_hits": len(hits),
            "keyword_total": len(expected_kws),
            "missing_keywords": misses,
        })

    recall = hit_count / total if total > 0 else 0
    kw_rate = keyword_hits / keyword_total if keyword_total > 0 else 0

    return {
        "retrieval_recall_50pct": round(recall, 4),
        "keyword_hit_rate": round(kw_rate, 4),
        "queries_tested": total,
        "queries_passed": hit_count,
        "details": details,
    }


def evaluate_hallucination(test_cases: list[dict], verbose: bool = False) -> dict:
    """Check if the LLM generates train numbers not present in the retrieved context."""
    from app.retriever import get_unified_retriever
    from app.rag import get_rag_chain

    rag_chain = get_rag_chain()
    retriever = get_unified_retriever(top_k=10)

    total = 0
    hallucination_count = 0
    details = []

    # Only test a small subset to save API calls
    sample = [tc for tc in test_cases if tc["intent"] == "STATIC"][:5]

    for tc in sample:
        question = tc["question"]
        total += 1

        docs = retriever.retrieve(question)
        context = " ".join(d.page_content for d in docs)
        result = rag_chain.invoke(question)
        answer = result["answer"]

        # Find train numbers in answer vs context
        answer_trains = set(re.findall(r'\b(\d{5})\b', answer))
        context_trains = set(re.findall(r'\b(\d{5})\b', context))

        hallucinated = answer_trains - context_trains
        if hallucinated:
            hallucination_count += 1
            if verbose:
                print(f"  ⚠️  HALLUCINATION: {question}")
                print(f"      Fabricated trains: {hallucinated}")
        else:
            if verbose:
                print(f"  ✅ No hallucination: {question}")

        details.append({
            "question": question,
            "answer_trains": list(answer_trains),
            "context_trains": list(context_trains),
            "hallucinated": list(hallucinated),
        })

    rate = hallucination_count / total if total > 0 else 0
    return {
        "hallucination_rate": round(rate, 4),
        "hallucinations_found": hallucination_count,
        "queries_tested": total,
        "details": details,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate RAG quality")
    parser.add_argument("--dataset", default="tests/test_queries.json", help="Path to test queries JSON")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed results")
    parser.add_argument("--skip-hallucination", action="store_true", help="Skip hallucination test (saves API calls)")
    args = parser.parse_args()

    test_cases = load_test_dataset(args.dataset)
    print(f"\n{'='*60}")
    print(f"  RailGPT Quality Evaluation — {len(test_cases)} test queries")
    print(f"{'='*60}\n")

    # 1. Intent Accuracy
    print("📋 1. Intent Classification Accuracy")
    print("-" * 40)
    t0 = time.time()
    intent_results = evaluate_intent(test_cases, verbose=args.verbose)
    print(f"\n   Accuracy: {intent_results['accuracy']*100:.1f}% ({intent_results['correct']}/{intent_results['total']})")
    if intent_results['failures']:
        print(f"   ❌ Failures: {len(intent_results['failures'])}")
    print(f"   ⏱  Time: {time.time()-t0:.1f}s\n")

    # 2. Retrieval Recall
    print("🔍 2. Retrieval Quality (Keyword Recall)")
    print("-" * 40)
    t0 = time.time()
    retrieval_results = evaluate_retrieval(test_cases, verbose=args.verbose)
    print(f"\n   Retrieval Recall (≥50% keywords): {retrieval_results['retrieval_recall_50pct']*100:.1f}%")
    print(f"   Overall Keyword Hit Rate: {retrieval_results['keyword_hit_rate']*100:.1f}%")
    print(f"   Queries: {retrieval_results['queries_passed']}/{retrieval_results['queries_tested']} passed")
    print(f"   ⏱  Time: {time.time()-t0:.1f}s\n")

    # 3. Hallucination Detection
    if not args.skip_hallucination:
        print("🤥 3. Hallucination Detection")
        print("-" * 40)
        t0 = time.time()
        hallucination_results = evaluate_hallucination(test_cases, verbose=args.verbose)
        print(f"\n   Hallucination Rate: {hallucination_results['hallucination_rate']*100:.1f}%")
        print(f"   Hallucinations: {hallucination_results['hallucinations_found']}/{hallucination_results['queries_tested']}")
        print(f"   ⏱  Time: {time.time()-t0:.1f}s\n")
    else:
        print("🤥 3. Hallucination Detection — SKIPPED (use --no-skip-hallucination)\n")

    # Summary
    print(f"{'='*60}")
    print("  📊 SUMMARY")
    print(f"{'='*60}")
    print(f"  Intent Accuracy:      {intent_results['accuracy']*100:.1f}%")
    print(f"  Retrieval Recall:     {retrieval_results['retrieval_recall_50pct']*100:.1f}%")
    print(f"  Keyword Hit Rate:     {retrieval_results['keyword_hit_rate']*100:.1f}%")
    if not args.skip_hallucination:
        print(f"  Hallucination Rate:   {hallucination_results['hallucination_rate']*100:.1f}%")
    print(f"{'='*60}\n")

    # Save results to JSON
    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "num_test_cases": len(test_cases),
        "intent": intent_results,
        "retrieval": {k: v for k, v in retrieval_results.items() if k != "details"},
    }
    if not args.skip_hallucination:
        output["hallucination"] = {k: v for k, v in hallucination_results.items() if k != "details"}

    os.makedirs("tests", exist_ok=True)
    with open("tests/eval_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("  💾 Results saved to tests/eval_results.json\n")


if __name__ == "__main__":
    main()
