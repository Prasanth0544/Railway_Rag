from app.retriever import get_unified_retriever

r = get_unified_retriever()

queries = [
    "Which trains run between Vijayawada and Hyderabad?",
    "Stops of train 12727",
    "Cancellation charges for sleeper class",
]

for q in queries:
    docs = r.retrieve(q)
    print(f"\nQuery: {repr(q)}")
    print(f"  -> {len(docs)} docs retrieved")
    for d in docs[:3]:
        col = d.metadata.get("collection", "?")
        sc = d.metadata.get("relevance_score", 0)
        print(f"     [{col} score={sc}] {d.page_content[:90]}")
