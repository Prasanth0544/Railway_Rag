"""
compact_train_routes.py — Merge 2 HNSW segments into 1

When embed_routes.py resumes from a checkpoint mid-collection,
ChromaDB sometimes creates a second HNSW segment file instead of
extending the first. Both segments load into RAM simultaneously,
doubling memory usage.

This script:
  1. Reads ALL documents + embeddings from train_routes (no API calls)
  2. Deletes the collection (removes both segment files)
  3. Re-creates the collection and adds everything back in one shot
     → ChromaDB creates a single merged HNSW segment

Run once after ingestion is complete:
  .venv\Scripts\python scripts/compact_train_routes.py
"""

import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb

CHROMA_DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")
COLLECTION = "train_routes"
BATCH_SIZE = 500  # safe batch size for add() without hitting memory spike


def main():
    print("\n" + "=" * 60)
    print("train_routes HNSW Segment Compactor")
    print("=" * 60)

    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

    # Check collection exists
    existing = {c.name for c in client.list_collections()}
    if COLLECTION not in existing:
        print(f"[ERROR] Collection '{COLLECTION}' not found in {CHROMA_DB_DIR}")
        sys.exit(1)

    col = client.get_collection(COLLECTION)
    total = col.count()
    print(f"[INFO] Found {total:,} documents in '{COLLECTION}'")

    # Count segment files before
    seg_dirs = [d for d in os.listdir(CHROMA_DB_DIR)
                if os.path.isdir(os.path.join(CHROMA_DB_DIR, d))
                and d not in ("__pycache__",)]
    print(f"[INFO] Segment directories before: {len(seg_dirs)}")

    # Step 1: Read all data in batches (no embedding API calls)
    print(f"\n[STEP 1] Reading all {total:,} docs + embeddings from ChromaDB...")
    all_ids        = []
    all_docs       = []
    all_metadatas  = []
    all_embeddings = []

    offset = 0
    while offset < total:
        batch = col.get(
            limit=BATCH_SIZE,
            offset=offset,
            include=["documents", "metadatas", "embeddings"],
        )
        ids   = batch["ids"]
        docs  = batch["documents"]
        metas = batch["metadatas"]
        embs  = batch["embeddings"]

        all_ids.extend(ids)
        all_docs.extend(docs)
        all_metadatas.extend(metas)
        all_embeddings.extend(embs)

        offset += len(ids)
        pct = offset / total * 100
        bar = "#" * int(pct / 5) + "-" * (20 - int(pct / 5))
        print(f"  [{bar}] {offset:,}/{total:,} ({pct:.0f}%)", end="\r", flush=True)
        if len(ids) == 0:
            break

    print(f"\n[OK] Read {len(all_ids):,} documents + embeddings")

    # Step 2: Delete collection (this removes BOTH segment directories)
    print(f"\n[STEP 2] Deleting collection to remove duplicate segments...")
    client.delete_collection(COLLECTION)
    print(f"[OK] Collection deleted")

    # Step 3: Re-create and add everything back in one pass
    # ChromaDB will create a single clean HNSW segment
    print(f"\n[STEP 3] Re-creating collection with single HNSW segment...")
    new_col = client.create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    for start in range(0, len(all_ids), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(all_ids))
        new_col.add(
            ids=all_ids[start:end],
            documents=all_docs[start:end],
            metadatas=all_metadatas[start:end],
            embeddings=all_embeddings[start:end],
        )
        pct = end / len(all_ids) * 100
        bar = "#" * int(pct / 5) + "-" * (20 - int(pct / 5))
        print(f"  [{bar}] {end:,}/{len(all_ids):,} ({pct:.0f}%)", end="\r", flush=True)

    final_count = new_col.count()
    print(f"\n[OK] Re-inserted {final_count:,} documents")

    # Size comparison
    seg_dirs_after = [d for d in os.listdir(CHROMA_DB_DIR)
                      if os.path.isdir(os.path.join(CHROMA_DB_DIR, d))
                      and d not in ("__pycache__",)]
    print(f"[INFO] Segment directories after: {len(seg_dirs_after)}")

    total_size = sum(
        os.path.getsize(os.path.join(root, f))
        for seg in seg_dirs_after
        for root, _, files in os.walk(os.path.join(CHROMA_DB_DIR, seg))
        for f in files
    )
    print(f"[INFO] chroma_db size after: {total_size / 1024 / 1024:.1f} MB")

    print(f"\n{'=' * 60}")
    print(f"[DONE] Compaction complete — {final_count:,} docs in 1 segment")
    print(f"       RAM usage on next startup: ~half of before")
    print(f"{'=' * 60}\n")
    print("Next steps:")
    print("  git add chroma_db/")
    print("  git commit -m 'chore: compact train_routes HNSW segments'")
    print("  git push  →  Render redeploys with lower RAM usage")


if __name__ == "__main__":
    main()
