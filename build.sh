#!/bin/bash
# Render build script — runs BEFORE the app starts

echo "=== Railway RAG Build Script ==="

# Pull LFS files if chroma.sqlite3 is just a pointer (< 1MB = not real file)
SQLITE_PATH="chroma_db/chroma.sqlite3"
if [ -f "$SQLITE_PATH" ]; then
    SIZE=$(stat -c%s "$SQLITE_PATH" 2>/dev/null || stat -f%z "$SQLITE_PATH" 2>/dev/null || echo "0")
    echo "chroma.sqlite3 size: $SIZE bytes"
    if [ "$SIZE" -lt 1000000 ]; then
        echo ">>> LFS pointer detected — pulling real file..."
        git lfs pull
        echo ">>> LFS pull done"
    else
        echo ">>> chroma.sqlite3 looks good ($SIZE bytes)"
    fi
else
    echo ">>> chroma.sqlite3 not found — running git lfs pull..."
    git lfs pull
fi

echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "=== Build complete ==="
