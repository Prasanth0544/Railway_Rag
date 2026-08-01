#!/usr/bin/env python3
"""
🚂 Railway RAG Assistant - Industry Standard LOC (Lines of Code) Counter
========================================================================
Analyzes physical, source (SLOC), comment, and blank lines across all files,
categorizing by language and project layer.

Usage:
    python scripts/count_loc.py
    python scripts/count_loc.py --format markdown
    python scripts/count_loc.py --format json
    python scripts/count_loc.py --format badge
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Ensure stdout handles UTF-8 characters safely on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Root directory of the repository
REPO_ROOT = Path(__file__).resolve().parent.parent

# Default directories and files to exclude from LOC analysis
DEFAULT_EXCLUDES = {
    ".git", ".venv", "venv", "env", "chroma_db", "data",
    "__pycache__", ".pytest_cache", ".gemini", ".idea", ".vscode"
}

# Vendor or third-party assets to flag / handle separately
VENDOR_FILES = {
    "web/assets/marked.min.js",
}

# Language mapping based on extension / filename
LANG_MAP = {
    ".py": "Python",
    ".js": "JavaScript",
    ".css": "CSS",
    ".html": "HTML",
    ".md": "Markdown",
    ".sh": "Shell",
    ".yml": "YAML",
    ".yaml": "YAML",
    ".json": "JSON",
    "Dockerfile": "Docker",
    ".env": "Config",
    ".env.example": "Config",
    ".gitignore": "Config",
    ".gitattributes": "Config",
    "requirements.txt": "Config",
}

# Subsystem / Layer mapping by path prefix
def determine_layer(rel_path: str) -> str:
    path_str = rel_path.replace("\\", "/")
    if path_str.startswith("app/") or path_str == "streamlit_app.py":
        return "Backend (Core App)"
    elif path_str.startswith("web/"):
        if path_str in VENDOR_FILES:
            return "Frontend (Vendor Assets)"
        return "Frontend (UI Web)"
    elif path_str.startswith("scripts/"):
        return "Scripts & Pipelines"
    elif path_str.endswith(".md"):
        return "Documentation"
    else:
        return "Config & Infrastructure"


def analyze_file(filepath: Path) -> Tuple[int, int, int, int]:
    """
    Analyzes a single file and returns (sloc, comments, blank, total).
    SLOC = Source Lines of Code (logical executable/content lines).
    """
    total = 0
    blank = 0
    comments = 0
    sloc = 0
    
    ext = filepath.suffix.lower()
    filename = filepath.name
    
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except Exception:
        return (0, 0, 0, 0)
        
    total = len(lines)
    in_multiline_comment = False
    multiline_char = None
    
    for line in lines:
        stripped = line.strip()
        
        if not stripped:
            blank += 1
            continue
            
        # Multiline comment tracking for Python (""" or ''')
        if ext == ".py":
            if in_multiline_comment:
                comments += 1
                if multiline_char in stripped:
                    in_multiline_comment = False
                continue
            elif stripped.startswith('"""') or stripped.startswith("'''"):
                comments += 1
                # Check if it starts and ends on the same line
                quote_type = stripped[:3]
                remainder = stripped[3:]
                if quote_type not in remainder:
                    in_multiline_comment = True
                    multiline_char = quote_type
                continue
            elif stripped.startswith("#"):
                comments += 1
                continue
        # Multiline comment tracking for C-style (JS/CSS)
        elif ext in (".js", ".css"):
            if in_multiline_comment:
                comments += 1
                if "*/" in stripped:
                    in_multiline_comment = False
                continue
            elif stripped.startswith("/*"):
                comments += 1
                if "*/" not in stripped:
                    in_multiline_comment = True
                continue
            elif stripped.startswith("//"):
                comments += 1
                continue
        # HTML comment tracking
        elif ext == ".html":
            if in_multiline_comment:
                comments += 1
                if "-->" in stripped:
                    in_multiline_comment = False
                continue
            elif stripped.startswith("<!--"):
                comments += 1
                if "-->" not in stripped:
                    in_multiline_comment = True
                continue
        # Shell / YAML / Config comments
        elif ext in (".sh", ".yml", ".yaml", ".env") or filename in ("Dockerfile", ".gitignore", "requirements.txt"):
            if stripped.startswith("#"):
                comments += 1
                continue
        elif ext == ".md":
            if stripped.startswith("<!--"):
                comments += 1
                continue
                
        # If not blank and not comment line, count as SLOC
        sloc += 1

    return (sloc, comments, blank, total)


def collect_loc_stats(root_dir: Path) -> List[Dict[str, Any]]:
    stats = []
    
    for root, dirs, files in os.walk(root_dir):
        # Prune excluded directories
        dirs[:] = [d for d in dirs if d not in DEFAULT_EXCLUDES]
        
        for file in files:
            full_path = Path(root) / file
            rel_path = str(full_path.relative_to(root_dir)).replace("\\", "/")
            
            ext = full_path.suffix.lower()
            filename = full_path.name
            
            lang = LANG_MAP.get(ext) or LANG_MAP.get(filename, "Other")
            layer = determine_layer(rel_path)
            
            sloc, comments, blank, total = analyze_file(full_path)
            
            if total > 0:
                stats.append({
                    "file": rel_path,
                    "language": lang,
                    "layer": layer,
                    "sloc": sloc,
                    "comments": comments,
                    "blank": blank,
                    "total": total
                })
                
    return sorted(stats, key=lambda x: x["total"], reverse=True)


def print_table(stats: List[Dict[str, Any]]):
    print("\n" + "="*80)
    print(" 🚂 RAILWAY RAG ASSISTANT - LINES OF CODE (LOC) METRICS SUMMARY")
    print("="*80)
    
    # Layer Breakdown
    layers: Dict[str, Dict[str, int]] = {}
    langs: Dict[str, Dict[str, int]] = {}
    grand_totals = {"files": 0, "sloc": 0, "comments": 0, "blank": 0, "total": 0}
    
    for item in stats:
        lyr = item["layer"]
        lng = item["language"]
        
        if lyr not in layers:
            layers[lyr] = {"files": 0, "sloc": 0, "comments": 0, "blank": 0, "total": 0}
        if lng not in langs:
            langs[lng] = {"files": 0, "sloc": 0, "comments": 0, "blank": 0, "total": 0}
            
        for key in ("sloc", "comments", "blank", "total"):
            layers[lyr][key] += item[key]
            langs[lng][key] += item[key]
            grand_totals[key] += item[key]
            
        layers[lyr]["files"] += 1
        langs[lng]["files"] += 1
        grand_totals["files"] += 1

    print("\n📊 BREAKDOWN BY PROJECT LAYER:")
    print("-" * 80)
    print(f"{'Layer':<28} | {'Files':<6} | {'SLOC':<8} | {'Comments':<9} | {'Blank':<7} | {'Total LOC':<9}")
    print("-" * 80)
    for lyr, data in sorted(layers.items(), key=lambda x: x[1]["total"], reverse=True):
        print(f"{lyr:<28} | {data['files']:<6d} | {data['sloc']:<8d} | {data['comments']:<9d} | {data['blank']:<7d} | {data['total']:<9d}")
    print("-" * 80)
    print(f"{'TOTAL':<28} | {grand_totals['files']:<6d} | {grand_totals['sloc']:<8d} | {grand_totals['comments']:<9d} | {grand_totals['blank']:<7d} | {grand_totals['total']:<9d}")
    print("-" * 80)

    print("\n💻 BREAKDOWN BY LANGUAGE:")
    print("-" * 80)
    print(f"{'Language':<20} | {'Files':<6} | {'SLOC':<8} | {'Comments':<9} | {'Blank':<7} | {'Total LOC':<9}")
    print("-" * 80)
    for lng, data in sorted(langs.items(), key=lambda x: x[1]["total"], reverse=True):
        print(f"{lng:<20} | {data['files']:<6d} | {data['sloc']:<8d} | {data['comments']:<9d} | {data['blank']:<7d} | {data['total']:<9d}")
    print("-" * 80)

    print("\n📄 TOP 10 LARGEST FILES:")
    print("-" * 80)
    print(f"{'File Path':<40} | {'Layer':<22} | {'SLOC':<7} | {'Total':<7}")
    print("-" * 80)
    for item in stats[:10]:
        print(f"{item['file']:<40} | {item['layer']:<22} | {item['sloc']:<7d} | {item['total']:<7d}")
    print("-" * 80 + "\n")


def generate_markdown(stats: List[Dict[str, Any]]) -> str:
    layers: Dict[str, Dict[str, int]] = {}
    langs: Dict[str, Dict[str, int]] = {}
    grand = {"files": 0, "sloc": 0, "comments": 0, "blank": 0, "total": 0}
    
    for item in stats:
        lyr, lng = item["layer"], item["language"]
        if lyr not in layers:
            layers[lyr] = {"files": 0, "sloc": 0, "comments": 0, "blank": 0, "total": 0}
        if lng not in langs:
            langs[lng] = {"files": 0, "sloc": 0, "comments": 0, "blank": 0, "total": 0}
            
        for k in ("sloc", "comments", "blank", "total"):
            layers[lyr][k] += item[k]
            langs[lng][k] += item[k]
            grand[k] += item[k]
            
        layers[lyr]["files"] += 1
        langs[lng]["files"] += 1
        grand["files"] += 1
        
    lines = [
        "### 📊 Codebase Statistics (LOC)",
        "",
        "| Layer | Files | Source Code (SLOC) | Comments | Blank | Total LOC |",
        "|---|---|---|---|---|---|"
    ]
    
    for lyr, d in sorted(layers.items(), key=lambda x: x[1]["total"], reverse=True):
        lines.append(f"| **{lyr}** | {d['files']:,} | {d['sloc']:,} | {d['comments']:,} | {d['blank']:,} | **{d['total']:,}** |")
        
    lines.append(f"| **TOTAL** | **{grand['files']:,}** | **{grand['sloc']:,}** | **{grand['comments']:,}** | **{grand['blank']:,}** | **{grand['total']:,}** |")
    lines.append("")
    lines.append("#### Breakdown by Language")
    lines.append("")
    lines.append("| Language | Files | SLOC | Comments | Blank | Total LOC |")
    lines.append("|---|---|---|---|---|---|")
    for lng, d in sorted(langs.items(), key=lambda x: x[1]["total"], reverse=True):
        lines.append(f"| {lng} | {d['files']:,} | {d['sloc']:,} | {d['comments']:,} | {d['blank']:,} | {d['total']:,} |")
        
    lines.append("")
    return "\n".join(lines)


def format_badge(stats: List[Dict[str, Any]]):
    total_loc = sum(s["total"] for s in stats)
    sloc = sum(s["sloc"] for s in stats)
    
    loc_k = f"{total_loc / 1000:.1f}k" if total_loc >= 1000 else str(total_loc)
    sloc_k = f"{sloc / 1000:.1f}k" if sloc >= 1000 else str(sloc)
    
    print("\n📛 SHIELDS.IO BADGES FOR README.MD:")
    print("-" * 60)
    print(f"[![Total LOC](https://img.shields.io/badge/Lines_of_Code-{loc_k}-blue?logo=python)](#codebase-statistics)")
    print(f"[![Source SLOC](https://img.shields.io/badge/Source_SLOC-{sloc_k}-green)](#codebase-statistics)")
    print("-" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Industry Standard LOC Analyzer for Railway RAG Assistant")
    parser.add_argument("--format", choices=["table", "markdown", "json", "badge"], default="table",
                        help="Output format (default: table)")
    args = parser.parse_args()
    
    stats = collect_loc_stats(REPO_ROOT)
    
    if args.format == "table":
        print_table(stats)
    elif args.format == "markdown":
        print(generate_markdown(stats))
    elif args.format == "json":
        print(json.dumps(stats, indent=2))
    elif args.format == "badge":
        format_badge(stats)


if __name__ == "__main__":
    main()
