#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
remove_white_rooks.py

Removes ALL White Rook samples from:
  - embeddings (ROOT/embeddings/*.pt + *.csv)
  - one or more data reference files (CSV or JSONL) you pass via --data-ref

Behavior:
  - Dry run by default (no write) — pass --commit to apply.
  - Makes timestamped backups before overwriting.
  - Label matching is robust: "White Rook", "white_rook", "WR", "wR", etc.
  - Skips anything that looks like Black.

Examples:
  # Dry run (just show what would change)
  python remove_white_rooks.py

  # Apply changes to embeddings + two data ref files
  python remove_white_rooks.py --data-ref /path/data_refs.csv --data-ref /path/refs.jsonl --commit

  # Custom embeddings dir / filenames
  python remove_white_rooks.py --emb-dir /path/embeddings --tensor embeddings_master.pt --meta embeddings_master_meta.csv --commit
"""

from __future__ import annotations
import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import pandas as pd

# Defaults (adjust ROOT if needed)
ROOT = Path("/Users/andredelacruz/Documents/GitHub/cpchess/perception_api")
EMB_DIR_DEFAULT = ROOT / "embeddings"
TENSOR_NAME_DEFAULT = "embeddings_master.pt"
META_NAME_DEFAULT = "embeddings_master_meta.csv"


def ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def is_white_rook(label: str) -> bool:
    """
    Heuristic matcher for 'White Rook' class labels.

    Matches:
      - "White Rook", "white_rook", "WhiteRook"
      - Abbrev: "WR", "wR", "WROOK"
      - Patterns like "R_white", "RWhite"
    Excludes any label that looks Black.
    """
    if not isinstance(label, str):
        return False
    low = label.strip().lower()

    # hard exclude black-ish labels
    if "black" in low:
        return False
    if re.fullmatch(r"\s*b\s*r(?:ook)?\s*", label.strip().lower()):
        return False
    if re.fullmatch(r"br", label.strip().lower()):
        return False

    # explicit tokens
    if ("white" in low and "rook" in low):
        return True

    # tokenization
    tokens = re.findall(r"[a-zA-Z]+", low)
    if "rook" in tokens and ("white" in tokens or "w" in tokens):
        return True

    # abbreviations
    up = label.strip().upper()
    if up in {"WR", "WROOK"} or re.fullmatch(r"W+R", up):
        return True

    # loose fallback
    if ("rook" in low or low.endswith("r")) and ("white" in low or low.startswith("w")):
        return True

    return False


def backup_file(p: Path) -> Path:
    b = p.with_suffix(p.suffix + f".bak{ts()}")
    b.write_bytes(p.read_bytes())
    return b


def load_embeddings_and_meta(emb_dir: Path, tensor_name: str, meta_name: str):
    tpath = emb_dir / tensor_name
    mpath = emb_dir / meta_name

    if not tpath.exists():
        raise FileNotFoundError(f"Tensor file not found: {tpath}")
    if not mpath.exists():
        raise FileNotFoundError(f"Meta CSV not found: {mpath}")

    embs = torch.load(tpath, map_location="cpu")
    if not isinstance(embs, torch.Tensor):
        raise TypeError(f"Expected a torch.Tensor in {tpath}, got {type(embs)}")

    meta = pd.read_csv(mpath)
    if "label" not in meta.columns:
        raise ValueError(f"Meta CSV {mpath} must contain a 'label' column.")

    # align like your pipeline
    n = min(len(meta), embs.shape[0])
    if n != len(meta) or n != embs.shape[0]:
        print(f"[WARN] Aligning to first {n} rows due to mismatch (meta={len(meta)}, embs={embs.shape[0]}).")
        meta = meta.iloc[:n].reset_index(drop=True)
        embs = embs[:n]

    return embs, meta, tpath, mpath


def compute_keep_remove_from_meta(meta: pd.DataFrame) -> Tuple[List[int], List[int]]:
    remove_idx = [i for i, lbl in enumerate(meta["label"].tolist()) if is_white_rook(lbl)]
    keep_idx = [i for i in range(len(meta)) if i not in remove_idx]
    return keep_idx, remove_idx


def detect_label_column(df: pd.DataFrame) -> Optional[str]:
    """Try to find the label column in a data_references table."""
    candidates = ["label", "class", "piece", "target", "y"]
    for c in candidates:
        if c in df.columns:
            return c
    # looser search
    for c in df.columns:
        if "label" in c.lower() or "class" in c.lower():
            return c
    return None


def filter_data_ref(path: Path, commit: bool) -> Tuple[int, int, Optional[Path]]:
    """
    Filter a data_references file (CSV or JSONL) in-place (with backup if commit).
    Returns: (original_count, new_count, backup_path_if_any)
    """
    suffix = path.suffix.lower()

    if suffix in {".csv"}:
        df = pd.read_csv(path)
        col = detect_label_column(df)
        if col is None:
            print(f"[WARN] {path}: could not find a label column; skipping.")
            return len(df), len(df), None

        orig = len(df)
        mask = df[col].apply(is_white_rook).fillna(False)
        to_remove = int(mask.sum())
        new_df = df.loc[~mask].reset_index(drop=True)
        new_cnt = len(new_df)

        print(f"{path}: found {to_remove} white rook rows (CSV).")
        if commit and to_remove > 0:
            bak = backup_file(path)
            new_df.to_csv(path, index=False)
            return orig, new_cnt, bak
        return orig, new_cnt, None

    elif suffix in {".jsonl", ".json"}:
        # Treat .json as JSONL if it appears line-delimited; otherwise try list-of-objects
        lines = path.read_text().splitlines()
        # Heuristic: if file has multiple lines and each looks like JSON, assume JSONL
        is_jsonl = len(lines) > 1
        records = []
        if is_jsonl:
            for i, line in enumerate(lines):
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    print(f"[WARN] {path}: invalid JSON on line {i+1}; keeping line.")
                    obj = {"__raw__": line}  # keep as-is
                records.append(obj)
        else:
            try:
                maybe = json.loads(path.read_text())
                if isinstance(maybe, list):
                    records = maybe
                else:
                    print(f"[WARN] {path}: JSON is not a list; wrapping as single record.")
                    records = [maybe]
            except Exception as e:
                print(f"[WARN] {path}: cannot parse JSON ({e}); skipping.")
                return 0, 0, None

        # Convert to DataFrame to reuse label detection
        df = pd.DataFrame(records)
        if df.empty:
            return 0, 0, None

        col = detect_label_column(df)
        if col is None:
            print(f"[WARN] {path}: could not find a label field; skipping.")
            return len(df), len(df), None

        orig = len(df)
        mask = df[col].apply(is_white_rook).fillna(False)
        to_remove = int(mask.sum())
        new_df = df.loc[~mask].reset_index(drop=True)
        new_cnt = len(new_df)

        print(f"{path}: found {to_remove} white rook rows ({'JSONL' if is_jsonl else 'JSON list'}).")
        if commit and to_remove > 0:
            bak = backup_file(path)
            if is_jsonl:
                with path.open("w", encoding="utf-8") as f:
                    for _, row in new_df.iterrows():
                        json.dump(row.to_dict(), f, ensure_ascii=False)
                        f.write("\n")
            else:
                path.write_text(json.dumps(new_df.to_dict(orient="records"), indent=2))
            return orig, new_cnt, bak
        return orig, new_cnt, None

    else:
        print(f"[WARN] {path}: unsupported extension (use .csv, .jsonl, or .json).")
        return 0, 0, None


def main():
    ap = argparse.ArgumentParser(description="Remove all White Rook samples from embeddings/meta and data references.")
    ap.add_argument("--emb-dir", type=Path, default=EMB_DIR_DEFAULT, help="Directory containing embeddings files")
    ap.add_argument("--tensor", type=str, default=TENSOR_NAME_DEFAULT, help="Embeddings tensor filename")
    ap.add_argument("--meta", type=str, default=META_NAME_DEFAULT, help="Embeddings meta CSV filename")
    ap.add_argument("--data-ref", type=Path, action="append", default=[],
                    help="Data reference file to also filter (CSV or JSONL). Repeatable.")
    ap.add_argument("--commit", action="store_true", help="Apply changes (otherwise dry run)")
    args = ap.parse_args()

    # --- Embeddings + meta ---
    embs, meta, tpath, mpath = load_embeddings_and_meta(args.emb_dir, args.tensor, args.meta)
    keep_idx, remove_idx = compute_keep_remove_from_meta(meta)

    print(f"Loaded: {len(meta)} samples from:\n  {tpath}\n  {mpath}")
    print(f"Found {len(remove_idx)} White Rook samples in embeddings/meta.")

    if remove_idx:
        preview = meta.iloc[remove_idx[:10]][["label"]]
        print("\nExamples to be removed from meta (up to 10):")
        print(preview.to_string(index=False))
    else:
        print("\nNo White Rook samples found in embeddings/meta.")

    if args.commit and remove_idx:
        print("\n--commit specified: creating backups and writing filtered embeddings/meta...")
        tbak = backup_file(tpath)
        mbak = backup_file(mpath)
        print(f"Backed up tensor to: {tbak}")
        print(f"Backed up meta   to: {mbak}")

        embs_new = embs[keep_idx]
        meta_new = meta.iloc[keep_idx].reset_index(drop=True)
        torch.save(embs_new, tpath)
        meta_new.to_csv(mpath, index=False)

        print(f"Embeddings: {embs.shape} -> {embs_new.shape}")
        print(f"Meta rows : {len(meta)} -> {len(meta_new)}")
    else:
        print("\n(Dry run) No changes written to embeddings/meta.")

    # --- Data references ---
    if args.data_ref:
        print("\nProcessing data reference files:")
        for dr in args.data_ref:
            if not dr.exists():
                print(f"[WARN] {dr}: not found; skipping.")
                continue
            orig, new_cnt, bak = filter_data_ref(dr, commit=args.commit)
            if args.commit and bak:
                print(f"  Wrote filtered file. Backup: {bak}")
            else:
                if orig or new_cnt:
                    print(f"  (Dry run) {dr}: {orig} -> {new_cnt}")
    else:
        print("\nNo --data-ref files provided. (Nothing to filter outside embeddings/meta.)")

    if not args.commit:
        print("\nDry run complete. Re-run with --commit to apply changes (backups will be created).")


if __name__ == "__main__":
    main()
