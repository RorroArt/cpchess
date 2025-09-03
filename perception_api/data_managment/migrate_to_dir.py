#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
migrate_embeddings_to_dir.py

Move/merge embeddings + meta CSV from:
  /Users/andredelacruz/Documents/GitHub/cpchess/perception_api/embeddings_master.pt
  /Users/andredelacruz/Documents/GitHub/cpchess/perception_api/embeddings_master_meta.csv
to:
  /Users/andredelacruz/Documents/GitHub/cpchess/perception_api/embeddings/embeddings_master.pt
  /Users/andredelacruz/Documents/GitHub/cpchess/perception_api/embeddings/embeddings_master_meta.csv

Behavior:
- Dry run by default (prints actions).
- With --commit: creates timestamped backups of destination if present, merges (row-wise / dim-0),
  writes combined outputs, and (optionally) deletes source with --delete-source.

Usage:
  python migrate_embeddings_to_dir.py                  # dry run
  python migrate_embeddings_to_dir.py --commit         # do it
  python migrate_embeddings_to_dir.py --commit --delete-source
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime
import argparse
import shutil
import pandas as pd
import torch

ROOT = Path("/Users/andredelacruz/Documents/GitHub/cpchess/perception_api")

SRC_TENSOR = ROOT / "embeddings_master.pt"
SRC_META   = ROOT / "embeddings_master_meta.csv"

DST_DIR    = ROOT / "embeddings"
DST_TENSOR = DST_DIR / "embeddings_master.pt"
DST_META   = DST_DIR / "embeddings_master_meta.csv"


def ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def backup(p: Path) -> Path:
    b = p.with_suffix(p.suffix + f".bak{ts()}")
    b.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(p, b)
    return b


def load_src():
    if not SRC_TENSOR.exists():
        raise FileNotFoundError(f"Source tensor not found: {SRC_TENSOR}")
    if not SRC_META.exists():
        raise FileNotFoundError(f"Source meta CSV not found: {SRC_META}")

    src_emb = torch.load(SRC_TENSOR, map_location="cpu")
    if not isinstance(src_emb, torch.Tensor):
        raise TypeError(f"Expected a torch.Tensor in {SRC_TENSOR}, got {type(src_emb)}")
    src_meta = pd.read_csv(SRC_META)
    return src_emb, src_meta


def ensure_cols_match(df_src: pd.DataFrame, df_dst: pd.DataFrame) -> pd.DataFrame:
    # Align columns (union), fill missing with ""
    all_cols = list(dict.fromkeys(list(df_dst.columns) + list(df_src.columns)))
    return df_src.reindex(columns=all_cols, fill_value="")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", action="store_true", help="Write outputs and backups")
    ap.add_argument("--delete-source", action="store_true", help="Delete source files after successful commit")
    args = ap.parse_args()

    print("Loading sources...")
    src_emb, src_meta = load_src()
    print(f"  Source embeddings: {tuple(src_emb.shape)} from {SRC_TENSOR}")
    print(f"  Source meta rows : {len(src_meta)} from {SRC_META}")

    DST_DIR.mkdir(parents=True, exist_ok=True)

    if DST_TENSOR.exists() and DST_META.exists():
        print("Destination exists; will MERGE into destination.")
        dst_emb = torch.load(DST_TENSOR, map_location="cpu")
        if dst_emb.shape[1] != src_emb.shape[1]:
            raise ValueError(f"Dim mismatch: dst D={dst_emb.shape[1]} vs src D={src_emb.shape[1]}")
        dst_meta = pd.read_csv(DST_META)

        # Align meta columns across both
        src_meta_aligned = ensure_cols_match(src_meta, dst_meta)
        dst_meta_aligned = ensure_cols_match(dst_meta, src_meta)

        # Merge
        merged_emb = torch.cat([dst_emb, src_emb], dim=0)
        merged_meta = pd.concat([dst_meta_aligned, src_meta_aligned], ignore_index=True)

        print(f"  Merged embeddings: {tuple(dst_emb.shape)} + {tuple(src_emb.shape)} → {tuple(merged_emb.shape)}")
        print(f"  Merged meta rows : {len(dst_meta)} + {len(src_meta)} → {len(merged_meta)}")

        if args.commit:
            b1 = backup(DST_TENSOR)
            b2 = backup(DST_META)
            print(f"Backed up: {b1}")
            print(f"Backed up: {b2}")
            torch.save(merged_emb, DST_TENSOR)
            merged_meta.to_csv(DST_META, index=False)
            print(f"Wrote merged tensor → {DST_TENSOR}")
            print(f"Wrote merged meta   → {DST_META}")
    else:
        print("Destination missing; will MOVE sources to destination paths.")
        if args.commit:
            # If one of the dest files exists (partially), back it up
            if DST_TENSOR.exists(): print(f"[WARN] {DST_TENSOR} exists; backing up → {backup(DST_TENSOR)}")
            if DST_META.exists():   print(f"[WARN] {DST_META} exists; backing up → {backup(DST_META)}")
            torch.save(src_emb, DST_TENSOR)
            src_meta.to_csv(DST_META, index=False)
            print(f"Wrote tensor → {DST_TENSOR}")
            print(f"Wrote meta   → {DST_META}")

    if args.commit and args.delete_source:
        try:
            SRC_TENSOR.unlink(missing_ok=True)
            SRC_META.unlink(missing_ok=True)
            print("Deleted source files.")
        except Exception as e:
            print(f"[WARN] Could not delete sources: {e}")

    if not args.commit:
        print("\nDry run complete. Re-run with --commit to apply changes.")


if __name__ == "__main__":
    main()
