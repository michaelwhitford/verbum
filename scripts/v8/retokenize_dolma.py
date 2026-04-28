#!/usr/bin/env python3
"""Re-tokenize Dolma parquet shards with Qwen3 BBPE for v8 training.

Reads raw Dolma parquet files (text column), tokenizes each document
with the Qwen3 BBPE tokenizer, packs into fixed-size numpy shards
(50M tokens each, int32), and writes to a new output directory.

The existing GPT-NeoX shards (vocab 50277) are NOT overwritten.

Input:  /Users/mwhitford/data/fractal-bitnet/dolma-raw/*.parquet
Output: /Users/mwhitford/data/fractal-bitnet/shards-qwen3/shard_NNNNN.npy

Each document is tokenized and terminated with EOD_ID (151643).
Documents are packed contiguously within shards — no padding.
The last shard is zero-padded to SHARD_SIZE if needed.

Usage:
    cd ~/src/verbum
    uv run python scripts/v8/retokenize_dolma.py [--target-tokens 3_000_000_000]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

# ── Local imports (tokenizer from same directory) ──
sys.path.insert(0, str(Path(__file__).parent))
from tokenizer import (
    EOD_ID,
    VOCAB_SIZE,
    encode_document,
    load_tokenizer,
)

# ═══════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════

RAW_DIR = Path("/Users/mwhitford/data/fractal-bitnet/dolma-raw")
OUT_DIR = Path("/Users/mwhitford/data/fractal-bitnet/shards-qwen3")

SHARD_SIZE = 50_000_000  # tokens per shard (matches existing format)
TARGET_TOKENS = 3_000_000_000  # 3B tokens = 60 shards

# Batch size for parquet reading — avoid loading full parquet into memory
PARQUET_BATCH_SIZE = 1000  # documents per batch


# ═══════════════════════════════════════════════════════════════════
# Shard writer — accumulates tokens and flushes to disk
# ═══════════════════════════════════════════════════════════════════


class ShardWriter:
    """Accumulates token IDs and writes fixed-size .npy shards."""

    def __init__(self, out_dir: Path, shard_size: int, target_tokens: int):
        self.out_dir = out_dir
        self.shard_size = shard_size
        self.target_tokens = target_tokens

        self.buffer = np.zeros(shard_size, dtype=np.int32)
        self.buf_pos = 0
        self.shard_idx = 0
        self.total_tokens = 0
        self.total_docs = 0
        self.done = False

        out_dir.mkdir(parents=True, exist_ok=True)

    def add_document(self, token_ids: list[int]) -> bool:
        """Add a tokenized document (already includes trailing EOD).

        Returns True if target reached and writing is done.
        """
        if self.done:
            return True

        ids = np.array(token_ids, dtype=np.int32)
        remaining = len(ids)
        src_pos = 0

        while remaining > 0:
            space = self.shard_size - self.buf_pos
            take = min(remaining, space)

            self.buffer[self.buf_pos : self.buf_pos + take] = ids[src_pos : src_pos + take]
            self.buf_pos += take
            src_pos += take
            remaining -= take

            if self.buf_pos >= self.shard_size:
                self._flush_shard()

                if self.total_tokens >= self.target_tokens:
                    self.done = True
                    return True

        self.total_docs += 1
        return False

    def _flush_shard(self):
        """Write current buffer as a shard and reset."""
        path = self.out_dir / f"shard_{self.shard_idx:05d}.npy"
        np.save(path, self.buffer)
        self.total_tokens += self.shard_size
        self.shard_idx += 1
        self.buffer = np.zeros(self.shard_size, dtype=np.int32)
        self.buf_pos = 0

    def finalize(self):
        """Flush any remaining partial shard (zero-padded)."""
        if self.buf_pos > 0 and not self.done:
            # Zero-pad the rest (already zeros from allocation)
            path = self.out_dir / f"shard_{self.shard_idx:05d}.npy"
            np.save(path, self.buffer)
            self.total_tokens += self.buf_pos  # count actual tokens, not padding
            self.shard_idx += 1

    @property
    def shards_written(self) -> int:
        return self.shard_idx


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="Re-tokenize Dolma with Qwen3 BBPE")
    parser.add_argument("--target-tokens", type=int, default=TARGET_TOKENS,
                        help=f"Total tokens to produce (default: {TARGET_TOKENS:,})")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR,
                        help="Directory containing Dolma parquet files")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR,
                        help="Output directory for Qwen3 shards")
    parser.add_argument("--shard-size", type=int, default=SHARD_SIZE,
                        help=f"Tokens per shard (default: {SHARD_SIZE:,})")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Process only first parquet, cap at 1 shard")
    args = parser.parse_args()

    # Force lazy import of pyarrow only when running
    import pyarrow.parquet as pq

    print("=" * 60)
    print("  Dolma Re-tokenization — GPT-NeoX → Qwen3 BBPE")
    print("=" * 60)
    print()

    # ── Load tokenizer ────────────────────────────────────────────
    t0 = time.time()
    tok = load_tokenizer()
    print(f"  Tokenizer: Qwen3 BBPE (vocab={len(tok)}, model_dim={VOCAB_SIZE})")
    print(f"  EOD_ID: {EOD_ID}")
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # ── Discover parquet files ────────────────────────────────────
    parquet_files = sorted(
        p for p in args.raw_dir.iterdir()
        if p.suffix == ".parquet" and p.is_file()
    )
    print(f"\n  Raw parquets: {len(parquet_files)} files in {args.raw_dir}")

    if not parquet_files:
        print("  ERROR: No parquet files found!")
        sys.exit(1)

    if args.smoke_test:
        parquet_files = parquet_files[:1]
        args.target_tokens = args.shard_size  # just 1 shard
        print(f"  SMOKE TEST: 1 file, {args.target_tokens:,} tokens")

    # ── Initialize shard writer ───────────────────────────────────
    writer = ShardWriter(args.out_dir, args.shard_size, args.target_tokens)
    print(f"  Output: {args.out_dir}")
    print(f"  Target: {args.target_tokens:,} tokens ({args.target_tokens // args.shard_size} shards)")
    print(f"  Shard size: {args.shard_size:,} tokens")

    # ── Process parquets ──────────────────────────────────────────
    print(f"\n  Processing...")
    t_start = time.time()
    file_tokens = 0
    file_docs = 0
    n_errors = 0

    for fi, pq_path in enumerate(parquet_files):
        if writer.done:
            break

        pf = pq.ParquetFile(pq_path)
        file_tokens = 0
        file_docs = 0

        for batch in pf.iter_batches(batch_size=PARQUET_BATCH_SIZE, columns=["text"]):
            if writer.done:
                break

            texts = batch.column("text").to_pylist()
            for text in texts:
                if writer.done:
                    break
                if not text or not text.strip():
                    continue

                try:
                    ids = encode_document(text)
                except Exception as e:
                    n_errors += 1
                    if n_errors <= 5:
                        print(f"    WARN: encode error ({e}), skipping doc")
                    continue

                file_tokens += len(ids)
                file_docs += 1
                writer.add_document(ids)

        elapsed = time.time() - t_start
        tps = writer.total_tokens / elapsed if elapsed > 0 else 0
        pct = 100 * writer.total_tokens / args.target_tokens
        print(
            f"    [{fi + 1:2d}/{len(parquet_files)}] "
            f"{pq_path.name}: {file_docs:,} docs, {file_tokens:,} tokens | "
            f"Total: {writer.total_tokens:,} ({pct:.1f}%) | "
            f"{writer.shards_written} shards | "
            f"{tps:,.0f} tok/s"
        )

    # Flush any remaining
    writer.finalize()

    elapsed = time.time() - t_start

    # ── Write provenance ──────────────────────────────────────────
    status = {
        "tokenizer": "Qwen3-BBPE",
        "tokenizer_model": "Qwen/Qwen3-8B",
        "vocab_size": VOCAB_SIZE,
        "eod_id": EOD_ID,
        "source": str(args.raw_dir),
        "source_files": len(parquet_files),
        "shards_written": writer.shards_written,
        "shard_size": args.shard_size,
        "total_tokens": writer.total_tokens,
        "total_documents": writer.total_docs,
        "target_tokens": args.target_tokens,
        "errors_skipped": n_errors,
        "elapsed_seconds": round(elapsed, 1),
        "tokens_per_second": round(writer.total_tokens / elapsed) if elapsed > 0 else 0,
        "timestamp": datetime.now(UTC).isoformat(),
        "dtype": "int32",
    }

    status_path = args.out_dir / "prep_status.json"
    status_path.write_text(json.dumps(status, indent=2))

    # ── Summary ───────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"  DONE — {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    print("=" * 60)
    print(f"  Shards: {writer.shards_written}")
    print(f"  Tokens: {writer.total_tokens:,}")
    print(f"  Docs:   {writer.total_docs:,}")
    print(f"  Errors: {n_errors}")
    print(f"  Rate:   {writer.total_tokens / elapsed:,.0f} tok/s")
    print(f"  Output: {args.out_dir}")
    print(f"  Status: {status_path}")

    # ── Quick verification ────────────────────────────────────────
    if writer.shards_written > 0:
        print(f"\n  Verification (shard_00000):")
        s = np.load(args.out_dir / "shard_00000.npy")
        print(f"    shape={s.shape}, dtype={s.dtype}")
        print(f"    min={s.min()}, max={s.max()}")
        n_eod = (s == EOD_ID).sum()
        print(f"    EOD tokens: {n_eod:,} (≈{n_eod} documents in shard)")
        # Decode first 50 tokens
        from tokenizer import decode
        snippet = decode(s[:50].tolist())
        print(f"    First 50 tokens decode: {snippet[:120]!r}...")


if __name__ == "__main__":
    main()
