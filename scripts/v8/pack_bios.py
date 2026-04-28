#!/usr/bin/env python3
"""Pack BIOS flash training data into Qwen3 BBPE shards.

Reads plain text examples (one per line) from stdin or a file,
tokenizes with Qwen3 BBPE, packs into .npy shards.

Designed to consume output from: bb gen-bios > examples.txt

Usage:
    bb gen-bios | uv run python scripts/v8/pack_bios.py
    uv run python scripts/v8/pack_bios.py --input examples.txt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

# ── Local imports ──
sys.path.insert(0, str(Path(__file__).parent))
from tokenizer import EOD_ID, VOCAB_SIZE, encode_document, load_tokenizer

OUT_DIR = Path("/Users/mwhitford/data/fractal-bitnet/shards-bios")
SHARD_SIZE = 50_000_000


def main():
    parser = argparse.ArgumentParser(description="Pack BIOS examples into Qwen3 shards")
    parser.add_argument("--input", type=Path, default=None,
                        help="Input file (default: stdin)")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--shard-size", type=int, default=SHARD_SIZE)
    args = parser.parse_args()

    print("=" * 60, file=sys.stderr)
    print("  BIOS Flash — Qwen3 BBPE Shard Packer", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    # Load tokenizer
    t0 = time.time()
    tok = load_tokenizer()
    print(f"  Tokenizer loaded in {time.time() - t0:.1f}s", file=sys.stderr)
    print(f"  EOD_ID: {EOD_ID}, vocab: {VOCAB_SIZE}", file=sys.stderr)

    # Read examples
    if args.input:
        print(f"  Reading from {args.input}", file=sys.stderr)
        source = open(args.input, "r", encoding="utf-8")
    else:
        print("  Reading from stdin", file=sys.stderr)
        source = sys.stdin

    # Tokenize + accumulate
    t_start = time.time()
    all_ids: list[int] = []
    n_examples = 0

    for line in source:
        line = line.strip()
        if not line:
            continue
        ids = encode_document(line)
        all_ids.extend(ids)
        n_examples += 1

        if n_examples % 100_000 == 0:
            print(f"    {n_examples:,} examples, {len(all_ids):,} tokens...",
                  file=sys.stderr)

    if args.input:
        source.close()

    total_tokens = len(all_ids)
    elapsed_tok = time.time() - t_start
    print(f"\n  Tokenized: {n_examples:,} examples → {total_tokens:,} tokens "
          f"({elapsed_tok:.1f}s)", file=sys.stderr)
    print(f"  Avg tokens/example: {total_tokens / max(1, n_examples):.1f}",
          file=sys.stderr)

    # Pack into shards
    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_ids_np = np.array(all_ids, dtype=np.int32)

    shard_idx = 0
    pos = 0
    while pos + args.shard_size <= len(all_ids_np):
        shard = all_ids_np[pos : pos + args.shard_size]
        path = args.out_dir / f"shard_{shard_idx:05d}.npy"
        np.save(path, shard)
        shard_idx += 1
        pos += args.shard_size

    # Last partial shard (zero-padded if > 1000 tokens)
    remainder = len(all_ids_np) - pos
    if remainder > 1000:
        shard = np.zeros(args.shard_size, dtype=np.int32)
        shard[:remainder] = all_ids_np[pos:]
        path = args.out_dir / f"shard_{shard_idx:05d}.npy"
        np.save(path, shard)
        shard_idx += 1

    # Verify first shard
    s0 = np.load(args.out_dir / "shard_00000.npy")
    n_eod = int((s0 == EOD_ID).sum())

    # Write provenance
    status = {
        "type": "bios-flash",
        "generator": "bb gen-bios (babashka)",
        "tokenizer": "Qwen3-BBPE",
        "tokenizer_model": "Qwen/Qwen3-8B",
        "vocab_size": VOCAB_SIZE,
        "eod_id": EOD_ID,
        "total_examples": n_examples,
        "total_tokens": total_tokens,
        "shards_written": shard_idx,
        "shard_size": args.shard_size,
        "avg_tokens_per_example": round(total_tokens / max(1, n_examples), 1),
        "eod_in_shard_0": n_eod,
        "max_token_id": int(all_ids_np.max()),
        "dtype": "int32",
        "timestamp": datetime.now(UTC).isoformat(),
    }
    status_path = args.out_dir / "prep_status.json"
    status_path.write_text(json.dumps(status, indent=2))

    # Summary
    print(f"\n  Packed: {shard_idx} shard(s) × {args.shard_size:,} tokens",
          file=sys.stderr)
    print(f"  Max token ID: {all_ids_np.max()} (vocab: {VOCAB_SIZE})",
          file=sys.stderr)
    print(f"  EOD in shard_0: {n_eod:,}", file=sys.stderr)
    print(f"  Output: {args.out_dir}", file=sys.stderr)
    print(f"  Status: {status_path}", file=sys.stderr)

    # Decode spot check
    from tokenizer import decode
    eod_pos = np.where(s0 == EOD_ID)[0]
    print(f"\n  Spot check (first 5 examples):", file=sys.stderr)
    start = 0
    for i in range(min(5, len(eod_pos))):
        end = eod_pos[i]
        text = decode(s0[start:end].tolist())
        print(f"    {text}", file=sys.stderr)
        start = end + 1


if __name__ == "__main__":
    main()
