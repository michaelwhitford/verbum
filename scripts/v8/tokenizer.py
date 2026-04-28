"""
v8 — Qwen3 BBPE Tokenizer Utility

Wraps the Qwen3 byte-level BPE tokenizer for use with the v8 model.

Vocab: 151,643 regular tokens + 26 control tokens = 151,669 used.
       Model embedding dim: 151,936 (padded for hardware alignment).
       BBPE: no unknown tokens — all text encoded via byte fallback.

Special tokens:
  <|endoftext|>  (151643) — end of document / document separator in packed sequences
  <|im_start|>   (151644) — start of turn (ChatML)
  <|im_end|>     (151645) — end of turn (ChatML)

Padding: Qwen has no native pad token. For packed training (our default),
no padding is needed. For variable-length eval batches, we assign a
dedicated pad token from the unused control token range.

Usage:
    from tokenizer import load_tokenizer, SPECIAL_TOKENS
    tok = load_tokenizer()
    ids = tok.encode("(+ 3 7)")
    text = tok.decode(ids)
"""

from __future__ import annotations

from pathlib import Path

# ═══════════════════════════════════════════════════════════════════
# Special token constants
# ═══════════════════════════════════════════════════════════════════

# These match the Qwen3 tokenizer configuration.
# Model embedding dim (151936) > len(tokenizer) (151669) — unused slots exist.

VOCAB_SIZE = 151936           # model embedding dimension (hardware-aligned)

# Qwen3 control tokens
EOD_TOKEN = "<|endoftext|>"   # end of document (document separator in packing)
EOD_ID = 151643

IM_START_TOKEN = "<|im_start|>"  # start of turn (ChatML)
IM_START_ID = 151644

IM_END_TOKEN = "<|im_end|>"     # end of turn (ChatML / eos for inference)
IM_END_ID = 151645

# Dedicated pad token — we pick an unused control slot (151646+).
# Qwen3 has ~208 control tokens, many inactive. ID 151646 is <|object_ref_start|>
# but IDs 151660+ are unused in most configs. We'll use 151665.
PAD_TOKEN = "<|pad|>"
PAD_ID = 151665  # unused control token slot in Qwen3

# Verbum-specific control tokens (for io!, partial, value output modes)
# Reserved from the unused range. Not yet active — placeholders for training.
VALUE_TOKEN = "<|value|>"
VALUE_ID = 151666

PARTIAL_TOKEN = "<|partial|>"
PARTIAL_ID = 151667

# Note: 151668 = <|/think|> in Qwen3, avoid it.

IO_TOKEN = "<|io|>"
IO_ID = 151670  # safely above Qwen3's used range


# ═══════════════════════════════════════════════════════════════════
# Tokenizer loading
# ═══════════════════════════════════════════════════════════════════

_QWEN_MODEL = "Qwen/Qwen3-8B"
_tokenizer = None


def load_tokenizer(model_name: str = _QWEN_MODEL):
    """Load the Qwen3 tokenizer.

    Uses transformers.AutoTokenizer. Caches the instance — safe to call
    repeatedly. The tokenizer files are downloaded once to HF cache.

    Returns a PreTrainedTokenizerFast instance.
    """
    global _tokenizer
    if _tokenizer is not None:
        return _tokenizer

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)

    # Verify expected special tokens
    assert tok.convert_tokens_to_ids(EOD_TOKEN) == EOD_ID, \
        f"EOD token mismatch: expected {EOD_ID}"
    assert tok.convert_tokens_to_ids(IM_END_TOKEN) == IM_END_ID, \
        f"IM_END token mismatch: expected {IM_END_ID}"

    _tokenizer = tok
    return tok


def encode(text: str, add_special_tokens: bool = False) -> list[int]:
    """Encode text to token IDs using Qwen3 BBPE.

    Default: no special tokens added (raw BPE encoding).
    For packed training, documents are separated by EOD_ID manually.
    """
    tok = load_tokenizer()
    return tok.encode(text, add_special_tokens=add_special_tokens)


def decode(ids: list[int], skip_special_tokens: bool = False) -> str:
    """Decode token IDs back to text."""
    tok = load_tokenizer()
    return tok.decode(ids, skip_special_tokens=skip_special_tokens)


def encode_document(text: str) -> list[int]:
    """Encode a document with EOD separator appended.

    For packed training: each document → encode(text) + [EOD_ID].
    Multiple documents packed into one sequence, separated by EOD.
    """
    ids = encode(text)
    ids.append(EOD_ID)
    return ids


# ═══════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  v8 — Qwen3 BBPE Tokenizer")
    print("=" * 60)

    tok = load_tokenizer()
    print(f"\nTokenizer: {_QWEN_MODEL}")
    print(f"  vocab_size (BPE regular): {tok.vocab_size}")
    print(f"  len(tokenizer):           {len(tok)}")
    print(f"  model embedding dim:      {VOCAB_SIZE}")
    print(f"  eos_token: {tok.eos_token!r} (id={tok.eos_token_id})")
    print(f"  pad_token: {tok.pad_token!r} (id={tok.pad_token_id})")
    print(f"  unk_token: {tok.unk_token!r}")

    print(f"\nSpecial tokens:")
    print(f"  EOD:      {EOD_TOKEN:20s} id={EOD_ID}")
    print(f"  IM_START: {IM_START_TOKEN:20s} id={IM_START_ID}")
    print(f"  IM_END:   {IM_END_TOKEN:20s} id={IM_END_ID}")
    print(f"  PAD:      {PAD_TOKEN:20s} id={PAD_ID} (verbum-assigned)")
    print(f"  VALUE:    {VALUE_TOKEN:20s} id={VALUE_ID} (verbum-reserved)")
    print(f"  PARTIAL:  {PARTIAL_TOKEN:20s} id={PARTIAL_ID} (verbum-reserved)")
    print(f"  IO:       {IO_TOKEN:20s} id={IO_ID} (verbum-reserved)")

    # Test encoding examples
    print(f"\nTokenization examples:")
    examples = [
        # Math
        "(+ 3 7)",
        "(* 123 456)",
        # Clojure
        "(fn [x] (* x x))",
        "(reduce + (map #(* % %) (filter even? (range 10))))",
        "(defn factorial [n] (if (<= n 1) 1 (* n (factorial (dec n)))))",
        # Lambda notation
        "(λ x. (+ x 1))",
        "(λ f. (λ x. (f (f x))))",
        # io! with :as
        '(io! :read {:path "data.csv"} :as :text)',
        # Unicode / edge cases
        "Hello, 世界! 🌍",
        "α=1.18, fixed_point=40.0",
    ]
    for ex in examples:
        ids = encode(ex)
        roundtrip = decode(ids)
        ok = "✓" if roundtrip == ex else "✗"
        print(f"  {ok} {ex!r}")
        print(f"    → {len(ids)} tokens: {ids[:12]}{'...' if len(ids) > 12 else ''}")

    # Test document encoding
    print(f"\nDocument packing:")
    doc1 = encode_document("(+ 3 7)")
    doc2 = encode_document("Hello world")
    packed = doc1 + doc2
    print(f"  doc1: {doc1}")
    print(f"  doc2: {doc2}")
    print(f"  packed: {packed}")
    print(f"  decoded: {decode(packed)!r}")

    print(f"\n{'='*60}")
    print(f"  ✓ Tokenizer smoke test passed")
    print(f"{'='*60}")
