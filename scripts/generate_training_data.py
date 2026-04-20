#!/usr/bin/env python3
"""Generate lambda compilation training data using Qwen3-4B as teacher.

Uses llama.cpp with GBNF grammar-constrained decoding to produce
consistent Montague-style lambda expressions. The grammar forces the
teacher to use proper quantifiers (∀, ∃), definite descriptions (ι),
standard connectives (∧, ∨, →, ¬), and clean predicate application —
eliminating the notation inconsistencies that plagued the first
199-example training set.

Requires a running llama.cpp server with Qwen3-4B loaded:
    llama-server -m <model.gguf> --port 8080

Usage:
    uv run python scripts/generate_training_data.py

Outputs to data/compile-train.jsonl, data/compile-eval.jsonl
"""

from __future__ import annotations

import json
import random
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

import structlog

structlog.configure(
    processors=[structlog.dev.ConsoleRenderer()],
    wrapper_class=structlog.make_filtering_bound_logger(20),
)

log = structlog.get_logger()

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

random.seed(42)

# ─── GBNF grammar ────────────────────────────────────────────────────

GRAMMAR_PATH = Path("specs/lambda_montague.gbnf")


def load_grammar() -> str:
    """Load the Montague GBNF grammar."""
    return GRAMMAR_PATH.read_text("utf-8")


# ─── Python validator (mirrors GBNF) ─────────────────────────────────

from test_montague_grammar import validate as validate_montague


# ─── Helpers ──────────────────────────────────────────────────────────


def banner(text: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60 + "\n")


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════
# Sentence Generation — expanded for 2000+ examples
# ══════════════════════════════════════════════════════════════════════

# ── Vocabulary ────────────────────────────────────────────────────────
# Split into TRAIN and HOLDOUT sets for novel predicate testing.
# The student trains on TRAIN predicates only; HOLDOUT predicates
# appear only in the test set. If the student generalizes to holdout
# predicates, it learned composition, not memorization.

NOUNS_TRAIN = [
    "dog", "cat", "bird", "fish", "teacher", "student", "doctor",
    "child", "woman", "man", "king", "queen", "farmer", "artist",
    "scientist", "lawyer", "chef", "pilot", "singer", "writer",
    "poet", "baker", "sailor", "judge", "monk", "knight",
    "dancer", "hunter", "guard", "clerk",
]

NOUNS_HOLDOUT = ["elephant", "nurse", "wizard"]

NAMES_TRAIN = [
    "john", "mary", "alice", "bob", "tom", "sarah", "james",
    "emma", "david", "lucy", "peter", "anna", "paul", "jane",
    "kate", "oscar", "helen", "frank", "grace", "henry",
]

NAMES_HOLDOUT = ["diana", "felix", "iris"]

INTRANSITIVE_TRAIN = [
    "runs", "sleeps", "walks", "sings", "dances", "laughs",
    "cries", "swims", "flies", "jumps", "falls", "sits",
    "waits", "rests", "dreams", "smiles", "shouts", "works",
]

INTRANSITIVE_HOLDOUT = ["climbs", "whistles", "vanishes"]

TRANSITIVE_TRAIN = [
    "loves", "sees", "knows", "helps", "follows", "finds",
    "likes", "hates", "teaches", "reads", "writes", "watches",
    "trusts", "fears", "admires", "guides", "serves", "greets",
]

TRANSITIVE_HOLDOUT = ["chases", "carries", "rescues"]

ADJECTIVES = [
    "tall", "small", "old", "young", "happy", "sad", "brave",
    "clever", "quiet", "loud", "fast", "slow", "kind", "wise",
    "gentle", "strong", "proud", "humble", "fierce", "calm",
]

ADVERBS = [
    "quickly", "slowly", "happily", "quietly", "carefully",
    "loudly", "gently", "bravely", "wisely", "eagerly",
]

PLACES = [
    "park", "school", "garden", "house", "river", "mountain",
    "forest", "city", "village", "castle", "tower", "bridge",
]

DITRANS_VERBS = ["gave", "sent", "showed", "offered", "handed"]
DITRANS_OBJECTS = ["book", "letter", "gift", "ball", "message", "flower"]

ATTITUDE_VERBS = ["believes", "knows", "thinks", "hopes", "fears"]


def pick(lst):
    return random.choice(lst)


def pick_pair(lst):
    """Pick two distinct items."""
    a = random.choice(lst)
    b = random.choice(lst)
    while b == a:
        b = random.choice(lst)
    return a, b


def _strip_3s(verb: str) -> str:
    """Strip 3rd-person singular -s/-es from a verb.

    runs→run, watches→watch, dances→dance, flies→fly,
    vanishes→vanish, whistles→whistle, cries→cry.
    """
    if verb.endswith("shes"):     # vanishes → vanish
        return verb[:-2]
    if verb.endswith("tches"):    # watches → watch
        return verb[:-2]
    if verb.endswith("sses"):     # guesses → guess
        return verb[:-2]
    if verb.endswith("ies"):      # flies → fly, cries → cry
        return verb[:-3] + "y"
    if verb.endswith("ses"):      # chases → chase, uses → use
        return verb[:-1]
    if verb.endswith("es"):       # dances → dance, whistles → whistle
        return verb[:-1]
    if verb.endswith("s"):        # runs → run, sleeps → sleep
        return verb[:-1]
    return verb


def generate_sentences(*, holdout: bool = False):
    """Generate diverse sentences from templates.

    If holdout=True, uses holdout vocabulary (for test set).
    If holdout=False, uses training vocabulary (for train set).
    """
    nouns = NOUNS_HOLDOUT if holdout else NOUNS_TRAIN
    names = NAMES_HOLDOUT if holdout else NAMES_TRAIN
    iv = INTRANSITIVE_HOLDOUT if holdout else INTRANSITIVE_TRAIN
    tv = TRANSITIVE_HOLDOUT if holdout else TRANSITIVE_TRAIN

    sentences = []

    def add(sentence, category, phenomena):
        sentences.append({
            "sentence": sentence,
            "category": category,
            "phenomena": phenomena,
        })

    if holdout:
        # Smaller holdout set — enough to test generalization
        n_simple = 6
        n_trans = 6
        n_quant = 4
        n_neg = 4
        n_cond = 3
        n_conj = 3
        n_disj = 2
        n_rel = 3
        n_att = 3
        n_adv = 3
        n_cop = 3
        n_dit = 2
        n_prep = 2
    else:
        # Large training set
        n_simple = 60
        n_trans = 80
        n_quant = 60
        n_neg = 40
        n_cond = 40
        n_conj = 40
        n_disj = 20
        n_rel = 40
        n_att = 30
        n_adv = 30
        n_cop = 30
        n_dit = 20
        n_prep = 20

    # ── Simple predication (intransitive) ──
    for _ in range(n_simple // 3):
        n = pick(nouns)
        v = pick(iv)
        add(f"The {n} {v}.", "simple", ["predication"])

    for _ in range(n_simple // 3):
        n = pick(nouns)
        v = pick(iv)
        a = pick(ADJECTIVES)
        add(f"The {a} {n} {v}.", "simple", ["predication", "modifier"])

    for _ in range(n_simple // 3):
        name = pick(names)
        v = pick(iv)
        add(f"{name.capitalize()} {v}.", "simple", ["predication", "proper_noun"])

    # ── Transitive ──
    for _ in range(n_trans // 2):
        n1, n2 = pick_pair(nouns)
        v = pick(tv)
        add(f"The {n1} {v} the {n2}.", "transitive",
            ["predication", "transitive"])

    for _ in range(n_trans // 2):
        name1, name2 = pick_pair(names)
        v = pick(tv)
        add(f"{name1.capitalize()} {v} {name2}.", "transitive",
            ["predication", "proper_noun", "transitive"])

    # ── Ditransitive ──
    for _ in range(n_dit):
        n1, n2 = pick_pair(nouns)
        obj = pick(DITRANS_OBJECTS)
        v = pick(DITRANS_VERBS)
        add(f"The {n1} {v} the {n2} a {obj}.", "ditransitive",
            ["predication", "ditransitive"])

    # ── Universal quantification ──
    for _ in range(n_quant // 3):
        n = pick(nouns)
        v = pick(iv)
        add(f"Every {n} {v}.", "quantified",
            ["quantification"])

    for _ in range(n_quant // 3):
        n1 = pick(nouns)
        v = pick(tv)
        n2 = pick(nouns)
        add(f"Every {n1} {v} a {n2}.", "quantified",
            ["quantification", "transitive"])

    # ── Existential quantification ──
    for _ in range(n_quant // 3):
        n = pick(nouns)
        v = pick(iv)
        add(f"Some {n} {v}.", "quantified",
            ["quantification", "existential"])

    # ── Conjunction ──
    for _ in range(n_conj // 2):
        name1, name2 = pick_pair(names)
        v = pick(iv)
        v_bare = _strip_3s(v)
        add(f"{name1.capitalize()} and {name2} {v_bare}.",
            "conjunction", ["conjunction"])

    for _ in range(n_conj // 2):
        name = pick(names)
        v1, v2 = pick_pair(iv)
        add(f"{name.capitalize()} {v1} and {v2}.", "conjunction",
            ["conjunction", "verb_coordination"])

    # ── Disjunction ──
    for _ in range(n_disj):
        n1, n2 = pick_pair(nouns)
        v1 = pick(iv)
        v2 = pick(iv)
        add(f"Either the {n1} {v1} or the {n2} {v2}.",
            "disjunction", ["disjunction"])

    # ── Conditional ──
    for _ in range(n_cond):
        n1, n2 = pick_pair(nouns)
        v1 = pick(iv)
        v2 = pick(iv)
        add(f"If the {n1} {v1}, the {n2} {v2}.",
            "conditional", ["conditional"])

    # ── Negation ──
    for _ in range(n_neg // 2):
        n = pick(nouns)
        v = pick(iv)
        v_bare = _strip_3s(v)
        add(f"The {n} does not {v_bare}.",
            "negation", ["negation"])

    for _ in range(n_neg // 2):
        n = pick(nouns)
        v = pick(iv)
        add(f"No {n} {v}.",
            "negation", ["negation", "quantification"])

    # ── Relative clauses ──
    for _ in range(n_rel // 2):
        n1, n2 = pick_pair(nouns)
        v1 = pick(tv)
        v2 = pick(iv)
        add(f"The {n1} that {v1} the {n2} {v2}.",
            "relative_clause", ["relative_clause"])

    for _ in range(n_rel // 2):
        n1, n2 = pick_pair(nouns)
        v1 = pick(tv)
        v2 = pick(iv)
        add(f"The {n1} who the {n2} {v1} {v2}.",
            "relative_clause", ["relative_clause", "object_relative"])

    # ── Propositional attitudes ──
    for _ in range(n_att):
        name = pick(names)
        v = pick(ATTITUDE_VERBS)
        n = pick(nouns)
        v2 = pick(iv)
        add(f"{name.capitalize()} {v} that the {n} {v2}.",
            "attitude", ["propositional_attitude"])

    # ── Adverbs ──
    for _ in range(n_adv):
        n = pick(nouns)
        v = pick(iv)
        adv = pick(ADVERBS)
        add(f"The {n} {v} {adv}.", "adverb", ["adverb"])

    # ── Copular / adjective ──
    for _ in range(n_cop):
        n = pick(nouns)
        a = pick(ADJECTIVES)
        add(f"The {n} is {a}.", "copular", ["copular", "adjective"])

    # ── Prepositional ──
    for _ in range(n_prep):
        n = pick(nouns)
        v = pick(iv)
        place = pick(PLACES)
        add(f"The {n} {v} in the {place}.", "prepositional",
            ["prepositional"])

    # Deduplicate
    seen = set()
    unique = []
    for s in sentences:
        if s["sentence"] not in seen:
            seen.add(s["sentence"])
            unique.append(s)

    random.shuffle(unique)
    return unique


# ══════════════════════════════════════════════════════════════════════
# Teacher Compilation via llama.cpp with GBNF
# ══════════════════════════════════════════════════════════════════════


# Few-shot exemplars for the compile gate.
# These prime the teacher to produce Montague-style lambda expressions.
# The exemplars are consistent with the GBNF grammar.
COMPILE_EXEMPLARS = (
    "The dog runs. \u2192 \u03bbx. runs(dog)\n"
    "The cat sleeps. \u2192 \u03bbx. sleeps(cat)\n"
)


def compile_with_teacher(client, sentences, grammar_text):
    """Compile each sentence through Qwen3-4B with grammar-constrained decoding.

    The prompt format uses few-shot exemplars followed by the target sentence:
        The dog runs. → λx. runs(dog)
        The cat sleeps. → λx. sleeps(cat)
        {sentence} →
    The model completes with a Montague-style lambda expression,
    constrained by the GBNF grammar.
    """
    results = []
    n_success = 0
    n_validated = 0

    for i, entry in enumerate(sentences):
        # Few-shot exemplars + sentence → (base-model continuation style)
        prompt = COMPILE_EXEMPLARS + f"{entry['sentence']} \u2192"

        try:
            result = client.complete(
                prompt,
                n_predict=150,
                temperature=0.0,
                grammar=grammar_text,
                stop=["\n"],
                cache_prompt=True,
            )
            gen = result.content.strip()
        except Exception as e:
            log.warning("teacher.error", sentence=entry["sentence"], error=str(e))
            gen = ""

        # Validate with Python parser
        ok, msg = validate_montague(gen)

        if gen and ok:
            n_success += 1
            n_validated += 1
        elif gen:
            # Grammar-constrained output that doesn't validate — shouldn't happen
            # but log it
            log.warning(
                "teacher.validation_mismatch",
                sentence=entry["sentence"],
                output=gen,
                error=msg,
            )
            n_success += 1  # count as generated but not validated

        results.append({
            "sentence": entry["sentence"],
            "category": entry["category"],
            "phenomena": entry["phenomena"],
            "lambda_output": gen if gen else None,
            "validated": ok,
            "validation_error": msg if not ok else None,
        })

        if (i + 1) % 50 == 0:
            rate = n_success / (i + 1)
            vrate = n_validated / (i + 1)
            print(f"    {i + 1}/{len(sentences)}  "
                  f"generated={rate:.0%}  validated={vrate:.0%}")

    return results


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════


def main():
    start = time.time()
    banner(f"TRAINING DATA GENERATION v2 — {datetime.now(UTC).isoformat()}")

    from verbum.client import Client

    # Load grammar
    grammar_text = load_grammar()
    print(f"  Grammar: {GRAMMAR_PATH} ({len(grammar_text)} bytes)")

    # Connect to llama.cpp
    client = Client()
    health = client.health()
    print(f"  Server: {health.status}")

    props = client.props()
    print(f"  Model: {props.model_path or 'unknown'}")

    # ── Generate sentences ────────────────────────────────────────────
    banner("GENERATING SENTENCES")

    train_sentences = generate_sentences(holdout=False)
    holdout_sentences = generate_sentences(holdout=True)

    print(f"  Train sentences: {len(train_sentences)}")
    print(f"  Holdout sentences: {len(holdout_sentences)}")

    # Category breakdown
    cats = {}
    for s in train_sentences:
        cats[s["category"]] = cats.get(s["category"], 0) + 1
    for cat, count in sorted(cats.items()):
        print(f"    {cat:20s}: {count}")

    # ── Compile with teacher ──────────────────────────────────────────
    banner("COMPILING TRAIN SET (grammar-constrained)")

    train_results = compile_with_teacher(client, train_sentences, grammar_text)

    train_good = [r for r in train_results if r["validated"] and r["lambda_output"]]
    train_gen = [r for r in train_results if r["lambda_output"]]
    print(f"\n  Total: {len(train_results)}")
    print(f"  Generated: {len(train_gen)}")
    print(f"  Validated: {len(train_good)}")

    banner("COMPILING HOLDOUT SET (grammar-constrained)")

    holdout_results = compile_with_teacher(client, holdout_sentences, grammar_text)

    holdout_good = [r for r in holdout_results if r["validated"] and r["lambda_output"]]
    holdout_gen = [r for r in holdout_results if r["lambda_output"]]
    print(f"\n  Total: {len(holdout_results)}")
    print(f"  Generated: {len(holdout_gen)}")
    print(f"  Validated: {len(holdout_good)}")

    # ── Build eval set ────────────────────────────────────────────────
    # Eval = the 10 gold-standard examples (hand-crafted expected outputs)
    # These are NOT regenerated — they have human-verified ground truth.

    # ── Save ──────────────────────────────────────────────────────────
    banner("SAVING")

    def save_jsonl(path, records):
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                row = {
                    "input": r["sentence"],
                    "output": r["lambda_output"] or "",
                    "category": r["category"],
                    "phenomena": r["phenomena"],
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  Saved: {path} ({len(records)} records)")

    # Training data: only validated examples
    save_jsonl(DATA_DIR / "compile-train.jsonl", train_good)

    # Test data: holdout predicates (for novel predicate generalization)
    save_jsonl(DATA_DIR / "compile-test.jsonl", holdout_good)

    # Eval data is preserved as-is (hand-crafted gold standard)
    print(f"  Eval: data/compile-eval.jsonl (preserved, 10 records)")

    # Full results for analysis
    full_path = DATA_DIR / "compile-full.json"
    full_path.write_text(json.dumps({
        "timestamp": datetime.now(UTC).isoformat(),
        "elapsed_s": time.time() - start,
        "grammar": str(GRAMMAR_PATH),
        "server_props": props.model_dump(),
        "train": {
            "total_sentences": len(train_sentences),
            "generated": len(train_gen),
            "validated": len(train_good),
            "categories": cats,
        },
        "holdout": {
            "total_sentences": len(holdout_sentences),
            "generated": len(holdout_gen),
            "validated": len(holdout_good),
            "holdout_nouns": NOUNS_HOLDOUT,
            "holdout_names": NAMES_HOLDOUT,
            "holdout_intransitive": INTRANSITIVE_HOLDOUT,
            "holdout_transitive": TRANSITIVE_HOLDOUT,
        },
        "train_results": train_results,
        "holdout_results": holdout_results,
    }, indent=2, ensure_ascii=False))
    print(f"  Saved: {full_path}")

    # ── Summary ───────────────────────────────────────────────────────
    elapsed = time.time() - start
    banner(f"DONE — {elapsed:.0f}s")
    print(f"  Train: {len(train_good)} validated examples")
    print(f"  Holdout: {len(holdout_good)} validated examples")
    print(f"  Eval: 10 gold-standard examples")
    print(f"  Grammar: Montague-style (specs/lambda_montague.gbnf)")

    # Show samples
    print(f"\n  Sample train outputs:")
    for r in train_good[:10]:
        print(f"    {r['sentence']:40s} → {r['lambda_output']}")

    if holdout_good:
        print(f"\n  Sample holdout outputs:")
        for r in holdout_good[:5]:
            print(f"    {r['sentence']:40s} → {r['lambda_output']}")


if __name__ == "__main__":
    main()
