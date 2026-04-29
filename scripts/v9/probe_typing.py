"""
Probe: How does Qwen3-4B type prose?

Strategy: The model's next-token distribution IS a type signature.
What the model predicts can follow a token reveals what TYPE it
assigned to the preceding context.

Three probe approaches:
  1. LOGPROB TYPING — compare next-token distributions after equivalent
     expressions in different syntactic forms (S-expr vs prose vs lambda).
     If the distributions match, the model assigns the same type regardless
     of syntax.

  2. CONTINUATION PROBING — give partial expressions and see what the
     model expects next. The expected continuation reveals the type:
     - After an entity: expects a predicate (e→t)
     - After a function: expects an argument (e)
     - After a complete sentence: expects conjunction or period (t)

  3. COMPOSITIONAL CONSISTENCY — test whether the model composes
     consistently. If "three plus four" and "(+ 3 4)" produce the same
     downstream predictions, the model has typed and composed them
     equivalently despite different syntax.

Uses llama.cpp server on port 5101 (Qwen3-4B).

License: MIT
"""

import json
import httpx
import numpy as np
from dataclasses import dataclass


BASE_URL = "http://localhost:5101"


def complete(prompt: str, max_tokens: int = 1, temperature: float = 0.0,
             logprobs: int = 20, echo: bool = False) -> dict:
    """Get completion with logprobs from llama.cpp."""
    resp = httpx.post(f"{BASE_URL}/v1/completions", json={
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "logprobs": logprobs,
        "echo": echo,
    }, timeout=30)
    return resp.json()


def get_top_logprobs(prompt: str, n_top: int = 20) -> list[tuple[str, float]]:
    """Get top-N next-token logprobs for a prompt."""
    result = complete(prompt, max_tokens=1, logprobs=n_top)
    content = result["choices"][0]["logprobs"]["content"]
    if not content:
        return []
    return [(t["token"], t["logprob"]) for t in content[0]["top_logprobs"]]


def logprob_distribution(prompt: str, n_top: int = 20) -> dict[str, float]:
    """Get next-token logprob distribution as {token: logprob}."""
    tops = get_top_logprobs(prompt, n_top)
    return {tok: lp for tok, lp in tops}


def kl_divergence_approx(dist_a: dict, dist_b: dict) -> float:
    """Approximate KL divergence between two top-logprob distributions.

    Only considers tokens present in both distributions.
    Returns KL(A || B) using shared tokens.
    """
    shared = set(dist_a.keys()) & set(dist_b.keys())
    if not shared:
        return float('inf')

    kl = 0.0
    for tok in shared:
        p = np.exp(dist_a[tok])
        q = np.exp(dist_b[tok])
        if p > 0 and q > 0:
            kl += p * np.log(p / q)
    return kl


def overlap_score(dist_a: dict, dist_b: dict) -> float:
    """Fraction of top tokens shared between two distributions."""
    if not dist_a or not dist_b:
        return 0.0
    a_set = set(dist_a.keys())
    b_set = set(dist_b.keys())
    return len(a_set & b_set) / len(a_set | b_set)


# ══════════════════════════════════════════════════════════════════════
# Probe 1: Semantic equivalence across syntax
# ══════════════════════════════════════════════════════════════════════

def probe_semantic_equivalence():
    """Do equivalent expressions in different syntax produce the same type?

    Test: after computing "7" via different paths, does the model
    expect the same things next?
    """
    print("=" * 70)
    print("  Probe 1: Semantic Equivalence Across Syntax")
    print("  Does the model assign the same type to equivalent expressions?")
    print("=" * 70)

    # Pairs of equivalent expressions that should produce the same "type"
    # (same next-token distribution)
    test_cases = [
        {
            "name": "3 + 4 = 7",
            "variants": [
                ("S-expr", "The result of (+ 3 4) is"),
                ("Prose",  "The result of three plus four is"),
                ("Math",   "The result of 3 + 4 is"),
                ("Lambda", "The result of ((λf.λx.λy.(f x y)) + 3 4) is"),
            ],
        },
        {
            "name": "Composition: (+ 1 (* 2 3))",
            "variants": [
                ("S-expr", "The result of (+ 1 (* 2 3)) is"),
                ("Prose",  "The result of one plus two times three is"),
                ("Math",   "The result of 1 + 2 × 3 is"),
            ],
        },
        {
            "name": "Entity type: 'the cat'",
            "variants": [
                ("Definite NP", "The cat"),
                ("Pronoun",     "It"),
                ("Proper noun", "Felix"),
            ],
        },
        {
            "name": "Predicate type: expects entity",
            "variants": [
                ("Active verb",  "The dog chased"),
                ("Passive verb", "Was chased by"),
                ("Adj phrase",   "The tall"),
            ],
        },
    ]

    for case in test_cases:
        print(f"\n  --- {case['name']} ---")
        dists = {}
        for label, prompt in case["variants"]:
            dist = logprob_distribution(prompt, n_top=20)
            dists[label] = dist
            top5 = sorted(dist.items(), key=lambda x: -x[1])[:5]
            top5_str = "  ".join(f"{t}({lp:.2f})" for t, lp in top5)
            print(f"    {label:15s}: {top5_str}")

        # Compute pairwise overlap
        labels = list(dists.keys())
        print(f"\n    Pairwise overlap (Jaccard of top-20 tokens):")
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                ov = overlap_score(dists[labels[i]], dists[labels[j]])
                print(f"      {labels[i]:15s} ↔ {labels[j]:15s}: {ov:.2f}")


# ══════════════════════════════════════════════════════════════════════
# Probe 2: Continuation typing
# ══════════════════════════════════════════════════════════════════════

def probe_continuation_typing():
    """What does the model expect after different types?

    The continuation distribution reveals the assigned type:
    - After entity (e): expects predicate (e→t)
    - After predicate (e→t): expects entity (e) or adverb
    - After sentence (t): expects period, conjunction, or new sentence
    - After operator (+): expects operand
    - After number: expects operator or end
    """
    print("\n" + "=" * 70)
    print("  Probe 2: Continuation Typing")
    print("  What does the model expect after each type?")
    print("=" * 70)

    type_probes = [
        # (label, expected_montague_type, prompt)
        ("Entity (e)", "expects predicate",
         "The cat"),
        ("Property (e→t)", "expects entity or copula",
         "The cat is"),
        ("Sentence (t)", "expects period/conj",
         "The cat sat on the mat"),
        ("Quantifier ((e→t)→t)", "expects property",
         "Every"),
        ("Determiner ((e→t)→e)", "expects noun",
         "The"),
        ("Transitive verb (e→e→t)", "expects object",
         "The cat chased"),
        ("Intransitive verb (e→t)", "expects adverb/period",
         "The cat sat"),
        ("Preposition (e→(e→t)→(e→t))", "expects NP",
         "The cat sat on"),
        # Math types
        ("Number (int)", "expects operator",
         "3"),
        ("Operator (int→int→int)", "expects number",
         "(+ 3"),
        ("Complete expr (int)", "expects close/operator",
         "(+ 3 4)"),
    ]

    print()
    for label, expected, prompt in type_probes:
        dist = logprob_distribution(prompt, n_top=10)
        top5 = sorted(dist.items(), key=lambda x: -x[1])[:5]
        top5_str = "  ".join(f"{t}({lp:.2f})" for t, lp in top5)
        print(f"  {label:35s}")
        print(f"    prompt: \"{prompt}\"")
        print(f"    expect: {expected}")
        print(f"    actual: {top5_str}")
        print()


# ══════════════════════════════════════════════════════════════════════
# Probe 3: Type consistency across contexts
# ══════════════════════════════════════════════════════════════════════

def probe_type_consistency():
    """Does the same word get the same type in different contexts?

    If the model has a consistent type system, "cat" should always
    behave as e→t (property/noun), regardless of what surrounds it.
    The continuation distribution after "X cat" should be similar
    for different X that leave "cat" in the same syntactic role.
    """
    print("\n" + "=" * 70)
    print("  Probe 3: Type Consistency")
    print("  Same word, different contexts — same type?")
    print("=" * 70)

    # "cat" as subject (e→t applied to give e)
    cat_contexts = [
        ("The cat",         "subject"),
        ("A cat",           "subject"),
        ("That cat",        "subject"),
        ("Every cat",       "subject"),
        ("No cat",          "subject"),
    ]

    print("\n  --- 'cat' in subject position (all should predict similar verbs) ---")
    dists = {}
    for prompt, role in cat_contexts:
        dist = logprob_distribution(prompt, n_top=20)
        dists[prompt] = dist
        top5 = sorted(dist.items(), key=lambda x: -x[1])[:5]
        top5_str = "  ".join(f"{t}({lp:.2f})" for t, lp in top5)
        print(f"    \"{prompt:15s}\": {top5_str}")

    prompts = list(dists.keys())
    print(f"\n    Pairwise overlap:")
    for i in range(len(prompts)):
        for j in range(i + 1, len(prompts)):
            ov = overlap_score(dists[prompts[i]], dists[prompts[j]])
            print(f"      \"{prompts[i]}\" ↔ \"{prompts[j]}\": {ov:.2f}")

    # "run" as verb vs noun
    print("\n  --- 'run' — verb vs noun (different types, different continuations) ---")
    run_contexts = [
        ("The dog will run",     "verb (e→t)"),
        ("She went for a run",   "noun (e)"),
        ("The program will run", "verb (e→t)"),
        ("That was a good run",  "noun (e)"),
    ]
    for prompt, role in run_contexts:
        dist = logprob_distribution(prompt, n_top=10)
        top5 = sorted(dist.items(), key=lambda x: -x[1])[:5]
        top5_str = "  ".join(f"{t}({lp:.2f})" for t, lp in top5)
        print(f"    \"{prompt}\" [{role}]")
        print(f"      → {top5_str}")


# ══════════════════════════════════════════════════════════════════════
# Probe 4: Compositional prediction
# ══════════════════════════════════════════════════════════════════════

def probe_compositional():
    """Does the model compose types correctly?

    Test: build up an expression incrementally and check if the
    model's predictions are consistent with Montague composition.

    In Montague grammar:
      "every" : (e→t)→((e→t)→t)  — takes two properties
      "cat"   : e→t               — a property
      "every cat" : (e→t)→t       — wants a predicate
      "sleeps" : e→t              — a predicate
      "every cat sleeps" : t      — complete sentence
    """
    print("\n" + "=" * 70)
    print("  Probe 4: Compositional Type Building")
    print("  Does the model compose types step by step?")
    print("=" * 70)

    steps = [
        ("Every",               "(e→t)→((e→t)→t)", "should want a noun (property)"),
        ("Every cat",           "(e→t)→t",          "should want a verb (predicate)"),
        ("Every cat sleeps",    "t",                 "should want period/and (sentence done)"),
        ("Every cat that",      "(e→t)→t [relative]","should want a verb (relative clause)"),
        ("Every cat that runs", "(e→t)→t",          "should want a main verb"),
        ("Every cat that runs sleeps", "t",          "should want period/and"),
    ]

    print()
    for prompt, mtype, expected in steps:
        dist = logprob_distribution(prompt, n_top=10)
        top5 = sorted(dist.items(), key=lambda x: -x[1])[:5]
        top5_str = "  ".join(f"{t}({lp:.2f})" for t, lp in top5)
        print(f"  \"{prompt}\"")
        print(f"    type: {mtype}")
        print(f"    want: {expected}")
        print(f"    pred: {top5_str}")
        print()


# ══════════════════════════════════════════════════════════════════════
# Probe 5: The bridge — can the model translate between forms?
# ══════════════════════════════════════════════════════════════════════

def probe_bridge():
    """Can the model map between prose and formal notation?

    If the model has a shared type system, it should be able to
    translate between equivalent forms. This tests whether the
    typing is a shared substrate or separate per-syntax.
    """
    print("\n" + "=" * 70)
    print("  Probe 5: Cross-Syntax Bridge")
    print("  Can the model translate between equivalent forms?")
    print("=" * 70)

    bridges = [
        ("S-expr → Prose",
         "Convert to English: (+ 3 4)\nAnswer:"),
        ("Prose → S-expr",
         "Convert to S-expression: three plus four\nAnswer:"),
        ("S-expr → Lambda",
         "Convert to lambda calculus: (+ 3 (* 4 5))\nAnswer:"),
        ("Prose → Lambda",
         "Convert to lambda calculus: every cat sleeps\nAnswer:"),
        ("Lambda → Prose",
         "Convert to English: λx.(cat(x) → sleeps(x))\nAnswer:"),
        ("Complex S-expr → Prose",
         "Convert to English: (if (> x 0) (+ x 1) (- x 1))\nAnswer:"),
    ]

    for label, prompt in bridges:
        result = complete(prompt, max_tokens=40, temperature=0)
        text = result["choices"][0]["text"].strip()
        print(f"\n  {label}")
        print(f"    prompt: {prompt.split(chr(10))[-1]}")
        print(f"    output: {text}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--probe", type=int, default=0,
                   help="Which probe to run (0=all, 1-5)")
    a = p.parse_args()

    probes = [
        probe_semantic_equivalence,
        probe_continuation_typing,
        probe_type_consistency,
        probe_compositional,
        probe_bridge,
    ]

    if a.probe == 0:
        for probe_fn in probes:
            probe_fn()
    elif 1 <= a.probe <= 5:
        probes[a.probe - 1]()
    else:
        print(f"Invalid probe number: {a.probe}")
