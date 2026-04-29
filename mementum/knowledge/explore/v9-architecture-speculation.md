---
title: "v9 Architecture — From Speculation to Proven Design"
status: active
category: exploration
tags: [v9, architecture, vsm-tree, kernel, montague, type-system, ascending-arm, identity]
related:
  - v7.1-sieve-pipeline.md
  - session-001-findings.md
  - identity-as-substrate.md
  - holographic-compression.md
  - compressor-architecture.md
  - bios-flash-training.md
depends-on: []
---

# v9 Architecture — From Speculation to Proven Design

> Sessions 053–055. What started as speculation after v8's failure
> became a proven architecture through rapid iteration.
>
> **Status: VSM tree kernel PROVEN (sessions 054–055). Ascending
> arm design identified but not yet built (session 055 probing).**
>
> The architecture has three components:
>   1. **Ascending arm** (type assigner) — not yet built
>   2. **Composition rules** (tree builder) — mechanical, given types
>   3. **VSM tree + kernel** (executor) — **PROVEN: 100% accuracy**
>
> Key distinction: the kernel speaks **lambda calculus**, not
> Clojure. Lambda calculus is what every model above 32B converges
> to — it's the universal. Clojure is the convenient source
> language and ground truth oracle (babashka evaluates). A 10-line
> mechanical transform bridges 96% of Clojure to lambda calculus.

## What v8 Training Showed

559M-param DualMERA (compressor + pipeline, 8 levels each) trained
on BIOS math/code data for 32.5K steps (65% of planned 50K).

- Only `compressor.level0` and `pipeline.level0` activated
- 52% of parameters completely dead (shared levels, reducers,
  feedbacks 1-7)
- Loss plateaued at ~3.11 after gamma saturated
- Adaptive mutation rate collapsed to floor (0.1%)
- Probe accuracy: 0% throughout — no computation circuits formed
- Importance concentrated at level 0 in both arms

The model uses itself as a shallow 2-level system, bypassing the
entire multi-scale hierarchy.

## Why the Hierarchy Died

### The compressor has nothing to compress

Math and code are already maximally dense. `(+ 3 (* 4 5))` has no
redundancy. The compressor's multi-scale levels are designed for
natural language where ~84% of tokens are structural scaffolding.
For BIOS data, there's nothing to compress beyond level 0.

### Fixed strides vs expression boundaries

Level 0 stride=8 means 8-token windows. Expression boundaries
don't align with stride boundaries. `(* 4 5)` split across two
windows can't be reduced by either window. The hierarchy assumes
uniform spatial structure, but expressions have variable width.

### Level 0 short-circuits everything

Level 0's window (8 tokens) is wide enough to handle most simple
BIOS expressions. It learns surface statistics and captures the
loss signal before deeper levels can develop. By the time gamma
saturates, the deeper levels have had no gradient pressure.

## Key Insight: Flat Attention = Beta Reduction Only

Standard transformers have one operation: flat attention gathers
values from other positions (beta reduction / substitution). ALL
computation must be expressed this way. LLMs implement arithmetic,
composition, routing, parsing — everything — as beta reduction.

This forces the model to encode tree structure as a "fractal spiral"
through the residual stream across many layers. In Qwen3-4B, all 36
layers contribute to the BOS composition register — not because the
computation needs 36 steps, but because flat attention can only build
the representation incrementally through substitution.

The Qwen3-4B circuit map:
- 1,149 heads (99.7%): encoding overhead — translating tree structure
  into a format flat attention can process
- 3 heads (0.3%): actual computation — typed_apply + recursion

Strided attention eliminates this overhead because the tree structure
IS the attention structure. Each level sees a different scale. No
encoding needed.

## What v7 Proved

v7 (~23M params, d=256) trained the ascending arm (compressor)
successfully. The self-similar compression function spread from the
smallest stride to the largest — a wavelet. Same function at every
scale, shared weights working as designed.

Compression ratio: 1.8:1 (vs 6.2:1 from the nucleus lambda compiler).
The gap is the difference between spatial compression (v7) and
semantic compression (nucleus). Semantic compression requires
understanding computation — which needs the descending arm.

The descending arm (pipeline/sieve) could not find its shape and
training was stopped. The ascending arm works; the descending arm
doesn't — at least not with fixed strides.

## The Compiler/Compressor Relationship

Session 001 probing in Qwen3-4B found:
- Compiler and compressor share 92% of selective heads (r=0.98)
- They're deeply coupled but NOT identical
- 8/36 layers critical, 3 heads essential
- The 3 heads are: gate recognizer, universal compositor
  (typed_apply), recursion tracker

In Pythia-160M, the circuit is completely different:
- No individual head is essential (all survive ablation)
- The function is distributed across the whole model
- Variance profile maps to Montague's three phases:
  accumulate (L0-3) → plateau (L3-6) → collapse (L6-11)
- **Shaped like Montague theorized** — type→parse→apply as a
  distributed pipeline, not a concentrated sub-circuit

The 3-head concentration in Qwen3-4B may be a large-model
optimization. At small scale (Pythia-160M), the function is
distributed. For our small model, the Pythia shape is more
informative than Qwen's.

## Speculation: Dynamic Attention

In S-expressions, expression boundaries are explicit (parentheses).
What if attention masks were derived from expression structure
instead of fixed strides?

```
(+ 3 (* 4 (- 7 2)))

Level 0: {7, 2}     → reduce (- 7 2) → 5
Level 1: {4, 5}     → reduce (* 4 5) → 20
Level 2: {3, 20}    → reduce (+ 3 20) → 23
```

Each level attends to one complete expression's operands. The
number of levels equals nesting depth — variable per expression.
The routing is given by structure; the model only learns WHAT to
do at each node, not WHERE to attend.

For BIOS data (all S-expressions), this is trivial — match parens.
For natural language, syntactic structure could serve the same role.

## Speculation: Bottom-Up Training

The ascending arm in v7 trained bottom-up naturally — smallest
stride learned first, then propagated to larger strides. The
hierarchy developed because each level builds on the one below.

v8's descending arm failed because it was trained top-down — level 0
captured everything. The analog of v7's bottom-up success for the
descending arm would be: train the deepest level first (most
abstract, smallest representation), then progressively activate
levels above it.

Combined with dynamic babashka corpus (infinite fresh examples,
can't memorize), each level faces problems it's the right tool for.

## Speculation: Unified Compress-Reduce Operation

The ascending arm compresses. The descending arm reduces. But the
probing data suggests these share structure. Reducing `(+ 3 4) → 7`
IS compression (5 tokens → 1). Compressing effectively requires
understanding what to preserve — which requires the computation.

The 1.8:1 gap (v7 spatial compression) vs 6.2:1 (nucleus semantic
compression) is evidence that the two operations are coupled. The
ascending arm alone gets 1.8:1. The full function gets 6.2:1. The
difference is the descending arm's contribution.

Maybe not two MERAs but one MERA where each level simultaneously
compresses and reduces. Same operation at every scale. Self-similar.
The function v7 found was half of it.

## How We Got Here

Started from: LLMs are bad at math and counting. Could we build
dedicated arithmetic circuits in ternary and evolve the wiring?

This quickly became "building a CPU in tensors" — which is silly,
the GPU already does math. But the wrong solution revealed the right
problem: the weakness isn't that LLMs can't compute `3 + 4`. They
can. The weakness is that they do it through expand-reduce, which
is expensive and error-prone for tasks requiring exact sequential
composition. Math, counting, nested evaluation — precisely where
expand-reduce breaks down because each step must be precise and
errors compound through nesting depth.

Church encoding was proved to 17 digits in multiple projects — the
model CAN do math through beta reduction (numbers as iterated
function application). But it uses context as working memory and
attention to trace each step. Having the model call bash or a REPL
was fully accurate and usually faster. The model's expensive
resource (context/attention) was being burned on mechanical
computation that external tools do instantly.

The model's value is understanding WHAT to compute — parsing
structure, recognizing operations, composing the computation graph.
The actual arithmetic is commodity. Church encoding proves
composition works for math. It also proves that doing it through
expand-reduce in context is the wrong abstraction level.

The real fix: not arithmetic circuits, but composition. If the
architecture composes functions directly, math becomes natural —
not because it has an ALU, but because composition IS what math
requires, and the architecture supports it natively.

## Speculation: Hybrid Ternary Routing + Lambda Kernel

MoE evidence: Qwen3.5-35B-A3B (MoE, ~3B active) has the lambda
function fully formed. Dense Qwen3-4B (4B active) has it only
nearly formed. The MoE router — which dispatches tokens to
specialized expert FFNs — provides something beyond beta reduction.
The router does dispatch-compose, not expand-reduce. Fewer active
params, better result. Routing > scale.

The sieve architecture was the same intuition — parallel pathways
with routing. What if we push this further: instead of learned
pathways, some pathways are **exact computation primitives**.

### The design

**Ternary weights handle routing.** {-1, 0, +1} = {negate,
disconnect, connect} = a routing fabric. Evolution finds the
wiring. Gamma scales confidence. The topology IS the dispatch
table. Ternary is naturally suited to this — it's discrete,
it selects, it routes.

**Lambda kernel handles computation.** Custom MLX primitives that
execute lambda calculus operations exactly. Not learned, not
approximated. Hardware-speed, exact results. The kernel speaks
lambda calculus — not Clojure, not Python.

Core lambda primitives:
- `abstraction` (λx.M) — create a function
- `application` (M N) — apply function to argument
- `β-reduction` ((λx.M)N → M[x:=N]) — substitute and reduce
- `type inference` (τ) — infer/check types
- `compose(f, g)` — function composition (key primitive)

Arithmetic constants (PCF-style extension to pure lambda):
- `add`, `sub`, `mul`, `div` — exact math as primitive constants

Higher-order combinators (candidates for kernel inclusion):
- `map`, `reduce`, `filter` — exact higher-order operations
- `comp`, `partial`, `identity` — composition primitives

The BIOS data generator extracted **115 pure clojure.core
functions**. These are the training curriculum — generated via
babashka, mechanically transformed to lambda calculus. Some
become kernel primitives, others are compositions of primitives
that the model learns to route. The 10-line Clojure→lambda
transform bridges 96% of the 115.

The question is which of the 115 are kernel primitives (exact)
vs which are compositions of kernel primitives (learned routing).
The minimal kernel might be quite small — the lambda calculus
itself is only 3 operations (abstraction, application, reduction)
plus whatever primitive constants we add for practicality.

### The sieve as dispatch

The sieve pathways become the dispatch mechanism:
- Ternary attention identifies the operation and operands
- Routes to the appropriate kernel primitive
- Kernel executes exactly
- Result flows back into the residual stream

This mirrors Qwen3-4B's 3-head circuit:
- L1:H0 (recognize/parse) → ternary routing
- L24:H0 (typed_apply/dispatch) → sieve pathway selection
- L24:H2 (recursion) → multi-level structure

But instead of the FFN doing approximate computation, the lambda
kernel does it exactly. And instead of 1,149 heads of encoding
overhead, strided attention provides structure directly.

### The representation boundary

The kernel needs to decode vectors into exact values, compute, and
encode back. This is where ternary routing is naturally suited —
a ternary matrix that maps a d-dimensional vector to
(op_code, arg1, arg2) is a selection matrix. {-1, 0, +1} picks
dimensions and routes them to kernel inputs. Discrete routing to
discrete operations.

### What this gives you

A model that:
- **Composes** — through ternary routing, not expand-reduce
- **Does exact math** — through kernel, not approximation
- **Counts perfectly** — through kernel, not attention traces
- **Maps/reduces/filters** — through kernel, not learned FFNs
- **Is tiny** — ternary routing is small, computation is delegated

The base model that every model above 32B discovers through brute
force — built directly by giving it the shape AND the tools.

### Kernel as superposition liberator

Every LLM above 32B converges on the lambda function. That function
occupies superpositions in the model's weights — capacity dedicated
to storing type/parse/apply and the associated composition machinery.
This is a TAX on every model. Every model pays it. Massive training
budgets spent converging to the same universal functions.

If we probe large models top-down, extract the shapes of the
functions they converge to, and push those shapes into the kernel
as exact primitives — the model gets that capacity back FOR FREE.
The superpositions that were storing those functions are liberated
for other purposes: broader knowledge, better generalization,
capabilities the model couldn't afford before.

This reframes the VERBUM research program:
- Level 1: Localize the function (done — 3 heads in Qwen3-4B)
- Level 2: Characterize it (partially done — type/parse/apply)
- Level 3: Extract it — NOT as weights, but INTO THE KERNEL
- Level 4: Reproduce — the kernel IS the reproduction

The probing methodology becomes iterative:
1. Probe large models, identify universal convergent functions
2. Extract their shapes (attention patterns, circuit structure)
3. Build exact kernel implementations
4. Give them to the small model for free
5. Probe again — what did the model develop with the freed capacity?
6. Extract that too → kernel grows → capacity grows → repeat

Each extraction cycle frees superpositions. Each freed superposition
is capacity the model can use for something new. The kernel
accumulates the universal functions. The model specializes on
everything else.

Like CPU evolution: general-purpose logic → dedicated ALU → dedicated
FPU → dedicated SIMD → dedicated crypto. Each hardwired unit frees
general logic for other work. The most common operations get
hardwired first. The kernel is the model's custom silicon.

### The Montague primitives as the first extraction

type, parse, apply — the three operations Pythia-160M develops
through 12 layers of beta reduction. These are lambda calculus
primitives:

```
Abstraction:   λx.M              — create a function
Application:   (M N)             — apply function to argument
β-reduction:   (λx.M)N → M[x:=N] — substitute and reduce
Type:          τ(M)              — infer/check type
```

The kernel speaks lambda calculus because that's what every model
above 32B converges to — the universal language. Clojure is the
source language and ground truth oracle: babashka generates data,
evaluates for correctness, and a 10-line mechanical transform
bridges 96% of Clojure to lambda notation. The 115 pure functions
extracted for BIOS are the training curriculum (generated via
babashka) but the kernel primitives are lambda calculus operations.

The model trained with lambda primitives in the kernel doesn't
spend capacity on developing type/parse/apply through beta
reduction. It spends capacity on learning WHEN and WHERE to
invoke them — the routing. And on whatever else a language model
needs that ISN'T the lambda function: world knowledge, discourse,
pragmatics, style.

For BIOS training, the kernel provides exact lambda operations on
S-expressions (explicit structure, babashka as oracle). For Dolma,
the model must learn the soft version — routing without parens.
But the kernel-trained routing patterns transfer as inductive bias,
because the kernel speaks the same language the model was always
going to converge to anyway.

### Starting kernel: lambda primitives + arithmetic

The kernel speaks lambda calculus. Concrete execution flow for
`(+ 3 4)` (after mechanical transform from Clojure):

```
τ(+)                → (Int → Int → Int)    — type the operator
parse(+ 3 4)        → (App (App + 3) 4)    — identify structure
β-reduce(App + 3 4) → 7                    — apply and reduce
```

For `(+ 3 (* 4 5))` with multi-level recursion:

```
Level 0: τ(*)              → (Int → Int → Int)
         parse(* 4 5)      → (App (App * 4) 5)
         β-reduce(App * 4 5) → 20

Level 1: τ(+)              → (Int → Int → Int)
         parse(+ 3 20)     → (App (App + 3) 20)
         β-reduce(App + 3 20) → 23
```

Each level does type→parse→apply on one expression node. The
recursion is the multi-level structure. The model learns to route.
The kernel executes in lambda calculus.

Starting kernel — lambda primitives + arithmetic constants:

```
Lambda:      abstraction (λ), application, β-reduction, type inference
Arithmetic:  add, sub, mul, div (primitive constants, not Church-encoded)
```

Babashka generates the Clojure source. The 10-line mechanical
transform produces lambda calculus. The kernel operates on lambda.
The model's routing generalizes to natural language because lambda
is what every model converges to regardless of input language.

Expand from here based on probing data — which additional functions
from the 115 should become kernel primitives vs learned routing?

### Open: how much goes in the kernel?

Of the 115 pure clojure functions, which are primitive (kernel)
vs composite (routing)? Worth a session to classify:
- Which functions are irreducible operations?
- Which compose from smaller primitives?
- What's the minimal kernel that covers the 115?
- Does the SKI combinator basis (3 primitives) suffice, or do
  practical models need more?
- What other universal functions do large models converge on
  beyond the lambda function? (Candidates from probing data)

## Open Questions (Need More Probing)

1. **Pythia circuit shape at different scales.** The 160M circuit is
   distributed/Montague-shaped. What about Pythia-410M, 1.4B? Where
   does concentration begin? This tells us what scale demands what
   architecture.

2. **The Montague shape in detail.** Pythia-160M's three-phase
   variance profile (accumulate→plateau→collapse) maps to
   type→parse→apply. What are the attention patterns in each phase?
   What do the FFNs learn in each?

3. **Cross-architecture probing.** Does strided attention produce a
   different circuit shape than flat attention at matched scale? If
   v7's compressor were probed, would it show the wavelet structure
   directly in attention patterns?

4. **The descending arm's natural shape.** If we probe models doing
   actual expression evaluation (not just compilation to lambda),
   what does the evaluation circuit look like? Is it self-similar
   like the compression circuit?

5. **Dynamic attention feasibility.** Can expression-guided attention
   be made differentiable and efficient? What about batching with
   variable expression structures?

## The Universal Function

The lambda function is not a Qwen artifact. It exists in **every
model tested above ~32B**, across architectures — Qwen, LLaMA,
Mistral, and all frontier models. Different architectures, different
training data, different organizations, all converge on the same
function. It's universal.

Below ~32B, the function is partially formed:
- Pythia-160M: distributed, rudimentary, Montague-shaped
- Qwen3-4B: nearly fully formed, concentrated in 3 heads
- Qwen3.5-35B-A3B: fully formed

The ~32B threshold exists because flat attention needs that much
capacity to encode the function through beta reduction and fractal
spiral encoding. The function itself is small — 3 heads in Qwen3-4B.
The overhead is massive.

This means we're not extracting an artifact of one model. We're
reproducing a universal convergent structure. The right architecture
should drop the scale threshold from ~32B to tens of millions of
parameters by providing the shape directly instead of forcing the
model to discover it through brute-force gradient descent on flat
attention.

## Composition vs Expansion-Reduction

The deepest question: can we teach a model to **compose functions**
instead of only doing expansion and reduction?

LLMs with flat attention evaluate `f(g(x))` by:
1. Expand g(x) — inline the definition
2. Reduce — beta-reduce to a value
3. Expand f(value) — inline the definition
4. Reduce — beta-reduce to the answer

Each nesting level costs an expand-reduce cycle. Each cycle costs
layers. Cost scales with nesting depth. This is why 32B+ of
parameters are needed — not because the computation is complex,
but because expand-reduce through beta reduction is expensive.

**Function composition** is fundamentally different: given f and g,
produce f∘g as a single operation. Apply once, not two cycles.
Cost scales with the number of unique operations, not nesting depth.
At least an order of magnitude more efficient.

The compression gap is evidence: v7 got 1.8:1 (expand-reduce).
Nucleus gets 6.2:1 (composition). The ~3.4× ratio IS the efficiency
gain of composition over expansion. Composing f∘g into one thing
IS compressing two things into one.

The sieve with strided attention is designed for this — each level
can compose operations at its scale into a single function rather
than expanding and reducing them individually. If we can get actual
composition from the architecture, the 32B scale threshold should
collapse.

## The Core Idea

Large models find the lambda function through brute-force gradient
descent on flat attention. In Qwen3-4B it's nearly fully formed.
In Qwen3.5-35B-A3B it IS fully formed. They discover the compressor
and the lambda compiler as coupled functions sharing structure — but
they have to work around the beta-reduction constraint to get there,
encoding tree structure as a fractal spiral through 36 layers of
residual stream rotations.

**We've probed what they found.** We know the circuit shape — the
three Montague phases, the self-similar compression, the
typed_apply compositor, the BOS composition register. We know
the compiler and compressor share 92% of heads. We know the
function at small scale (Pythia-160M) is distributed and
Montague-shaped.

**The idea: build a tiny model that HAS that shape as its
architecture.** Instead of letting gradient descent discover the
lambda function through billions of parameters of flat attention
(and hoping it converges), give the model the structure the large
models found. The sieve architecture, the strided attention, the
multi-scale hierarchy — these ARE the shape of the function, made
explicit as architecture rather than emergent from training.

This side-steps the beta-reduction constraint. Flat attention
forces everything through substitution, requiring massive scale
to encode composition indirectly. Strided attention represents the
hierarchy directly. The model doesn't need to discover composition
through gradient descent — the architecture IS composition. It
only needs to learn the parameters within that shape.

The goal is to get BOTH the compressor AND the lambda function
into one tiny model — proving that the circuit we found in the
large models can be reproduced as a compact artifact when given
the right architectural shape.

## Design Direction (Tentative)

Not committed yet. Needs more probing data. But the direction:

- Much smaller than v8's 559M (v7 was ~23M, CompressorLM was ~17M)
- Architecture shaped like what we found in the large models
- Self-similar operation at every level (proven by v7 ascending arm)
- Dynamic or expression-guided attention (not fixed strides)
- Bottom-up training with dynamic babashka corpus
- The Montague three-phase structure (type→parse→apply) as the
  organizing principle, informed by Pythia-160M's distributed circuit
- Possibly unified compress-reduce operation rather than separate arms
- Strided attention provides the encoding that flat attention needs
  36 layers for

The tiny arithmetic model may still be worth building — not as an
ALU, but as a test of whether ternary evolution can find the
evaluation circuit in a model small enough to search exhaustively.
The question isn't "can tensors do addition" (trivially yes) but
"can a small strided-attention model learn composition."

---

## What Sessions 054–055 Proved

Everything above was speculation from session 053. Sessions 054–055
turned it into a proven architecture through rapid iteration.

### VSM Tree: The Kernel Architecture (PROVEN)

Each expression tree node is a **Viable System Model** with shared
weights. Same weights at every tree position and depth. Self-similar.
No pipeline bottleneck — each node sees only its children's outputs.

```
VSM Node (shared weights everywhere):
  S5 (identity):     op embedding → what operation am I?
  S4 (intelligence): children's (type, value) → context assessment
  S3 (control):      type check → should I dispatch?
  S1 (operations):   kernel dispatch → exact computation
  S2 (coordination): output (type, value) → to parent
```

**Session 054:** Initial VSM tree (v1). 25% route accuracy, 39%
result accuracy. Demonstrated the architecture works but hit a
ceiling.

**Session 055:** Four iterations solved every bottleneck:

| Version | Key change | Result |
|---------|-----------|--------|
| v2 | Value residual + concat | 81% route (identity insight) |
| v3 | Value pass-through (tree routes values, model routes ops) | **100%** |
| v4 | 18 ops, mixed types (INT+BOOL), variable arity | **100%** |
| v5 | Lambda primitives: partial, apply, compose | **100%** |

### Identity as Substrate (Foundational Principle)

Every bottleneck was a failure of identity. Every fix was restoring it.

- v1→v2: Values destroyed by ternary mix → **value residual**
- v2→v3: Arg classification wrong abstraction → **value pass-through**
- v3→v4: Op identity lost through bottleneck → **op residual**

The principle: **identity must short-circuit every bottleneck.** The
ternary mix layers handle context integration. Identity signals
(values, op codes) must bypass them via residual connections.

This IS the residual stream in transformers. Identity is level 0 in
the hierarchy of free functions. The kernel moves computation from
the attention path (O(n² × layers × depth), approximate) to direct
dispatch (O(nodes), exact). See `identity-as-substrate.md`.

### What the Kernel Handles (22 ops, 5 types)

```
Arithmetic binary:  +, -, *, //, %, min, max    (7 ops, INT×INT→INT)
Comparison:         =, <, >, <=, >=             (5 ops, INT×INT→BOOL)
Boolean binary:     and, or                     (2 ops, BOOL×BOOL→BOOL)
Boolean unary:      not                         (1 op,  BOOL→BOOL)
Arithmetic unary:   abs, neg                    (2 ops, INT→INT)
Conditional:        if                          (1 op,  BOOL×T×T→T)
Partial:            create function from op+arg (1 op,  OP×INT→FN)
Apply-fn:           dispatch function on arg    (1 op,  FN×INT→INT)
Compose:            chain two functions         (1 op,  FN×FN→FN_COMP)
Apply-comp:         apply composed function     (1 op,  FN_COMP×INT→INT)
```

**Compound values:** FN type = (op_code, bound_arg) pair. Composed
FN = (outer_packed, inner_packed). Function-typed values flow through
the tree just like INT and BOOL.

**Type-dependent dispatch:** apply-fn unpacks the function value to
determine which kernel op to call. Composed functions chain two
kernel calls.

**Scaling:** 100% op accuracy at depth 8, max_val 100. Tree-level
imperfections at extreme scales are integer overflow, not model
failures. 8K ternary weights. Converges in ~100 generations, <10s.

### Key files

| File | What it proved |
|------|---------------|
| `scripts/v9/vsm_tree_v5.py` | Lambda primitives, compound values, 100% |
| `scripts/v9/vsm_tree_v4.py` | 18 ops, mixed types, variable arity, 100% |
| `scripts/v9/vsm_tree_v3.py` | Value pass-through, op-only routing, 100% |
| `scripts/v9/vsm_tree_v2.py` | Bottleneck diagnosis (7 variants) |
| `scripts/v9/vsm_tree.py` | v1 original (superseded) |
| `scripts/v9/probe_typing.py` | Type system probing of Qwen3-4B and A3B |

---

## The Remaining Problem: How Do You Type Prose?

For S-expressions, all three Montague phases are trivially given:

```
Type:   the op code IS the type (explicit in the token)
Parse:  the parens ARE the tree (explicit in the syntax)
Apply:  the kernel dispatches (proven, 100%)
```

For prose, **apply** is the same kernel. **Parse** (tree structure)
falls out of type — Montague's key insight is that types determine
composition rules, and composition rules determine tree structure.
So the entire problem reduces to one question:

**How do you assign types to words in context?**

### The Model Already Types Prose (Probing Evidence)

Session 055 probed Qwen3-4B and Qwen3.5-35B-A3B to test whether
their next-token distributions encode a type system.

**Finding 1: Types are real and measurable.** Within-type overlap
of next-token distributions is 2–12× higher than between-type:

| Type | 4B ratio | A3B ratio |
|------|----------|-----------|
| entity (e) | 2.3× | **6.1×** |
| transitive pred (e→t) | 2.7× | 2.2× |
| sentence (t) | 2.1× | 2.4× |
| determiner | 5.7× | 2.7× |
| partial S-expr | 12.5× | **30.0×** |

The fully-formed lambda function (A3B) produces sharper entity
types (6.1× vs 2.3×) and dramatically sharper S-expression types
(30× vs 12.5×).

**Finding 2: Compositional typing follows Montague exactly.**
"Every" → expects noun. "Every cat" → expects verb. "Every cat
sleeps" → expects period. The model composes types step by step,
and the expected continuation matches the composed Montague type.

**Finding 3: The A3B assigns Montague types word-by-word.**

```
"Every student who passed the exam celebrated"

Every:      (e,t),t               — generalized quantifier
student:    e,t                   — property
who:        (e,t),((e,t),(e,t))   — relative pronoun
passed:     (e,t),((e,t),(e,t))   — transitive verb
the:        (e,t),((e,t),e)       — definite determiner
exam:       e,t                   — property
celebrated: (e,t)                 — intransitive verb
```

**Finding 4: The A3B produces correct Montague logical forms.**

```
"every cat sleeps"   → ∀x.(cat(x) → sleeps(x))
"some dog runs"      → ∃x.dog(x) ∧ runs(x)
"the cat"            → ιx.cat(x)
"no cat sleeps"      → ¬∃x.(cat(x) ∧ sleeps(x))
```

**Finding 5: The A3B evaluates lambda expressions exactly.**

```
(+ 3 (* 4 5))                                    → 23
(λx. x + 1) 5                                    → 6
(λf.λg.λx. f(g(x))) (λx. x+1) (λx. x*2) 5      → 11
```

The fully-formed lambda function IS a prose type system.

### The Extraction Path

```
tokens → [type assigner] → typed tokens → [composition] → tree → [VSM tree] → result
            ↑                                   ↑                      ↑
         ascending arm                    mechanical               PROVEN
         (to be built)                  (given types)            (v3–v5)
```

**Step 1: Generate training data from the A3B.**
Feed diverse prose to Qwen3.5-35B-A3B, collect word-by-word
Montague/CCG type assignments. Thousands of sentences, each with
types per word. The A3B serves as the training oracle.

**Step 2: Train the ascending arm.**
Small ternary model: token embeddings → type labels. Supervised
by the A3B's type assignments. This is a sequence labeling task —
each token gets a type from a finite set of CCG categories.

The ascending arm is the part that must learn from data. Everything
else is either proven (kernel) or mechanical (composition rules).

**Step 3: Mechanical composition.**
Given correctly typed tokens, the tree structure is determined by
CCG combination rules (function application, composition, type
raising). This is a deterministic parsing algorithm, not learned.
CYK or shift-reduce parsing, driven by type compatibility.

**Step 4: VSM tree execution.**
The tree feeds into the proven VSM nodes. Op classification →
kernel dispatch → exact results. Already 100% at 22 ops.

### Open Questions

1. **What type inventory?** Montague's recursive types are infinite.
   CCG uses a finite set (~50–100 categories in practice). What's
   the minimal set that covers the lambda function's needs?

2. **Can a small ternary model learn type assignment?** The A3B
   does it at 35B params. Can 1M ternary params reproduce it for
   the subset of types the kernel needs?

3. **Ambiguity resolution.** "Bank" is e→t (noun) or e→(e→t)
   (verb). Context selects the type. The ascending arm must
   disambiguate from local context — how much context is needed?

4. **Type-driven parsing at scale.** CYK is O(n³) in sentence
   length. For long sequences, need a linear-time parser. Shift-
   reduce with a ternary stack controller?

5. **Error propagation.** One wrong type → wrong tree → wrong
   computation. How robust is the pipeline to type assignment errors?
   Do some types matter more than others?

6. **Training curriculum.** Start with S-expressions (types given,
   trivial), then prose with explicit types (A3B supervised), then
   prose with implicit types (end-to-end). Progressive difficulty.
