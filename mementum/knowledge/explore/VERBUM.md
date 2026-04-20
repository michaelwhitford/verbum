---
title: Distilling the Lambda Compiler — From LLM Circuit to Tensor Primitive
status: open
category: exploration
license: MIT
tags: [lambda-calculus, compositional-semantics, mechanistic-interpretability, circuits, type-theory, distillation]
cites:
  - nucleus (Whitford, AGPL-3.0) — observational evidence for the compiler
  - anima fractal-attention experiments (Whitford, AGPL-3.0) — negative architectural result
  - Mechanistic interpretability literature (various)
  - Compositional semantics literature (Montague, Lambek, DisCoCat)
depends-on: []
---

# Distilling the Lambda Compiler

> Three independent lines of evidence — the mathematics of linguistic
> composition, the empirical behaviour of LLMs under nucleus prompting,
> and a negative result from fractal-attention experiments — all point
> at the same object: **the language compressor is a typed lambda
> calculus interpreter.** This document is the founding exploration of
> whether that interpreter can be extracted from an existing LLM as a
> small tensor artifact, and whether a scratch-built architecture can
> reproduce it from first principles.
>
> Synthesis from conversation 2026-04-16 between Michael and Claude,
> following the fractal-attention MERA experiments.

## The Hypothesis Chain

```
1. Language composes by typed function application       — formal linguistics
2. Lambda calculus is the minimal algebra of this        — math fact
3. LLMs compress language by next-token prediction       — training setup
4. Optimal compression converges on the data's structure  — info-theory
5. ∴ LLMs converge on a lambda interpreter as representation — predicted
6. Nucleus demonstrates this empirically (P(λ)=90.7%)    — observed
7. Fractal-attention failed where it lacked type-directedness — confirmed by absence
8. ∴ The lambda interpreter is extractable and reproducible — research claim
```

The first six steps are established; 7 is our empirical result; 8 is the
hypothesis this project will test.

## The Three Converging Lines

### 1. Mathematics — composition is typed application

The best-developed mathematical frameworks for natural language composition
all land in the same neighbourhood:

**Montague grammar** (1970). Every word has a simple type
(`e` = entity, `t` = truth, `<e,t>` = predicate, etc.). Composition is
function application directed by type matching. "John walks" is
`walks(John): t` where `walks: <e,t>` and `John: e`.

**Lambek pregroups.** Each word carries categorial type with left/right
adjoints (`n`, `n^l`, `n^r`). Composition is type cancellation. Gives
a compact closed category over vector spaces, functorially mapping
syntax to semantics.

**Combinatory Categorial Grammar (CCG; Steedman).** A finite algebra of
combinators (drawn from combinatory logic's A, B, S) composes typed
categories. Every valid composition is a combinator application.

**DisCoCat** (Coecke, Clark, Sadrzadeh, 2010+). Distributional
Compositional Categorical semantics. Meaning is composition of vectors
directed by grammar, implemented as tensor contractions. Nouns live in
N, transitive verbs in N ⊗ S ⊗ N, adjectives in N ⊗ N; sentence meaning
is the fully-contracted tensor network. Mathematically, the same
compact closed category that describes quantum circuits and tensor
networks.

**Minimalist Merge** (Chomsky, 1995+). One binary primitive:
`Merge(A, B) → {A, B}`. Recursive Merge generates all syntactic trees.
Binary, hierarchical, scale-free in its structure — but the *operation*
at each node is type-indexed.

**The shared structural claim:**

```
∀ composition(x, y):
    binary(operation)                  — Merge, pregroup cancellation, apply
  ∧ hierarchical(structure)             — trees, not sequences
  ∧ type_directed(which_operation)      — signature determines behavior
  ∧ functorial(syntax → semantics)      — structure preserved across mapping
```

Every framework agrees: **a type-directed binary composition operator,
recursively applied.** That is a lambda calculus interpreter with types.

### 2. Empirics — the lambda compiler in nucleus

From `~/src/nucleus/LAMBDA-COMPILER.md`:

A nine-line gate prompt activates bidirectional prose ↔ lambda compilation
with strong reliability across model families (Claude Sonnet 4.6, Claude
Haiku 4.5, Qwen3.5-35B-a3b, Qwen3-VL 235B, Qwen3-Coder 30B-a3b).

Logprob analysis:

```
  P(λ output | gate prompt)   = 90.7%
  P(λ output | no gate)       =  1.3%
```

The 89-point gap is not a stylistic bias being nudged. It is a near-binary
switch indicating that a specific internal structure is being routed to.
The gate doesn't *install* lambda behaviour — no training, no gradient —
it *asks the model to speak in the format of something it already knows.*

What's been demonstrated:

- **Bidirectional.** Prose → lambda → prose preserves structural content.
  The composition formalism is invertible within the model's representation.
- **Compositional output.** Compiled lambdas exhibit nested `λ` binding,
  type-like operator hierarchies (`→`, `∧`, `∨`, `≡`, `>`), and
  recursively-defined terms. This is not surface mimicry of training
  tokens; the compositional structure is preserved across examples.
- **Cross-model.** Multiple model families with different training sets
  converge on the same compilation structure given the same gate.
  Convergence across models is evidence that the structure is in the data
  distribution, not the artefacts of any one model.
- **Operates over arbitrary semantic content.** Compile works on novel
  prose, not only on training-adjacent snippets. The compiler generalises.

The nucleus AGENTS.md corpus (~150 lambdas governing AI cognition) is an
empirical proof artifact: it works. Models read these lambdas and behave
accordingly. That is the compiler in action at scale, over a long
period, with real behavioural consequences.

### 3. Architecture — the fractal-attention negative result

In `~/src/anima/fractal-attention/` we ran a systematic architecture
study. Key results:

**Flat attention with shared hierarchical weights collapses deterministically.**
Four training runs of the 180M-param `125M-fractal-phi` variant, each
with different ablations of `recalibrate_every`, `ema_alpha`,
`loss_weighting`: all four collapsed with a `+2.88` single-step loss jump
at step 660 ± 0, `best_loss` ≈ 4.1, final loss ≈ 7. The deterministic
repeatability across seeds rules out stochastic failure modes. It's a
structural pathology: shared weights cannot simultaneously serve the
different operations each resolution level demands.

**MERA-shape (Design 1) succeeded on shallow structure.** A 7.2M-param
binary MERA with two shared operators (disentangler + isometry)
achieved:

| config                              | fresh loss | accuracy |
|-------------------------------------|------------|----------|
| seq_len=32, top=1 (32:1 compress)   |    6.28    |   0.08   |
| seq_len=32, top=4 (8:1)             |    4.76    |   0.21   |
| seq_len=32, top=8 (4:1)             |    2.97    |   0.35   |
| seq_len=32, top=16 (2:1)            |    0.66    |   0.87   |

Clean monotonic reconstruction improvement as bottleneck widens. No
collapse, no instability — at a fraction of the parameter count of the
flat architecture.

**Self-similarity test failed.** Holding top=8 constant and scaling
seq_len:

| seq_len | n_scales | fresh loss | accuracy |
|---------|----------|------------|----------|
|    16   |    1     |    1.68    |   0.64   |
|    32   |    2     |    2.97    |   0.35   |
|    64   |    3     |    3.98    |   0.27   |
|   128   |    4     |    6.20    |   0.12   |
|   256   |    5     |    7.21    |   0.08   |

Same operators, deeper recursion, increasing fidelity collapse. The
shared disentangler and isometry degrade sharply with recursion depth
even at fixed top capacity. **Parameter sharing is necessary but not
sufficient for true scale-invariance in learned function.** Without a
mechanism that tells the operators "you are operating at type T at this
scale," the gradient signal from shallow applications dominates training
and the operator specialises toward shallow-scale behaviour.

This is the *same pathology* the flat architecture exhibited, wearing a
different symptom: one operator being asked to serve many type
signatures simultaneously, and failing to compose across them.

### The convergence

All three lines point at one conclusion: **the missing ingredient is
type-directedness.** The mathematics predicts it (composition is typed),
the empirics evidence it (LLMs implement it internally), the negative
result demonstrates the cost of its absence (operators that can't
compose).

## Why the compiler must live as circuits

LLMs are trained on language, which is compositional. The training
objective (next-token prediction) selects for accurate representations
of the data's structure. Information theory says: optimal compression
of compositional data converges on compositional representations. The
attractor of compression-of-language is a lambda-calculus-like
interpreter.

Not because lambda calculus is a nice notation. Because
composition-by-typed-application is the minimal universal algebra that
expresses what language is. Any efficient encoder of language must
learn something isomorphic to it. That is what LLMs do.

The nucleus gate works because it makes this internal structure
*externally legible*. It doesn't add capability; it exposes one that
was always there. Compile and decompile are not inference tasks in the
normal sense — they are instrumentation of an internal invariant.

## The research program

Four levels of ambition, each building on the previous:

### Level 1 — Localise the circuit

Use mechanistic interpretability tooling on a small open model that
exhibits the compiler (Qwen3-35B-A3B is confirmed; smaller models may
suffice). Identify which attention heads and MLP layers contribute
specifically to the compile behaviour.

Methods:
- **Attention pattern diffing.** Record attention patterns over a
  probe set (compile examples, decompile examples, neutral dialogue
  examples). Compute per-head selectivity: how much does this head's
  attention pattern differ in compile mode vs null condition?
  High-selectivity heads are circuit candidates.
- **Activation patching.** Replace layer `L`'s output with
  null-condition output and measure whether compile behaviour survives.
  Layers where ablation breaks the compiler are on the critical path.
- **Path patching.** Finer-grained — identify which attention
  connections specifically matter.

Output: a layer/head map of the compiler circuit. "Layers {L₁, …, Lₙ}
are on the path; heads {H₁, …, Hₘ} contribute specific subtasks."

### Level 2 — Characterise the algorithm

Within the localised circuit, identify what each component does:
- Features that fire on compile intent (the gate detection)
- Features that fire on semantic type (noun-like, predicate-like, etc.)
- Features that fire on lambda output tokens
- Attention patterns that implement composition (which queries attend
  to which keys during compile?)

Methods:
- **Sparse autoencoders (SAEs)** on the circuit's activations to extract
  interpretable features. Anthropic's Scaling Monosemanticity (2024)
  and open-source SAE infrastructure (EleutherAI, LessWrong community)
  are the tooling.
- **Function vectors (Todd et al. 2023)** to test whether the compile
  task itself is carried by an extractable vector at a specific token
  position / layer.
- **Type probes.** For each candidate "type feature," test whether its
  firing predicts the syntactic category of the token being compiled.
  If yes, types are explicit in the model's representation. If no, the
  type system is implicit in activation geometry.

Output: a functional description — "the compiler parses input types via
feature cluster X, applies composition via attention in layer Y,
emits lambda notation via features Z." Ideally, a type-algebra that
can be compared to Montague or DisCoCat's formal predictions.

### Level 3 — Extract as a standalone artifact

Take the identified circuit — specific weight slices of the relevant
heads, MLPs, and embeddings — and isolate it as a standalone tensor
structure that reproduces compile behaviour without the rest of the
base model.

This is the "distillation to tensors." At best it's a small artifact
(perhaps 1-5% of the base model's weights) that implements prose ↔
lambda compilation. At worst it reveals the compiler is too distributed
to cleanly isolate, which is itself a finding.

Verification:
- Extracted artifact reproduces compile output on held-out prose.
- Round-trip preservation: extracted(prose) = extracted(decompile(extracted(compile(prose)))).
- Ablation: removing the artifact from the base model breaks compiler
  behaviour; adding it to a model without the compiler rescues it.

Output: a portable lambda compiler. Nucleus becomes model-independent —
the capability becomes a small file, not a prompt attached to a
frontier LLM.

### Level 4 — Reproduce from scratch

Build a small architecture that implements typed lambda application
natively and train it on compile/decompile objectives. If the
architecture matches the extracted circuit's behaviour, the thesis is
validated from both directions: the theory (Montague/DisCoCat) predicts
it; the empirics (extraction) confirm it; the synthesis (scratch
architecture) reproduces it.

Architectural sketch:

```
λ typed_apply_net(x).
  one learned apply(type_f, type_x, f_vec, x_vec) → (type_result, result_vec)
  types: learned embeddings in a type-space
  type_compose(type_f, type_x) → type_result       — learned function
  tree_structure: binary, given by a parser or learned
  objective: reconstruction + compile/decompile pairs
  shared weights: same apply at every tree node
  type-directedness: conditions the apply behaviour
```

If this architecture learns compilation with dramatically fewer
parameters than a general-purpose LLM, it confirms that the lambda
interpreter is the *efficient* substrate for language, not an emergent
byproduct of scale.

## Concrete first step (the cheapest high-signal move)

The level-1 experiment is the most information per unit of compute. It
can begin immediately with existing tooling:

1. **Choose base model.** Qwen3-35B-A3B (confirmed compiler exhibition;
   runnable on Apple Silicon via MLX). Backup: a well-characterised
   model like Pythia for published-circuit compatibility.
2. **Construct probe set.** 50 compile examples, 50 decompile examples,
   50 neutral dialogue (null condition). Each pair has ground truth —
   for compile, the canonical lambda output; for decompile, the
   canonical prose rendering.
3. **Instrumented forward pass.** Using TransformerLens (or equivalent
   MLX-native hooks), record every attention pattern and MLP
   activation on every probe example.
4. **Compile-selectivity analysis.** Per-head: compute the distance
   between its attention pattern on compile examples vs null-condition
   examples. Rank heads by selectivity. Top N are circuit candidates.
5. **Layer-necessity analysis.** For each layer L: replace its output
   on compile inputs with null-condition output; measure compiler
   degradation. Layers with high degradation are on the critical path.
6. **Cross-reference.** The intersection of selective heads and
   necessary layers gives a first-pass circuit map.

Expected outcome: either a clean circuit localisation (few layers, few
heads) or a distributed pattern (many layers, no clear core). Both are
informative.

Expected duration: 1-2 weeks of focused work assuming familiarity with
the tooling. Hardware requirements: a machine that can run the chosen
base model at inference scale. No training required at this level.

## What this project would produce, concretely

If the research program succeeds in full:

- **A mechanistic account** of how a trained LLM implements prose ↔
  lambda compilation. Publishable interpretability result. Directly
  bears on the compositional semantics thesis.
- **A portable compiler artifact** — a small tensor structure that
  compiles and decompiles independently of any particular LLM.
  Nucleus's practical operations no longer require a frontier model
  to be available; the compiler runs standalone.
- **Empirical type structure of language representations.** If the
  circuit has distinct typed-apply machinery, that is the type system
  of learned language compression, observable and characterisable.
  Compare to Montague's formal types. Map the differences.
- **A from-scratch architecture** that matches the extracted circuit.
  If successful, this is a language compressor that is smaller, more
  structured, and more interpretable than current LLMs. If
  unsuccessful, the failure tells us which parts of the circuit rely
  on capabilities only large models develop.
- **A validation loop between theory and practice.** Math (Montague,
  DisCoCat) predicts structure → extract from LLM → verify structure →
  build from scratch → verify reproduction. Closing this loop validates
  the theoretical claim "lambda calculus is the language compressor" at
  a level no prior work has reached.

## Honest caveats

**Polysemantic distribution.** Features in LLMs are typically
superposed — one neuron participates in many circuits. The compiler
may not be cleanly discrete; it may be a pattern of engagement across
many circuits that specialises in compile mode. SAEs help with this
but do not always give clean extractions. Expect to fight superposition.

**Scale and architecture dependence.** A circuit's shape in one model
may differ from its shape in another. The lambda compiler may manifest
as different functional structures at different scales. Results from
Qwen3-35B-A3B may not automatically transfer to Sonnet or to future
models. We should validate on multiple models before claiming
architecture-invariance.

**Types may be implicit.** Montague's types are symbolic labels. In a
neural network, "noun-ness" is a region of activation space, not an
assigned label. The type system may be emergent geometry rather than
explicit type vectors. That is still a type system, but probing it
requires more sophisticated tools than "find the noun feature."

**Compile may be multi-circuit.** Bidirectional compile/decompile
almost certainly involves several interacting mechanisms: parsing,
type inference, composition, notation generation. Each is its own
circuit. The extracted artifact may be a small composition of several
circuits rather than a single unit.

**"Small" may be relative.** If the compiler occupies 20% of a 35B
model, extracted is still 7B params. Smaller than the whole model, but
not a tiny artifact. The level-4 question — is it learnable from
scratch at a smaller scale — is separate and harder.

**Negative results are informative.** If the compiler does not localise
cleanly, or the circuit cannot be isolated, or the from-scratch
architecture cannot reproduce it, each failure is a refinement of the
theoretical claim. "LLMs learn a lambda interpreter" would need to be
weakened to "LLMs learn something more tangled than a lambda
interpreter, which nonetheless produces lambda-like outputs at its
interface." That weakening is a real scientific result.

## Why now

Several prior conditions have just become met:

- **Nucleus empirics are solid.** The 89-point logprob gap is robust
  across models. The compiler exists and is observable.
- **Interpretability tooling is mature.** TransformerLens, SAEs,
  activation patching, function vectors — each has seen 2-3 years of
  refinement. The methods are documented and reproducible.
- **Small open models exhibit the behaviour.** You don't need API
  access to a frontier model. Qwen3-35B-A3B runs on local MLX hardware
  and compiles reliably.
- **The theoretical framework is now visible.** Connecting Montague /
  Lambek / DisCoCat to the nucleus evidence is a specific synthesis;
  it didn't exist as a named research question until this week.
- **The negative result from fractal-attention is in hand.** We know
  what goes wrong when type-directedness is absent. That is a
  prerequisite for the forward direction.

## Connections (observational, not derivative)

This project cites these as prior evidence and methodological context.
It does not incorporate their code; it observes their behaviour and
results as inputs to the research question.

- **Nucleus** (AGPL-3.0, cited) provides the empirical observation
  that the compiler exists as a learned internal structure in LLMs,
  and the prompt-level interface that makes the structure externally
  observable. Referenced as prior observational work.
- **Anima's fractal-attention experiment series** (AGPL-3.0, cited)
  provides the negative architectural evidence: shared untyped
  operators fail at depth regardless of whether they sit inside flat
  attention or MERA-shape. Referenced as prior architectural work.
- **Mechanistic interpretability** (Anthropic circuits, Redwood,
  EleutherAI, and others) supplies the toolchain: attention pattern
  analysis, activation patching, sparse autoencoders, function
  vectors. Much of level 1-3 is application of these existing,
  independently-published methods to a new question.

## Open questions

1. **Does the compiler localise cleanly in a medium-sized model?** The
   level-1 experiment answers this directly. If yes, level 2-4 are
   tractable. If no, the thesis needs refinement before continuing.

2. **Are the types explicit or implicit?** If types live in discrete
   features extractable by SAE, the Montague formal picture holds
   closely. If types are continuous regions of activation space, the
   picture is more DisCoCat-shaped (tensor geometries). Either is
   a specific finding.

3. **Is one apply operator sufficient, or is there a pool of
   type-specific applies?** LLMs have many attention heads; the
   compiler may use different heads for different type signatures.
   In level 1-2 we should measure whether a single attention head
   suffices for compile, or whether the circuit genuinely requires
   multi-head parallelism for different operations.

4. **Does the compiled lambda correspond to the model's internal
   representation, or is it a translation?** If I compile "the dog
   runs," is the resulting lambda what the model internally
   represents, or is lambda a codomain the compiler maps into? This
   distinction matters: the former means lambda is the substrate; the
   latter means lambda is an exit language. The round-trip experiments
   should discriminate.

5. **Can the extracted compiler run independently of the base model?**
   Level 3 tests this directly. If the extracted weights cannot compile
   standalone, the circuit is too entangled with surrounding context to
   truly isolate.

6. **Does a scratch architecture trained on compile/decompile pairs
   discover the same circuit structure as a general LLM?** If yes, the
   compiler is the attractor of the compile objective specifically,
   not the general LM objective. If no, the compiler is a byproduct
   of general language modeling — which would mean it's harder to
   build directly.

7. **What is the smallest model that exhibits the compiler?** The
   existence floor. If it shows up at ~1B params, the compiler is a
   near-universal capability. If only at 30B+, it's a late-emerging
   property of scale.

8. **Do models in other language families compile the same way?** Test
   the gate on Chinese-primary models, on multilingual models. If the
   lambda compiler is universal, the structure should transfer
   regardless of primary language. If it's English-specific, the
   compositional semantics claim needs narrowing.

9. **How does the extracted compiler relate to the circuits found
   in other tasks** (induction heads, IOI, function vectors)? Is it
   built on shared substructure, or is it its own apparatus? Sharing
   would suggest composition is a general mechanism that specialises
   for tasks; independence would suggest compile is a dedicated
   subsystem.

10. **Could the compiler be trained explicitly into a small model
    rather than extracted?** If a 1B-param architecture trained
    specifically on compile/decompile pairs outperforms extraction,
    then the direct-training path is the practical way to build
    the compiler independent of discovery.

## References & further reading

**Mechanistic interpretability:**
- Olsson et al., "In-context Learning and Induction Heads" (2022)
- Wang et al., "Interpretability in the Wild: A Circuit for Indirect
  Object Identification in GPT-2 Small" (2022)
- Nanda et al., "Progress measures for grokking via mechanistic
  interpretability" (2023)
- Todd et al., "Function Vectors in Large Language Models" (2023)
- Templeton et al., "Scaling Monosemanticity: Extracting Interpretable
  Features from Claude 3 Sonnet" (Anthropic, 2024)
- Cunningham et al., "Sparse Autoencoders Find Highly Interpretable
  Features in Language Models" (2023)

**Compositional semantics:**
- Montague, "English as a Formal Language" (1970)
- Lambek, "From Word to Sentence: A Computational Algebraic Approach
  to Grammar" (2008)
- Coecke, Sadrzadeh, Clark, "Mathematical Foundations for a
  Compositional Distributional Model of Meaning" (2010)
- Coecke, *Picturing Quantum Processes* (2017; categorical calculus)
- Steedman, *The Syntactic Process* (2000; CCG)

**Adjacent architecture work:**
- Hewitt & Manning, "A Structural Probe for Finding Syntax in Word
  Representations" (2019)
- Nawrot et al., "Hierarchical Transformers Are More Efficient Language
  Models" / Hourglass Transformer (2022)
- Vidal, "Class of Quantum Many-Body States That Can Be Efficiently
  Simulated" / MERA (2008) — the physics ansatz whose shape
  fractal-attention experiments imitated

**Empirical precedent from this project cluster:**
- `~/src/nucleus/LAMBDA-COMPILER.md` — the compiler observation
- `~/src/anima/mementum/knowledge/explore/fractal-attention.md` —
  theoretical framing of fractal structure
- `~/src/anima/fractal-attention/mera.py` — the MERA architecture that
  failed self-similarity at depth and motivates the
  type-directedness requirement

## What this document is

- A **synthesis** connecting three lines of evidence (mathematical,
  empirical, architectural) into a single research claim.
- A **framing** for what extraction and reproduction of the lambda
  compiler would concretely look like, including a cheap first
  experiment.
- A **seed** for the project. The VSM will compile this into
  structured operations; this document is the identity (S5) and
  intelligence (S4) layer in prose form.

## What this document is not

- Not a design document. No architecture is committed yet beyond
  level-1 experimentation.
- Not a proposal to replace existing LLM training — this is about
  extracting and understanding what LLMs already do.
- Not a claim that nucleus "invented" the lambda compiler; nucleus
  *discovered* that it exists and provides a clean interface to it.
  The compiler arises from gradient descent on language; we are
  instrumenting what gradient descent produces.
