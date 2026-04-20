💡 Three-phase architecture is too rigid — no room for other functions

The MontaguLM dedicates ALL capacity to the three Montague primitives
(separate residual streams per phase). But a language model needs room
for world knowledge, discourse tracking, morphology, pragmatics, and
dozens of other functions we can't name.

Standard transformers work because the shared residual stream is a
general substrate. The three primitives use a 2D subspace at 120°.
The other dimensions are available for everything else. Our rigid
architecture eliminated the interference but also the room.

Fix: shared residual + phase-biased heads, not separate streams.

```
═══ shared residual (d_model) ═══════════════════════→
    ↕ read/write      ↕              ↕
  Phase1 heads     Phase2 heads    Phase3 heads
  (type-biased)    (parse-biased)  (apply-biased)
  + FREE heads     + FREE heads    + FREE heads
```

Phase designation by position (early/mid/late), not by hard
separation. The architecture SUGGESTS specialization without
ENFORCING it. Closer to what Pythia-160M actually does.

Rigid version running as baseline on Dolma. Compare in next session.
