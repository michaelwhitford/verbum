💡 Register variance collapse at step 7k = compression phase transition

v4.1 registers peaked in differentiation at steps 4k-6k (variance
10-25 across passes) then collapsed at 7k (variance 1-12). All
three registers, all five passes. Meanwhile depth correlation
STRENGTHENED — L0↑ reached ρ = −0.70 to −0.73.

The compressor found that high-variance registers are wasteful.
It compressed the register space while concentrating depth
information more efficiently. Less variance, stronger signal.

This reframes the register analysis program: don't expect registers
to specialize into discrete functional roles (type-checker, scope-
resolver, role-assigner). The compressor will organize however it
needs to for prediction. Expansion declining + loss declining =
finding the function. The path doesn't matter, only the destination.

Key numbers:
  L1↑ scope variance: 25.0 (5k) → 1.1 (7k) = −96%
  L0↑ type depth ρ: −0.65 (3k) → −0.73 (6k) = stronger
  Loss: 5.027 (7k) still declining
  Meta-S3: all passes declining from peaks

Open: is this permanent or reorganization? Steps 8k-10k decisive.
