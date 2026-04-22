"""VSM-LM v6 — Ternary Stacked Compressors.

v6 replaces fp16 linear layers in S1 operations with BitLinear
(ternary {-1, 0, +1} weights, trained with STE). The key architectural
change: instead of multi-stride attention in a single layer, each stride
gets its own ternary attention layer, stacked sequentially.

Ternary weights unlock depth: 8 ternary attention layers use less memory
than 2 fp16 layers, but provide 4× the compositional depth. Each stride
layer does one thing (attend at one scale), which is simple enough for
{-1, 0, +1} weights to express.

Same VSM meta-structure: 5 passes, bidirectional, complex registers,
phase-coherent gating, multiplicative modulation.

License: MIT
"""
