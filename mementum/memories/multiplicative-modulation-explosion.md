❌ Multiplicative modulation with shared weights across sequential passes
creates exponential gradient amplification. At gamma=0.05 (reached in
~200 AdamW steps), grad norms hit 3 billion. Three interacting fixes
needed: (1) additive modulation, (2) zero ternary grads before clip,
(3) per-parameter gradient clipping instead of global clip_grad_norm.
Per-param clipping is essential for 55-layer depth — gamma from any
single layer dominates the global norm and starves all other params.
