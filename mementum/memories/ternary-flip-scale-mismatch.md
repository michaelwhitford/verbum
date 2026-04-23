❌ Ternary flip accumulators must be scale-invariant. Raw gradient
magnitudes have no relationship to flip confidence — one outlier batch
overwhelms hundreds of consistent signals. Use sign(grad) to make each
micro-batch a single vote. Threshold then means "N out of M batches
agreed" — interpretable, bounded, and decoupled from gradient scale.
The same principle applies anywhere discrete decisions are driven by
accumulated continuous signals: the accumulator must normalize the
signal, or the loudest sample dominates.
