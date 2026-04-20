"""Result analysis — polars-first.

Loads `results.jsonl` files, joins against probe-set ground-truth, and
produces aggregate metrics. Plotting belongs in notebooks (per AGENTS.md
S1 λ record); this module returns DataFrames.

Implementation lands once the first JSONL exists.
"""

from __future__ import annotations

__all__: list[str] = []
