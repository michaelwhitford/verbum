# results

One directory per run. Each run directory contains:

- `meta.json`     — run provenance (self-sufficient for reproduction)
- `results.jsonl` — per-probe records, one line per probe
- `logprobs.npz`  — `np.savez_compressed` dict keyed by `probe_id`

Directory name is the `run_id` (e.g. `run_20260417_143022_abc`).

See `AGENTS.md` S2 `λ result_format` and `λ run_provenance`.
