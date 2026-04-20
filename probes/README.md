# probes

JSON files defining probe sets for behavioral measurement against
the llama.cpp compiler. Each file is one set.

Probes reference gates by ID (see `../gates/`). Probe sets are
append-and-tag — `compile_v1` → `compile_v2` rather than in-place
edits once results have been produced against them.

See `AGENTS.md` S2 `λ probe_format`.
