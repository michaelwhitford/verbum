# specs

Canonical contract artifacts. Reference-only — not code-generated.

- `llama_server.openapi.yaml` — **hand-curated** spec for the ~6-10
  llama-server HTTP endpoints we actually use. Not a trimmed copy of
  upstream; grows by use (each endpoint added when the client first
  needs it). Pinned to a llama.cpp commit SHA or release tag in
  `info.description`. See S2 `λ spec_artifact`.

- `lambda_*.gbnf` (future) — GBNF grammar for lambda outputs,
  written from observation in this project, independent of any
  external GBNF. See S2 `λ grammar_artifact`.
