# Documentation Principles

1. One source of truth: avoid duplicate docs and keep canonical instructions in `README.md` or `QUICKSTART.md`, linking out when needed.
2. Keep runtime paths explicit: document which Dockerfile, compose file, and requirements file each deployment path uses.
3. Separate dev vs prod guidance: anything unsafe for production (e.g. `reload=True`, auto `create_all`, local storage) must be labeled clearly.
4. Defaults should be safe: if a config option is optional, the default behavior should fail closed (auth), avoid data loss, and avoid surprise network egress.
5. Document contracts, not internals: for each API endpoint and model type, define inputs/outputs, error modes, and performance expectations.
6. Preserve backwards compatibility: when changing env vars, endpoints, or file paths, add a migration note and keep an alias when feasible.
7. Log for operations: document structured log fields and what to look for in debugging (request IDs, prediction IDs, model IDs, timings).
8. Avoid “magic” setup: every step should be runnable as a command with expected outputs and troubleshooting notes.
9. Optimize for copy/paste: commands should be complete, include required env vars, and avoid “edit X as needed” without concrete examples.
10. Keep docs close to code: if behavior is nuanced, add a short docstring/comment and link to the longer doc.

