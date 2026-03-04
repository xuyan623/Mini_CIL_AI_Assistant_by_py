# Feature Matrix

This matrix is the functional parity baseline for refactor and optimization work.

## CLI Modules

- `chat`
  - free-form prompt response
  - history append
- `file`
  - `ls/read/search/find/rm/rmdir`
- `code`
  - `check/comment/explain/optimize/generate/summarize`
- `context`
  - `set/add/list/ask/clear`
- `backup`
  - `create/status/list/restore/clean`
- `config`
  - `add/switch/list/current/delete/stream/export/import`
- `shell`
  - `run` with planning + step confirmation + dynamic replan

## Cross-cutting Features

- history persistence for commands and events
- planner traces and resolution traces
- entity extraction for file references
- model fallback with runtime feedback
- graceful interrupt (`Ctrl+C` => exit code 130)
- non-interactive mode does not execute shell steps

