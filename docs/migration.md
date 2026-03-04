# Migration Notes

## Scope

This document tracks storage/protocol and internal architecture changes introduced by the full refactor.

## Completed Migrations

- History schema upgraded to `v6`.
  - top-level persistence includes:
    - `messages`
    - `events`
    - `planner_traces`
    - `entities`
- Event metadata standardized with:
  - `ui_block_id`
  - `display_level`
  - `decision_source`
  - `io_stats`
  - `rewrite_reason`
  - `skipped`
  - `skip_reason`
  - `profile_order_used`
- State persistence unified through `JsonStateStore`.
  - integrated in `HistoryService`, `ConfigService`, `ContextService`
  - transaction-style flush support
  - read cache and write budget instrumentation

## Internal Split (Behavior-Preserving)

- CLI split:
  - `cli_runtime.py`
  - `cli_parser.py`
  - `cli_handlers.py`
  - `cli.py` reduced to thin entrypoint
- Shell split:
  - `shell/orchestrator.py`
  - `shell/planner_adapter.py`
  - `shell/reference_resolution.py`
  - `shell/execution_runtime.py`
  - `shell/event_recorder.py`
  - `services/shell_service.py` kept as facade + integration point
- Shell iteration enhancements:
  - execution facts extraction (`line count / exists / mutation`)
  - command rewrite before execution (line-count reuse)
  - redundant probe step skip
  - trace-level model routing order reuse
- UI split:
  - `ui/view_models.py`
  - `ui/output_renderer.py`
  - `ui/runtime_feedback.py` focused on thinking/loading and fallback signal

## Compatibility Policy

- Functional parity is mandatory and validated by golden tests.
- Command surface stays unchanged (`ai chat/file/code/context/backup/config/shell`).
- Output wording/layout is allowed to evolve (card-like text blocks + status tags).
