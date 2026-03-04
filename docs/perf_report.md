# Performance Report

This document records measurable outcomes after the v4 refactor.

## Baseline

- Functional tests: `63 passed`
- Coverage: `63%`
- Known hotspots:
  - `HistoryService.load_payload`
  - `storage.atomic_write_json`
  - `storage.atomic_write_text`
  - `ShellService._plan_from_description`

## Current Results

- Full test run: `181 passed` (`python -m pytest -q -o addopts=''`)
- Coverage gate: `93.73%` (`python -m pytest -q`, `--cov-fail-under=90` passed)
- Perf test suite: `3 passed in 0.32s` (`python -m pytest tests/perf -q -o addopts=''`)

## Performance Validation Mapping

- State I/O budget:
  - `tests/perf/test_state_io_budget.py`
  - verifies `HistoryService.load_payload` does not force write-backs
  - verifies append write count stays within budget
- Shell planning latency:
  - `tests/perf/test_shell_planning_latency.py`
  - validates average planning latency budget under mocked AI path

## v5 Iteration Optimizations

- Execution fact reuse:
  - `wc -l` step result is reused to rewrite later `--end` arguments.
  - after write operations (`code comment/optimize/generate`), cached line counts are invalidated.
- Redundant probe elimination:
  - repeated `test -f` and already-resolved `find -name` steps are skipped.
  - skip decisions are persisted in `shell_step.metadata.skipped/skip_reason`.
- Model fallback routing:
  - gateway records `preferred_profile_id` after first success.
  - same shell trace reuses `profile_order_used` to avoid restarting from repeatedly failing profiles.
  - failure diagnostics now include per-attempt `error_preview`.

## Notes

- The suite now uses budget-style performance assertions, which are stable across platforms and CI variance.
- cProfile artifacts can be generated on demand with:
  - `python -m cProfile -o .tmp_profile.prof -m pytest -q`
