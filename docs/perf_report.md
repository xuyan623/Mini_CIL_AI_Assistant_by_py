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

- Full test run: `151 passed` (`python -m pytest`)
- Coverage gate: `94.16%` (`--cov-fail-under=90` passed)
- Perf test suite: `3 passed in 0.42s` (`python -m pytest tests/perf -q -o addopts=''`)

## Performance Validation Mapping

- State I/O budget:
  - `tests/perf/test_state_io_budget.py`
  - verifies `HistoryService.load_payload` does not force write-backs
  - verifies append write count stays within budget
- Shell planning latency:
  - `tests/perf/test_shell_planning_latency.py`
  - validates average planning latency budget under mocked AI path

## Notes

- The suite now uses budget-style performance assertions, which are stable across platforms and CI variance.
- cProfile artifacts can be generated on demand with:
  - `python -m cProfile -o .tmp_profile.prof -m pytest -q`
