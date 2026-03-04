from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import threading
from typing import Any, Callable

from ai_assistant.storage import atomic_write_json, file_lock, safe_load_json

DefaultFactory = Callable[[], Any]
Normalizer = Callable[[Any], Any]
Updater = Callable[[Any], Any]


@dataclass
class StateTransaction:
    read_cache: dict[str, Any] = field(default_factory=dict)
    dirty_payloads: dict[str, Any] = field(default_factory=dict)
    write_count: int = 0
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class JsonStateStore:
    def __init__(self) -> None:
        self._transaction = StateTransaction()
        self._mutex = threading.RLock()

    def _read_path(self, path: Path, default_factory: DefaultFactory, normalizer: Normalizer | None) -> Any:
        raw_payload = safe_load_json(path, default_factory())
        payload = raw_payload if normalizer is None else normalizer(raw_payload)
        return payload

    def read_json(
        self,
        path: Path,
        default_factory: DefaultFactory,
        normalizer: Normalizer | None = None,
    ) -> Any:
        with self._mutex:
            path_key = str(path.resolve())
            if path_key in self._transaction.dirty_payloads:
                return deepcopy(self._transaction.dirty_payloads[path_key])

            cached = self._transaction.read_cache.get(path_key)
            if cached is not None:
                return deepcopy(cached)

            lock_path = path.with_suffix(path.suffix + ".lock")
            with file_lock(lock_path):
                payload = self._read_path(path, default_factory, normalizer)
            self._transaction.read_cache[path_key] = deepcopy(payload)
            return deepcopy(payload)

    def update_json(
        self,
        path: Path,
        updater: Updater,
        default_factory: DefaultFactory,
        normalizer: Normalizer | None = None,
    ) -> Any:
        with self._mutex:
            current = self.read_json(path, default_factory, normalizer)
            updated = updater(deepcopy(current))
            next_payload = updated if updated is not None else current
            path_key = str(path.resolve())
            self._transaction.dirty_payloads[path_key] = deepcopy(next_payload)
            self._transaction.read_cache[path_key] = deepcopy(next_payload)
            return deepcopy(next_payload)

    def set_json(
        self,
        path: Path,
        payload: Any,
        normalizer: Normalizer | None = None,
    ) -> Any:
        with self._mutex:
            next_payload = payload if normalizer is None else normalizer(payload)
            path_key = str(path.resolve())
            self._transaction.dirty_payloads[path_key] = deepcopy(next_payload)
            self._transaction.read_cache[path_key] = deepcopy(next_payload)
            return deepcopy(next_payload)

    def flush(self) -> int:
        with self._mutex:
            write_count = 0
            for path_key, payload in list(self._transaction.dirty_payloads.items()):
                path = Path(path_key)
                lock_path = path.with_suffix(path.suffix + ".lock")
                with file_lock(lock_path):
                    atomic_write_json(path, payload)
                write_count += 1
            self._transaction.write_count += write_count
            self._transaction.dirty_payloads.clear()
            return write_count

    def reset_transaction(self) -> None:
        with self._mutex:
            self._transaction = StateTransaction()

    def get_io_stats(self) -> dict[str, Any]:
        with self._mutex:
            return {
                "write_count": int(self._transaction.write_count),
                "dirty_count": int(len(self._transaction.dirty_payloads)),
                "cache_entries": int(len(self._transaction.read_cache)),
                "started_at": self._transaction.started_at,
            }
