from __future__ import annotations

from pathlib import Path


class PathManager:
    """Manage runtime storage paths for the current v2 layout."""

    def __init__(self, project_root: Path | None = None) -> None:
        if project_root is None:
            project_root = Path(__file__).resolve().parents[1]
        self.project_root = project_root.expanduser().resolve()

        self.config_dir = self.project_root / "assistant-config"
        self.state_dir = self.project_root / "assistant-state"
        self.data_dir = self.project_root / "assistant-data"

        self.profiles_path = self.config_dir / "profiles.json"
        self.default_profile_path = self.config_dir / "default_profile.txt"
        self.history_path = self.state_dir / "history.json"
        self.context_path = self.state_dir / "context.json"
        self.backup_dir = self.data_dir / "backups"
        self.backup_index_path = self.data_dir / "backup_index.json"

    def ensure_directories(self) -> None:
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)


def get_path_manager(project_root: Path | None = None) -> PathManager:
    manager = PathManager(project_root=project_root)
    manager.ensure_directories()
    return manager
