"""Runtime config constants loaded from current v2 profiles."""

from __future__ import annotations

from pathlib import Path

from ai_assistant.paths import get_path_manager
from ai_assistant.services.config_service import ConfigService


_path_manager = get_path_manager(Path(__file__).resolve().parent)
_config_service = ConfigService(_path_manager)
_profile = _config_service.get_active_profile()

API_KEY = _profile.api_key
API_URL = _profile.api_url
MODEL = _profile.model
STREAM = _profile.stream

HISTORY_FILE = _path_manager.history_path
SUMMARY_THRESHOLD = 10
KEEP_RECENT = 3
SUPPORTED_EXT = [
    ".txt",
    ".py",
    ".sh",
    ".md",
    ".json",
    ".yml",
    ".yaml",
    ".conf",
    ".ini",
    ".log",
    ".csv",
    ".c",
    ".h",
]
SENSITIVE_DIRS = [
    "/",
    "/root",
    "/etc",
    "/usr",
    "/bin",
    "/sbin",
    "/lib",
    "/lib64",
    "/sys",
    "/proc",
    "/dev",
    "/boot",
    "/opt",
    "/var",
    "/tmp",
]

BACKUP_DIR = str(_path_manager.backup_dir)
DEFAULT_BACKUP_ROTATION = 5
BACKUP_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
BACKUP_EXTENSION = ".bak"
