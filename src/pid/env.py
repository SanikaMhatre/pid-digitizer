"""Minimal .env loader (no extra dependency).  Existing environment variables are never overridden."""
from __future__ import annotations

import os
from pathlib import Path


def parse_env_line(line: str) -> tuple[str, str] | None:
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        return None
    key, val = line.split("=", 1)
    key, val = key.strip().removeprefix("export ").strip(), val.strip()
    if val[:1] in "\"'" and val.count(val[0]) >= 2:
        val = val[1 : val.index(val[0], 1)]
    elif " #" in val:
        val = val.split(" #", 1)[0].strip()
    elif val.startswith("#"):
        val = ""
    return (key, val) if key else None


def load_dotenv(path: str | Path = ".env") -> list[str]:
    """Load KEY=VALUE lines from `path` into os.environ.  Returns the names that were set."""
    p = Path(path)
    if not p.is_file():
        return []
    set_names = []
    for line in p.read_text(encoding="utf-8").splitlines():
        kv = parse_env_line(line)
        if kv and kv[0] not in os.environ and kv[1] != "":
            os.environ[kv[0]] = kv[1]
            set_names.append(kv[0])
    return set_names
