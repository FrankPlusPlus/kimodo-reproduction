"""Publication mode for non-secret derived training artifacts."""

from __future__ import annotations

import os
from pathlib import Path


def publish_file(path: str | Path) -> None:
    """Make atomic-temp outputs reusable on shared storage.

    The default 0664 supports a shared Unix group. Deployments needing private
    assets can set ``KIMODO_DERIVED_FILE_MODE=0600``.
    """

    raw = os.environ.get("KIMODO_DERIVED_FILE_MODE", "0664")
    try:
        mode = int(raw, 8)
    except ValueError as error:
        raise ValueError("KIMODO_DERIVED_FILE_MODE must be an octal file mode") from error
    if mode < 0 or mode > 0o666 or mode & 0o111:
        raise ValueError("KIMODO_DERIVED_FILE_MODE must be a non-executable mode <= 0666")
    os.chmod(Path(path), mode)
