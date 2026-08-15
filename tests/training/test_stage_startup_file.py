from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = PROJECT_ROOT / "scripts/stage_startup_file.py"
SPEC = importlib.util.spec_from_file_location("stage_startup_file", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class StageStartupFileTests(unittest.TestCase):
    def test_stages_exact_bytes_and_reuses_immutable_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "large.jsonl"
            source.write_bytes(b'{"id":"a"}\n' * 1000)
            cache = root / "cache"
            first = MODULE.stage(source, cache)
            second = MODULE.stage(source, cache)
            self.assertEqual(first, second)
            self.assertEqual(first.read_bytes(), source.read_bytes())

    def test_changed_source_identity_publishes_a_new_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "index.jsonl"
            source.write_text("one\n", encoding="utf-8")
            first = MODULE.stage(source, root / "cache")
            source.write_text("two-lines\n", encoding="utf-8")
            second = MODULE.stage(source, root / "cache")
            self.assertNotEqual(first, second)
            self.assertEqual(second.read_text(encoding="utf-8"), "two-lines\n")


if __name__ == "__main__":
    unittest.main()
