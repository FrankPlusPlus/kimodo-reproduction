from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WATCHER = PROJECT_ROOT / "scripts/eval_lastwd1_from750k_stratified_watcher.sh"


class EvalLastWd1From750kWatcherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.code_root = self.root / "code"
        self.storage_root = self.root / "storage"
        scripts = self.code_root / "scripts"
        scripts.mkdir(parents=True)
        self.marker = self.root / "eval.marker"
        downstream = scripts / "eval_company_watcher.sh"
        downstream.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            'printf "%s\\n" "$KIMODO_RUN_DIR|$KIMODO_EVAL_ROOT|$KIMODO_BENCHMARK_ROOT" '
            '> "$EVAL_TEST_MARKER"\n'
            'printf "%s\\n" "$@" >> "$EVAL_TEST_MARKER"\n',
            encoding="utf-8",
        )
        downstream.chmod(0o755)
        sidecar = scripts / "eval_export_hostnet_checkpoints.sh"
        sidecar.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        sidecar.chmod(0o755)
        fake_bin = self.root / "bin"
        fake_bin.mkdir()
        python = fake_bin / "python"
        python.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        python.chmod(0o755)
        self.fake_bin = fake_bin

        benchmark = (
            self.storage_root
            / "yezitao-kimodo-eval-v2/benchmark/stratified-10pct/content"
        )
        benchmark.mkdir(parents=True)
        baseline = (
            self.storage_root
            / "yezitao-kimodo-eval-v2/baselines/official-seed-v1.1/summary_rows.json"
        )
        baseline.parent.mkdir(parents=True)
        baseline.write_text("{}\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_points_at_750k_lastwd_run_and_800k_cadence(self) -> None:
        environment = dict(os.environ)
        environment.update(
            {
                "PATH": f"{self.fake_bin}:{environment.get('PATH', '')}",
                "KIMODO_CODE_ROOT": str(self.code_root),
                "KIMODO_STORAGE_ROOT": str(self.storage_root),
                "EVAL_TEST_MARKER": str(self.marker),
            }
        )
        subprocess.run(
            ["bash", str(WATCHER)],
            cwd=PROJECT_ROOT,
            env=environment,
            check=True,
            timeout=5,
        )
        lines = self.marker.read_text(encoding="utf-8").splitlines()
        run_dir, eval_root, benchmark = lines[0].split("|")
        self.assertTrue(run_dir.endswith("eval-exports/v2-1m-hostnet-lastwd1-from750k"))
        self.assertTrue(
            eval_root.endswith("eval-results/v2-1m-hostnet-lastwd1-from750k-stratified10pct")
        )
        self.assertTrue(benchmark.endswith("benchmark/stratified-10pct"))
        self.assertEqual(lines[1], "--minimum-step")
        self.assertEqual(lines[2], "800000")


if __name__ == "__main__":
    unittest.main()
