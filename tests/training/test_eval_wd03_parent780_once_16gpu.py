from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = PROJECT_ROOT / "scripts/eval_wd03_parent780_once_16gpu.sh"


class EvalWd03Parent780Once16GpuTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.code_root = self.root / "code"
        self.storage_root = self.root / "storage"
        scripts = self.code_root / "scripts"
        scripts.mkdir(parents=True)
        (scripts / "export_trainer_checkpoint_bundle.py").write_text(
            "print('export stub')\n",
            encoding="utf-8",
        )
        generate = self.code_root / "benchmark/generate_eval.py"
        generate.parent.mkdir(parents=True)
        generate.write_text("select_example_shard = True\n", encoding="utf-8")
        shards = self.code_root / "kimodo/evaluation/generate_shards.py"
        shards.parent.mkdir(parents=True)
        shards.write_text("def select_example_shard():\n    return None\n", encoding="utf-8")
        (self.code_root / "kimodo/evaluation/rank_cuda.py").write_text(
            "def pin_local_cuda_device():\n    return '0'\n",
            encoding="utf-8",
        )
        (scripts / "generate_eval_rank.py").write_text(
            "from kimodo.evaluation.rank_cuda import pin_local_cuda_device\n",
            encoding="utf-8",
        )
        self.marker = self.root / "eval.marker"
        fake_bin = self.root / "bin"
        fake_bin.mkdir()
        python = fake_bin / "python3"
        python.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            'printf "%s\\n" "$*" >> "$EVAL_TEST_MARKER"\n'
            "args=(\"$@\")\n"
            "for i in \"${!args[@]}\"; do\n"
            '  if [[ "${args[$i]}" == "--output-run-dir" ]]; then\n'
            '    dest="${args[$((i+1))]}/exports/step-000780000"\n'
            '    mkdir -p "$dest/stats"\n'
            '    : > "$dest/model.pt"\n'
            '    : > "$dest/config.yaml"\n'
            "  fi\n"
            '  if [[ "${args[$i]}" == "benchmark/parse_folder.py" ]]; then\n'
            "    for j in \"${!args[@]}\"; do\n"
            '      if [[ "${args[$j]}" == "--output" ]]; then\n'
            '        out="${args[$((j+1))]}"\n'
            '        mkdir -p "$(dirname "$out")"\n'
            '        printf "%s\\n" "{\"tables\":{\"content\":{}}}" > "$out"\n'
            "      fi\n"
            "    done\n"
            "  fi\n"
            "done\n"
            "exit 0\n",
            encoding="utf-8",
        )
        python.chmod(0o755)
        self.fake_bin = fake_bin

        run = self.storage_root / "runs/v2-1m-hostnet-wd03-from650k"
        (run / "checkpoints").mkdir(parents=True)
        (run / "config.resolved.yaml").write_text("schema_version: 1\n", encoding="utf-8")
        (run / "checkpoints/step-000780000.pt").write_bytes(b"x")
        benchmark = (
            self.storage_root / "yezitao-kimodo-eval-v2/benchmark/stratified-10pct/content"
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

    def _env(self, **extra: str) -> dict[str, str]:
        environment = dict(os.environ)
        environment.update(
            {
                "PATH": f"{self.fake_bin}:{environment.get('PATH', '')}",
                "KIMODO_CODE_ROOT": str(self.code_root),
                "KIMODO_STORAGE_ROOT": str(self.storage_root),
                "KIMODO_EXPORT_PYTHON": str(self.fake_bin / "python3"),
                "EVAL_TEST_MARKER": str(self.marker),
                "KIMODO_EVAL_SKIP_CUDA_CHECK": "1",
                "KIMODO_NNODES": "2",
                "KIMODO_NPROC_PER_NODE": "8",
                "KIMODO_NODE_RANK": "0",
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": "29500",
            }
        )
        environment.update(extra)
        return environment

    def test_missing_master_addr_exits_2(self) -> None:
        result = subprocess.run(
            ["bash", str(LAUNCHER)],
            cwd=PROJECT_ROOT,
            env=self._env(MASTER_ADDR=""),
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("MASTER_ADDR", result.stderr)

    def test_missing_checkpoint_exits_2(self) -> None:
        (
            self.storage_root
            / "runs/v2-1m-hostnet-wd03-from650k/checkpoints/step-000780000.pt"
        ).unlink()
        result = subprocess.run(
            ["bash", str(LAUNCHER)],
            cwd=PROJECT_ROOT,
            env=self._env(),
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("780k parent checkpoint", result.stderr)

    def test_rank0_exports_generates_and_scores(self) -> None:
        result = subprocess.run(
            ["bash", str(LAUNCHER)],
            cwd=PROJECT_ROOT,
            env=self._env(),
            check=False,
            capture_output=True,
            text=True,
            timeout=8,
        )
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        logged = self.marker.read_text(encoding="utf-8")
        self.assertIn("export_trainer_checkpoint_bundle.py", logged)
        self.assertIn("torch.distributed.run", logged)
        self.assertIn("generate_eval_rank.py", logged)
        self.assertNotIn("benchmark/generate_eval.py", logged)
        self.assertIn("--nproc-per-node=8", logged)
        self.assertIn("benchmark/embed_folder.py", logged)
        self.assertIn("benchmark/evaluate_folder.py", logged)
        self.assertIn("benchmark/parse_folder.py", logged)
        final = (
            self.storage_root
            / "eval-results/v2-1m-hostnet-wd03-from650k-parent780k-stratified10pct"
            / "step-000780000"
        )
        self.assertTrue((final / "complete.json").is_file())
        self.assertTrue((final / "summary_rows.json").is_file())
        self.assertIn("parent780 16gpu eval: wrote", result.stdout)

    def test_rank1_generates_but_does_not_score(self) -> None:
        bundle = (
            self.storage_root
            / "eval-exports/v2-1m-hostnet-wd03-from650k-step780k/exports/step-000780000"
        )
        (bundle / "stats").mkdir(parents=True)
        (bundle / "model.pt").write_bytes(b"x")
        (bundle / "config.yaml").write_text("x\n", encoding="utf-8")
        result = subprocess.run(
            ["bash", str(LAUNCHER)],
            cwd=PROJECT_ROOT,
            env=self._env(KIMODO_NODE_RANK="1"),
            check=False,
            capture_output=True,
            text=True,
            timeout=8,
        )
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        logged = self.marker.read_text(encoding="utf-8")
        self.assertNotIn("export_trainer_checkpoint_bundle.py", logged)
        self.assertIn("torch.distributed.run", logged)
        self.assertNotIn("benchmark/embed_folder.py", logged)
        self.assertIn("rank0 scores", result.stdout)

    def test_skips_when_already_complete(self) -> None:
        final = (
            self.storage_root
            / "eval-results/v2-1m-hostnet-wd03-from650k-parent780k-stratified10pct"
            / "step-000780000"
        )
        final.mkdir(parents=True)
        (final / "complete.json").write_text("{}\n", encoding="utf-8")
        result = subprocess.run(
            ["bash", str(LAUNCHER)],
            cwd=PROJECT_ROOT,
            env=self._env(),
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertFalse(self.marker.exists())
        self.assertIn("already complete", result.stdout)


if __name__ == "__main__":
    unittest.main()
