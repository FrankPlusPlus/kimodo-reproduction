from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = PROJECT_ROOT / "scripts/company_start_demo_ui.sh"

import importlib.util

MODULE_PATH = PROJECT_ROOT / "kimodo/demo/local_bundles.py"
SPEC = importlib.util.spec_from_file_location("kimodo_demo_local_bundles", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_export(root: Path, run: str, step: int) -> Path:
    dest = root / run / "exports" / f"step-{step:09d}"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "config.yaml").write_text("model: {}\n", encoding="utf-8")
    (dest / "model.pt").write_bytes(b"pt")
    (dest / "stats").mkdir(exist_ok=True)
    (dest / "stats" / "dummy.json").write_text("{}\n", encoding="utf-8")
    return dest


def _write_trainer(root: Path, run: str, step: int, *, readable: bool = True) -> Path:
    run_dir = root / run
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.resolved.yaml").write_text("model: {}\n", encoding="utf-8")
    path = ckpt_dir / f"step-{step:09d}.pt"
    path.write_bytes(b"trainer")
    if not readable:
        path.chmod(0o000)
    return path


class ParseLocalBundlesTests(unittest.TestCase):
    def test_parses_equals_and_comma_separated_env(self) -> None:
        bundles = MODULE.parse_local_bundles(
            "local-soma-wd03-700k=/exports/700k,local-soma-kf695k=/exports/695k"
        )
        self.assertEqual(
            bundles,
            {
                "local-soma-wd03-700k": "/exports/700k",
                "local-soma-kf695k": "/exports/695k",
            },
        )

    def test_cli_extra_overrides_same_label(self) -> None:
        bundles = MODULE.parse_local_bundles("a=/old", extra=["a=/new", "b=/other"])
        self.assertEqual(bundles, {"a": "/new", "b": "/other"})

    def test_rejects_missing_separator(self) -> None:
        with self.assertRaises(ValueError):
            MODULE.parse_local_bundles("not-a-pair")


class DiscoverLocalBundlesTests(unittest.TestCase):
    def test_collects_exports_and_unexported_trainer_ckpts(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            exports = tmp / "eval-exports"
            runs = tmp / "runs"
            ready_700 = _write_export(exports, "v2-1m-hostnet-wd03-from650k", 700_000)
            ready_650 = _write_export(exports, "v2-1m-hostnet-kf-smooth-lr1e5", 650_000)
            ready_695 = _write_export(exports, "v2-1m-hostnet-kf-smooth-lr1e5-step695k", 695_000)
            trainer_730 = _write_trainer(runs, "v2-1m-hostnet-wd03-from650k", 730_000)
            trainer_700 = _write_trainer(runs, "v2-1m-hostnet-wd03-from650k", 700_000)
            trainer_650 = _write_trainer(runs, "v2-1m-hostnet-kf-smooth-lr1e5", 650_000)

            bundles = MODULE.collect_local_bundles(
                export_roots=[exports],
                run_roots=[runs],
                auto_discover=True,
            )
            self.assertEqual(
                list(bundles),
                [
                    "wd03 730k (export on load)",
                    "wd03 700k",
                    "kf-smooth 695k",
                    "kf-smooth 650k",
                ],
            )
            self.assertEqual(bundles["wd03 700k"], str(ready_700))
            self.assertEqual(bundles["kf-smooth 650k"], str(ready_650))
            self.assertEqual(bundles["kf-smooth 695k"], str(ready_695))
            self.assertEqual(bundles["wd03 730k (export on load)"], str(trainer_730))
            self.assertNotIn(str(trainer_700), bundles.values())
            self.assertNotIn(str(trainer_650), bundles.values())

    def test_aliases_wd03_lr3e6_run(self) -> None:
        self.assertEqual(MODULE.run_alias("v2-1m-hostnet-wd03-from780k-lr3e6"), "wd03-lr3e6")
        self.assertEqual(
            MODULE.bundle_label("v2-1m-hostnet-wd03-from780k-lr3e6", 785_000, ready=False),
            "wd03-lr3e6 785k (export on load)",
        )

    def test_explicit_labels_override_discovered(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            exports = tmp / "eval-exports"
            _write_export(exports, "v2-1m-hostnet-wd03-from650k", 700_000)
            bundles = MODULE.collect_local_bundles(
                export_roots=[exports],
                run_roots=[],
                explicit={"wd03 700k": "/custom"},
                auto_discover=True,
            )
            self.assertEqual(bundles["wd03 700k"], "/custom")

    def test_auto_discover_can_be_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            exports = tmp / "eval-exports"
            _write_export(exports, "v2-1m-hostnet-wd03-from650k", 700_000)
            bundles = MODULE.collect_local_bundles(
                export_roots=[exports],
                run_roots=[],
                explicit={"only": "/x"},
                auto_discover=False,
            )
            self.assertEqual(bundles, {"only": "/x"})

    def test_default_roots_come_from_storage_env(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            exports = tmp / "eval-exports"
            runs = tmp / "runs"
            ready = _write_export(exports, "v2-1m-hostnet-wd03-from650k", 700_000)
            trainer = _write_trainer(runs, "v2-1m-hostnet-wd03-from650k", 725_000)
            with patch.dict(
                os.environ,
                {
                    "KIMODO_STORAGE_ROOT": str(tmp),
                    "KIMODO_DEMO_EXPORT_ROOTS": "",
                    "KIMODO_DEMO_RUN_ROOTS": "",
                    "KIMODO_DEMO_AUTO_DISCOVER": "1",
                },
                clear=False,
            ):
                bundles = MODULE.collect_local_bundles(auto_discover=True)
            self.assertEqual(bundles["wd03 700k"], str(ready))
            self.assertEqual(bundles["wd03 725k (export on load)"], str(trainer))


class EnsureInferenceBundleTests(unittest.TestCase):
    def test_returns_exported_dir_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            dest = _write_export(Path(raw), "run", 1)
            self.assertEqual(MODULE.ensure_inference_bundle(dest), dest.resolve())

    def test_exports_trainer_checkpoint_once(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            ckpt = _write_trainer(tmp / "runs", "v2-1m-hostnet-wd03-from650k", 730_000)
            cache = tmp / "eval-exports"
            exported: list[Path] = []

            def fake_export(*, checkpoint, resolved_config, output_run_dir, step, force):
                dest = _write_export(output_run_dir.parent, output_run_dir.name, step)
                exported.append(dest)
                self.assertEqual(Path(checkpoint), ckpt.resolve())
                self.assertTrue(Path(resolved_config).is_file())
                self.assertFalse(force)
                return dest

            dest = MODULE.ensure_inference_bundle(ckpt, export_cache=cache, exporter=fake_export)
            self.assertTrue(MODULE.is_exported_bundle(dest))
            self.assertEqual(len(exported), 1)
            again = MODULE.ensure_inference_bundle(ckpt, export_cache=cache, exporter=fake_export)
            self.assertEqual(again, dest)
            self.assertEqual(len(exported), 1)


class SkinningAndVersionOptionsTests(unittest.TestCase):
    def test_official_soma_and_local_labels_use_soma_skin(self) -> None:
        locals_ = {"wd03-lr3e6 785k (export on load)": "/tmp/x"}
        self.assertTrue(MODULE.uses_soma_visuals("kimodo-soma-seed"))
        self.assertTrue(MODULE.uses_soma_visuals("wd03-lr3e6 785k (export on load)", locals_))
        self.assertFalse(MODULE.uses_soma_visuals("kimodo-g1-seed", locals_))
        self.assertEqual(
            MODULE.skinning_mesh_mode("kimodo-soma-seed"),
            "soma_skin",
        )
        self.assertEqual(
            MODULE.skinning_mesh_mode(
                "wd03-lr3e6 785k (export on load)",
                local_labels=locals_,
            ),
            "soma_skin",
        )
        self.assertEqual(
            MODULE.skinning_mesh_mode(
                "wd03-lr3e6 785k (export on load)",
                use_soma_layer=True,
                local_labels=locals_,
            ),
            "soma_layer_skin",
        )
        self.assertEqual(MODULE.skinning_mesh_mode("kimodo-g1-seed"), "g1_stl")
        with self.assertRaises(ValueError):
            MODULE.skinning_mesh_mode("unknown-model")

    def test_version_dropdown_merges_training_only_for_seed_soma(self) -> None:
        official = ["Kimodo-SOMA-SEED-v1", "Kimodo-SOMA-SEED-v1.1"]
        locals_ = ["wd03 700k", "wd03-lr3e6 785k (export on load)"]
        self.assertEqual(
            MODULE.version_options_with_training(
                official,
                skeleton_key="SOMA",
                dataset_ui_label="SEED",
                local_labels=locals_,
            ),
            official + locals_,
        )
        self.assertEqual(
            MODULE.version_options_with_training(
                official,
                skeleton_key="G1",
                dataset_ui_label="SEED",
                local_labels=locals_,
            ),
            official,
        )
        self.assertEqual(
            MODULE.version_options_with_training(
                official,
                skeleton_key="SOMA",
                dataset_ui_label="Rigplay",
                local_labels=locals_,
            ),
            official,
        )


class DemoLauncherTests(unittest.TestCase):
    def test_launcher_auto_discovers_exports_and_runs(self) -> None:
        text = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("Kimodo-SOMA-SEED-v1.1", text)
        self.assertIn("KIMODO_DEMO_EXPORT_ROOTS", text)
        self.assertIn("KIMODO_DEMO_RUN_ROOTS", text)
        self.assertIn("eval-exports", text)
        self.assertIn("TEXT_ENCODER_DEVICE", text)
        self.assertIn(".venv-kimodo-demo", text)
        self.assertIn("PYTHONUNBUFFERED", text)
        self.assertIn("127.0.0.1:7993", text)
        self.assertIn("ssh -L", text)
        self.assertIn("-m kimodo.demo", text)
        self.assertNotIn("local-soma-wd03-700k", text)
        self.assertNotIn("local-soma-kf695k", text)
        self.assertNotIn("sparse_keyframes_max: 7", text)

    def test_ui_has_one_version_dropdown_not_a_second_training_picker(self) -> None:
        text = (PROJECT_ROOT / "kimodo/demo/ui.py").read_text(encoding="utf-8")
        self.assertNotIn("Training checkpoint", text)
        self.assertNotIn("Load training checkpoint", text)
        self.assertNotIn("gui_local_selector", text)
        self.assertIn("version_options_with_training", text)
        self.assertIn("uses_soma_visuals", text)
        self.assertIn("MOTION_CORRECTION_AVAILABLE", text)

    def test_demo_defaults_postprocess_off_without_motion_correction(self) -> None:
        cfg = (PROJECT_ROOT / "kimodo/demo/config.py").read_text(encoding="utf-8")
        self.assertIn("def motion_correction_available", cfg)
        self.assertIn("INIT_POSTPROCESSING = MOTION_CORRECTION_AVAILABLE", cfg)

    def test_motion_correction_available_false_when_missing(self) -> None:
        import sys
        from unittest.mock import patch

        def available() -> bool:
            try:
                from motion_correction import motion_postprocess  # noqa: F401
            except ImportError:
                return False
            return True

        with patch.dict(sys.modules, {"motion_correction": None}):
            self.assertFalse(available())

    def test_generate_reuses_one_notification_and_ignores_reentry(self) -> None:
        ui = (PROJECT_ROOT / "kimodo/demo/ui.py").read_text(encoding="utf-8")
        gen = (PROJECT_ROOT / "kimodo/demo/generation.py").read_text(encoding="utf-8")
        state = (PROJECT_ROOT / "kimodo/demo/state.py").read_text(encoding="utf-8")
        self.assertIn("generating: bool = False", state)
        self.assertIn("def try_begin_generation", gen)
        self.assertIn("try_begin_generation(session)", ui)
        self.assertIn('generating_notif.title = "Generation failed!"', ui)
        installer = (PROJECT_ROOT / "scripts/company_install_demo_motion_correction.sh").read_text(encoding="utf-8")
        self.assertIn("MotionCorrection", installer)
        self.assertIn("pip install", installer)

        class Session:
            generating = False

        session = Session()

        def try_begin(session):
            if session.generating:
                return False
            session.generating = True
            return True

        def finish(session):
            session.generating = False

        self.assertTrue(try_begin(session))
        self.assertFalse(try_begin(session))
        finish(session)
        self.assertTrue(try_begin(session))


if __name__ == "__main__":
    unittest.main()
