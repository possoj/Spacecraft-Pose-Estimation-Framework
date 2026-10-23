"""
Copyright (c) 2026 Julien Posso

Behavioral checks for the temporal comparison, using small synthetic tracks.
"""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src.temporal.benchmark_filters import (
    pmf_filter, qdistance, qexp, qlog,
)
from temporal_benchmark import (_component_metrics, ensure_manifest,
                                evaluate, family_of, main, infer)
from src.temporal.sweep import run_sweep, sweep_parameters
from src.temporal.evaluation import run_evaluation, grouped_average, summarize, METRICS


class TemporalBenchmarkTests(unittest.TestCase):
    """
    Verify causal filtering, fixed evaluation and manual-sweep behavior.
    """
    def test_default_and_explicit_stages_run_the_same_fixed_evaluation(self):
        """
        Verify that default/all and evaluate stages dispatch to the same fixed evaluator.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = root / "settings.json"
            settings.write_text(json.dumps({
                component: {"adaptive": {"alpha": 16.49, "n": 0.8},
                            "ema": {"beta": 0.2}, "moving_average": {"window": 10}}
                for component in ("ori", "pos")}))
            for stage, should_infer in (([], True), (["--stage", "all"], True),
                                        (["--stage", "evaluate"], False)):
                with (patch("sys.argv", ["temporal_benchmark.py", *stage,
                                        "--selected-parameters", str(settings),
                                        "--cache-dir", str(root / "cache"),
                                        "--output-dir", str(root / "output")]),
                      patch("temporal_benchmark.sequences", return_value=[root / "S_st"]),
                      patch("temporal_benchmark.load_bins", return_value=({}, {})),
                      patch("temporal_benchmark.manifest_for", return_value={}),
                      patch("temporal_benchmark.infer") as inference,
                      patch("temporal_benchmark.evaluate") as evaluation):
                    main()
                    self.assertEqual(inference.called, should_infer)
                    evaluation.assert_called_once()
                    self.assertEqual(evaluation.call_args.args[0].selected_parameters, settings)

    def test_complete_cache_needs_no_inference_runtime(self):
        """
        Verify that a complete cache is reusable without importing PyTorch.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "S_st.npz").touch()
            with patch.dict("sys.modules", {"torch": None}):
                infer(SimpleNamespace(cache_dir=root), [Path("S_st")], {}, {})

    def test_evaluation_averages_families_and_defines_residual_jitter(self):
        """
        Check family weighting and residual-based jitter on synthetic tracks.
        """
        rows = [dict(sequence=name, family=family, component=component, method=method,
                     **{metric: value for metric in METRICS})
                for name, family, value in (("A_eb", "A", 1.0), ("A_st", "A", 3.0), ("B", "B", 8.0))
                for component in ("ori", "pos")
                for method in ("raw", "moving_average", "ema", "adaptive")]
        families = grouped_average(rows, "family", ("family", "component", "method"))
        results = summarize(rows, families)
        self.assertEqual(results[0]["mean"], 5.0)
        self.assertEqual(results[8]["mean"], 4.0)
        truth = np.zeros((40, 3))
        prediction = np.c_[np.arange(40) * 0.04, np.zeros((40, 2))]
        metrics = _component_metrics(truth, prediction, "pos")
        self.assertAlmostEqual(metrics["jitter"], 1.0)
        self.assertAlmostEqual(metrics["std"], np.std(prediction[:, 0], ddof=0))

    def test_fixed_setting_evaluation_export(self):
        """
        Check settings, metrics, figures, trajectories and overwrite protection.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "cache"
            cache.mkdir()
            (cache / "manifest.json").write_text(json.dumps({"model_sha256": "fixture", "config_sha256": "fixture"}))
            settings = {component: {"adaptive": {"alpha": alpha, "n": n}, "ema": {"beta": 0.2},
                                    "moving_average": {"window": window}}
                        for component, alpha, n, window in (("ori", 16.49, 0.8, 10), ("pos", 48.64, 0.5, 6))}
            parameters = root / "parameters.json"
            parameters.write_text(json.dumps(settings))
            count = 40
            truth_q = np.tile([1.0, 0, 0, 0], (count, 1))
            truth_p = np.tile([0.2, 0, 8.0], (count, 1))
            arrays = {"frame": np.arange(count), "true_ori": truth_q, "pred_ori": truth_q,
                      "true_pos": truth_p, "pred_pos": truth_p,
                      "ori_pdf": np.tile([0.8, 0.2], (count, 1)),
                      "pos_pdf": np.tile([0.8, 0.2], (count, 1))}
            bins = {"ori": np.array([[1.0, 0, 0, 0], [1.0, 0, 0, 0]]),
                    "pos": np.array([[0, 0, 8.0], [1, 0, 8.0]])}
            output = root / "evaluation"
            args = SimpleNamespace(output_dir=str(output), cache_dir=str(cache), selected_parameters=parameters)
            run_evaluation(args, [("S_st", arrays)], bins, _component_metrics, family_of)
            self.assertEqual(json.loads((output / "selected_parameters.json").read_text()), settings)
            self.assertEqual(json.loads((output / "protocol.json").read_text())["status"], "complete")
            self.assertNotIn("{{", (output / "README.md").read_text())
            with (output / "per_sequence.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 8)
            self.assertTrue(all(float(row["mean"]) < 1e-10 for row in rows))
            self.assertTrue((output / "figures/S_st.png").exists())
            self.assertTrue((output / "figures/S_st.svg").exists())
            with np.load(output / "tracks/S_st.npz") as tracks:
                np.testing.assert_allclose(tracks["raw_pos"], tracks["true_pos"])
            with self.assertRaises(FileExistsError):
                run_evaluation(args, [("S_st", arrays)], bins, _component_metrics, family_of)

    def test_pmf_filters_are_causal_normalized_and_reset(self):
        """
        Check causality, independent filter state and normalized PMF outputs.
        """
        bins = np.array([[0.0, 0.0, 8.0], [1.0, 0.0, 8.0]])
        sequence = np.array([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9]])
        modified = sequence.copy()
        modified[-1] = [0.99, 0.01]
        for method, parameter in (("moving_average", 3), ("ema", 0.2), ("adaptive", (48.64, 0.5))):
            values, pdf = pmf_filter(sequence, bins, "pos", method, parameter)
            altered, _ = pmf_filter(modified, bins, "pos", method, parameter)
            repeated, _ = pmf_filter(sequence, bins, "pos", method, parameter)
            np.testing.assert_allclose(values[:2], altered[:2])
            np.testing.assert_allclose(values, repeated)
            np.testing.assert_allclose(np.sum(pdf, axis=1), 1.0, atol=1e-6)
            self.assertTrue(np.all(pdf >= 0))
            without_pdf, omitted = pmf_filter(sequence, bins, "pos", method, parameter, return_pdfs=False)
            np.testing.assert_allclose(values, without_pdf)
            self.assertIsNone(omitted)

    def test_manual_sweep_records_adjacent_settings_and_leaves_selection_empty(self):
        """
        Check candidate ordering, adjacent differences and manual-selection outputs.
        """
        alphas = {"ori": 16.49, "pos": 48.64}
        self.assertEqual(sweep_parameters("adaptive", "ori", alphas, 3)[-1], (16.49, 0.1))
        self.assertEqual(sweep_parameters("ema", "pos", alphas, 3),
                         [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
        self.assertEqual(sweep_parameters("moving_average", "ori", alphas, 3), [1, 2, 3])
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            asset = root / "plotly.min.js"
            asset.write_text("/* offline asset fixture; UI is verified with real Plotly separately */")
            quats = np.array([qexp([0, 0, i * 0.005]) for i in range(40)])
            positions = np.c_[np.arange(40) * 0.01, np.zeros(40), np.full(40, 15.0)]
            bins = {"ori": np.array([qexp([0, 0, a]) for a in (0, 0.1)]),
                    "pos": np.array([[0, 0, 15], [1, 0, 15]], dtype=float)}
            arrays = {"frame": np.array([f"{i:05d}.png" for i in range(40)]),
                      "true_ori": quats, "pred_ori": quats, "true_pos": positions, "pred_pos": positions,
                      "ori_pdf": np.tile([0.7, 0.3], (40, 1)), "pos_pdf": np.tile([0.7, 0.3], (40, 1))}
            output = root / "sweep"
            args = SimpleNamespace(output_dir=str(output), plotly_js=asset, sweep_window_max=3,
                                   model="synthetic", cache_dir="synthetic")
            run_sweep(args, [("CATR", arrays)], bins, _component_metrics, alphas)
            with (output / "metrics.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 46)
            first, second = rows[:2]
            self.assertEqual(first["previous"], "raw")
            self.assertEqual(second["previous"], "n=1")
            self.assertAlmostEqual(float(second["delta_mean"]), float(second["mean"]) - float(first["mean"]))
            self.assertIsNone(json.loads((output / "manual_selection.json").read_text())["ori"]["ema"]["beta"])
            self.assertEqual(json.loads((output / "protocol.json").read_text())["status"], "complete")
            self.assertTrue((output / "CATR" / "index.html").exists())
            self.assertEqual(len(list((output / "CATR").glob("*.png"))), 6)
            with np.load(output / "CATR" / "tracks.npz") as tracks:
                np.testing.assert_allclose(tracks["pred_ori"], quats)
                self.assertIn("adaptive_ori_09", tracks.files)
            with self.assertRaises(FileExistsError):
                run_sweep(args, [("CATR", arrays)], bins, _component_metrics, alphas)

    def test_quaternion_sign_and_tracking_metrics(self):
        """
        Check quaternion sign invariance and zero error for perfect tracking.
        """
        quats = np.array([qexp(np.array([0, 0, i * 0.005])) for i in range(40)])
        np.testing.assert_allclose(qdistance(quats, -quats), 0, atol=1e-8)
        np.testing.assert_allclose(qlog(qexp([0.01, -0.02, 0.03])), [0.01, -0.02, 0.03], atol=1e-8)
        metrics = _component_metrics(quats, -quats, "ori")
        self.assertLess(metrics["mean"], 1e-6)
        self.assertLess(metrics["jitter"], 1e-6)
        self.assertLess(metrics["rpe25"], 1e-6)
        positions = np.c_[np.arange(40)*0.01, np.zeros(40), np.full(40, 15.0)]
        metrics = _component_metrics(positions, positions, "pos")
        self.assertEqual(metrics["jitter"], 0)
        self.assertEqual(metrics["rpe25"], 0)

    def test_docker_and_host_paths_share_identical_cache(self):
        """
        Check that cache identities survive remounting but reject changed model hashes.
        """
        identity = {"schema": 1, "model_sha256": "abc", "config_sha256": "def",
                    "sequence_labels": {"S_eb": "ghi"}}
        with tempfile.TemporaryDirectory() as root:
            cache = Path(root) / "cache"
            ensure_manifest(cache, dict(identity, model="/workspace/model", dataset="/workspace/dspeed"), None)
            ensure_manifest(cache, dict(identity, model="/home/model", dataset="/home/dspeed"), None)
            with self.assertRaises(ValueError):
                ensure_manifest(cache, dict(identity, model="/home/model", dataset="/home/dspeed",
                                            model_sha256="different"), None)

    def test_family_grouping_and_end_to_end_output(self):
        """
        Check family grouping and fixed evaluation across multiple synthetic sequences.
        """
        self.assertEqual(family_of("TITR_eb"), family_of("TITR_st"))
        self.assertEqual(family_of("CATR"), "CATR")
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            cache, output = root / "cache", root / "results"
            cache.mkdir()
            bins = {"ori": np.array([qexp([0, 0, angle]) for angle in (-0.2, 0, 0.2)]),
                    "pos": np.array([[0, 0, 15], [1, 0, 15], [2, 0, 15]], dtype=float)}
            selected = []
            for name in ("S_eb", "S_st", "CATR"):
                directory = root / name
                directory.mkdir()
                selected.append(directory)
                count = 40
                truth_q = np.array([qexp([0, 0, i*0.002]) for i in range(count)])
                truth_p = np.c_[np.arange(count)*0.01, np.zeros(count), np.full(count, 15.0)]
                ori_pdf = np.tile([0.2, 0.6, 0.2], (count, 1)).astype(np.float32)
                pos_pdf = np.tile([0.6, 0.3, 0.1], (count, 1)).astype(np.float32)
                np.savez_compressed(cache / f"{name}.npz", frame=np.array([f"{i:05}.png" for i in range(count)]),
                                    true_ori=truth_q, true_pos=truth_p,
                                    pred_ori=truth_q, pred_pos=truth_p,
                                    ori_pdf=ori_pdf, pos_pdf=pos_pdf)
            args = SimpleNamespace(cache_dir=str(cache), output_dir=str(output),
                                   model="synthetic", dataset=str(root), max_frames=40)
            (cache / "manifest.json").write_text(json.dumps({"model_sha256": "fixture", "config_sha256": "fixture"}))
            args.selected_parameters = root / "parameters.json"
            args.selected_parameters.write_text(json.dumps({
                component: {"adaptive": {"alpha": alpha, "n": n}, "ema": {"beta": 0.2},
                            "moving_average": {"window": window}}
                for component, alpha, n, window in (("ori", 16.49, 0.8, 10), ("pos", 48.64, 0.5, 6))}))
            evaluate(args, selected, bins)
            with open(output / "per_sequence.csv") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 3*4*2)
            self.assertEqual({row["method"] for row in rows},
                             {"raw", "moving_average", "ema", "adaptive"})
            self.assertTrue((output / "summary.csv").exists())
            self.assertEqual({row["family"] for row in rows}, {"S", "CATR"})
            self.assertTrue((output / "figures" / "CATR.png").exists())
            self.assertEqual(json.loads((output / "protocol.json").read_text())["parameters"],
                             json.loads(args.selected_parameters.read_text()))
            self.assertFalse((output / "fold_parameters.csv").exists())


if __name__ == "__main__":
    unittest.main()
