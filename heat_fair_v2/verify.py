"""CPU correctness/regression checks. No real PDE model experiment is run."""
import copy
from contextlib import redirect_stdout
import importlib
import io
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import run as e
import common as c
import numpy as np
import torch
from models.quantum_compatible_orthogonal_layer import QuantumCompatibleOrthogonalLayer
from timing import as_stage_reference, stage_reference


DEFAULTS = c.read_json(e.HERE / "settings.json")


def scalar_reference(layer, x):
    if layer.in_features < layer.out_features:
        x = torch.nn.functional.pad(x, (layer.out_features - layer.in_features, 0))
    theta = 0
    for start, end in zip(layer.starts, layer.ends):
        cols = list(x.unbind(1))
        for i in range(start, end, 2):
            a, b = x[:, i], x[:, i + 1]
            co, si = layer.thetas[theta].cos(), layer.thetas[theta].sin()
            cols[i], cols[i + 1] = co * a - si * b, si * a + co * b
            theta += 1
        x = torch.stack(cols, 1)
    if layer.in_features > layer.out_features:
        x = x[:, layer.in_features - layer.out_features:]
    return x + layer.bias if layer.bias is not None else x


class Correctness(unittest.TestCase):
    def test_default_cpu_after_backend_import(self):
        previous = torch.get_default_device()
        try:
            torch.set_default_device("meta")
            importlib.reload(c)
            self.assertEqual(torch.get_default_device(), torch.device("cpu"))
            self.assertEqual(torch.ones(1, dtype=torch.float64).device.type, "cpu")
        finally:
            torch.set_default_device(previous)

    def test_distinct_grid_coordinate_permutation(self):
        nt, ny, nx = 3, 4, 5
        sol = np.arange(nt * ny * nx).reshape(nt, ny, nx)
        repaired = c.repair_targets(sol.transpose(0, 2, 1).reshape(1, -1), nt, nx, ny)[0]
        query = np.stack(np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nt), indexing="xy"), -1).reshape(-1, 3)
        for (x, y, t), value in zip(query, repaired):
            self.assertEqual(value, sol[t, y, x])

    def test_rotation_values_and_gradients(self):
        for nin, nout in [(1, 8), (2, 8), (8, 25), (11, 25), (4, 25), (25, 25), (25, 8), (5, 3), (1, 1)]:
            with self.subTest(shape=(nin, nout)):
                c.seed_all(289)
                layer = QuantumCompatibleOrthogonalLayer(nin, nout).to(device="cpu", dtype=torch.float64)
                other = copy.deepcopy(layer)
                x = torch.randn(9, nin, dtype=torch.float64, device="cpu", requires_grad=True)
                xo = x.detach().clone().requires_grad_()
                y, yo = layer(x), scalar_reference(other, xo)
                torch.testing.assert_close(y, yo, rtol=1e-11, atol=1e-11)
                probe = torch.randn_like(y)
                (y * probe).sum().backward()
                (yo * probe).sum().backward()
                torch.testing.assert_close(x.grad, xo.grad, rtol=1e-10, atol=1e-10)
                for a, b in zip(layer.parameters(), other.parameters()):
                    if a.grad is not None or b.grad is not None:
                        torch.testing.assert_close(a.grad, b.grad, rtol=1e-10, atol=1e-10)

    def test_whole_qofn_forward_equivalence_float32(self):
        c.seed_all(289)
        new = c.build_model("full_adam")
        old = as_stage_reference(copy.deepcopy(new))
        x = (torch.randn(32, 2), torch.randn(32, 11), torch.randn(128, 4))
        with torch.no_grad():
            torch.testing.assert_close(new(x), old(x), rtol=1e-5, atol=1e-6)

    def test_optimizer_routes_and_separate_lrs(self):
        for group in DEFAULTS["groups"]:
            net = c.build_model(group)
            hybrid = group.endswith("_muon_adam")
            opts, routing = e.make_optimizers(net, group, 3e-4, .03 if hybrid else None)
            self.assertEqual(opts[-1].defaults["lr"], 3e-4)
            self.assertEqual(opts[-1].defaults["weight_decay"], 0)
            if hybrid:
                self.assertEqual(opts[0].defaults["lr"], .03)
                self.assertEqual(opts[0].defaults["adjust_lr_fn"], "match_rms_adamw")
                self.assertEqual(opts[0].defaults["weight_decay"], 0)
            if group == "deeponet_output_muon_adam":
                self.assertEqual(set(routing["muon"]), {"branch.linears.6.weight", "trunk.linears.6.weight"})
            elif c.is_qofn(group) and hybrid:
                self.assertEqual(set(routing["muon"]), {f"{k}.output_layer.weight" for k in ("branch1", "branch2", "trunk")})
            all_ids = [id(p) for opt in opts for g in opt.param_groups for p in g["params"]]
            self.assertEqual(len(all_ids), len(set(all_ids)))
            self.assertEqual(set(all_ids), {id(p) for p in net.parameters()})

    def test_full_noflow_share_initialization_and_normalization(self):
        c.seed_all(289)
        full = c.build_model("full_adam")
        c.seed_all(289)
        noflow = c.build_model("noflow_adam")
        for name, value in noflow.state_dict().items():
            torch.testing.assert_close(full.state_dict()[name], value, rtol=0, atol=0)
        for encoder in ("branch1", "branch2", "trunk"):
            a, b = getattr(full, encoder), getattr(noflow, encoder)
            i = next(i for i, layer in enumerate(a.hidden_layers) if layer.in_features == layer.out_features)
            h = torch.randn(6, a.hidden_layers[i].in_features)
            normed = h / h.norm(dim=1, keepdim=True).clamp_min(1e-12)
            z = b.hidden_layers[i](normed)
            torch.testing.assert_close(b.hidden_preactivation(h, i), z)
            torch.testing.assert_close(a.hidden_preactivation(h, i), .5 * h + .5 * z)

    def test_plateau_patience_and_late_recovery(self):
        records = [dict(step=i, validation_relative_l2=.2) for i in range(0, 6500, 500)]
        self.assertFalse(e.plateau(records[:-2], DEFAULTS)["reached"])
        self.assertTrue(e.plateau(records, DEFAULTS)["reached"])
        recovered = records + [dict(step=i, validation_relative_l2=.1) for i in [6500, 7000, 7500]]
        self.assertFalse(e.plateau(recovered, DEFAULTS)["reached"])
        descending = [dict(step=i, validation_relative_l2=1 / (i + 500)) for i in range(0, 20500, 500)]
        self.assertFalse(e.plateau(descending, DEFAULTS)["reached"])

    def test_target_is_first_observed_and_missing_is_null(self):
        rows = [dict(step=i * 500, validation_relative_l2=v, training_seconds=i * 20, wall_seconds=i * 22)
                for i, v in enumerate([.2, .02, .009, .008])]
        self.assertEqual(e.first_target(rows, .01), dict(step=1000, training_seconds=40, wall_seconds=44))
        self.assertIsNone(e.first_target(rows, .001))

    def test_pilot_is_three_groups_only_one_can_stop_early(self):
        with patch.object(e, "settings", return_value=DEFAULTS), patch.object(e, "train_one") as trainer, redirect_stdout(io.StringIO()):
            e.run_pilots(None, None)
        self.assertEqual(trainer.call_count, 3)
        self.assertEqual([r.args[3] for r in trainer.call_args_list], e.PILOTS)
        self.assertEqual([r.kwargs["early_stop"] for r in trainer.call_args_list], [False, False, True])

    def test_sweep_freezes_auxiliary_adam_and_uses_135_trials(self):
        search = dict(adam_lrs=[.0001, .0003, .001, .002, .003], muon_lrs=[.001, .003, .01, .03, .1])
        def candidate(args, group, rates, fixed_adam=None):
            return [], dict(adam_lr=.0003), {}
        with TemporaryDirectory() as temp:
            args = SimpleNamespace(out=Path(temp))
            with patch.object(e, "settings", return_value=DEFAULTS), patch.object(e, "search_settings", return_value=search), \
                 patch.object(e, "frozen_T", return_value=5000), patch.object(e, "candidates", side_effect=candidate), \
                 patch.object(e, "train_one") as trainer, redirect_stdout(io.StringIO()):
                e.sweep(args, None)
            self.assertEqual(trainer.call_count, 135)
            hybrid = [call for call in trainer.call_args_list if call.args[3] not in e.PURE]
            self.assertEqual({call.args[4] for call in hybrid}, {.0003})
            self.assertEqual({call.args[5] for call in hybrid}, set(search["muon_lrs"]))
            self.assertEqual({call.args[6] for call in trainer.call_args_list}, {289, 290, 291})

    def test_selection_uses_three_seed_mean_and_rejects_partial_divergence(self):
        rates = [.0001, .0003, .001, .002, .003]
        with TemporaryDirectory() as temp:
            args = SimpleNamespace(out=Path(temp))
            c.write_json(args.out / "protocol.json", {})
            for lr in rates:
                for seed in DEFAULTS["tuning_seeds"]:
                    failed = lr == .003 and seed == 289
                    value = ({289: .001, 290: 1., 291: 1.}[seed] if lr == .0001 else (.2 if lr == .001 else .8))
                    path = e.run_dir(args.out, "sweep", "deeponet_adam", lr, None, seed)
                    path.mkdir(parents=True)
                    c.write_json(path / "result.json", dict(status="diverged" if failed else "complete",
                                 actual_steps=1250 if failed else 5000, protocol={},
                                 final_validation_relative_l2=None if failed else value))
            with patch.object(e, "settings", return_value=DEFAULTS), patch.object(e, "frozen_T", return_value=5000):
                table, best, hashes = e.candidates(args, "deeponet_adam", rates)
            self.assertEqual(best["adam_lr"], .001)
            self.assertEqual(len(hashes), 15)
            failed_row = next(r for r in table if r["adam_lr"] == .003)
            self.assertIsNone(failed_row["mean_validation"])
            self.assertEqual(failed_row["n_complete"], 2)

    def test_short_benchmark_on_synthetic_tensors(self):
        from timing import benchmark
        tiny = copy.deepcopy(DEFAULTS)
        tiny["groups"] = ["full_adam", "deeponet_muon_adam"]
        c.seed_all(2026)
        raw = (torch.randn(4, 1), torch.randn(4, 10), torch.randn(6, 3))
        q = (torch.randn(4, 2), torch.randn(4, 11), torch.randn(6, 4))
        y = torch.ones(4, 6)
        data = SimpleNamespace(manifest=dict(dataset_id="mock-dataset"), load=lambda split, qofn: (q if qofn else raw, y))
        with TemporaryDirectory() as temp:
            args = SimpleNamespace(out=Path(temp), device=torch.device("cpu"), warmup=1, repeats=3)
            with patch.object(e, "settings", return_value=tiny), \
                 patch.object(c, "environment", return_value=dict(device="cpu", threads=2)), redirect_stdout(io.StringIO()):
                benchmark(args, data)
            report = c.read_json(args.out / "benchmark.json")
            self.assertEqual(len(report["timings"]), 3)
            self.assertTrue(all(r["median_step_seconds"] > 0 for r in report["timings"]))
            self.assertEqual(report["correctness"]["functions"], 4)

    def test_final_has_ten_new_seeds_not_tuning_models(self):
        chosen = dict(configs={g: dict(adam_lr=.001, muon_lr=.003 if g.endswith("_muon_adam") else None)
                              for g in DEFAULTS["groups"]}, steps=5000, final_seeds=DEFAULTS["final_seeds"])
        with patch.object(e, "selection", return_value=chosen), patch.object(e, "settings", return_value=DEFAULTS), \
             patch.object(e, "train_one") as trainer, redirect_stdout(io.StringIO()):
            e.final_runs(None, None)
        self.assertEqual(trainer.call_count, 90)
        seeds = {call.args[6] for call in trainer.call_args_list}
        self.assertEqual(seeds, set(range(292, 302)))
        self.assertTrue(seeds.isdisjoint(DEFAULTS["tuning_seeds"]))

    def test_evaluation_keeps_test_closed_when_a_model_is_missing(self):
        chosen = dict(configs={g: dict(adam_lr=.001, muon_lr=None) for g in DEFAULTS["groups"]},
                      steps=5000, final_seeds=DEFAULTS["final_seeds"])
        with TemporaryDirectory() as temp:
            args = SimpleNamespace(out=Path(temp))
            data = SimpleNamespace(load=Mock())
            with patch.object(e, "selection", return_value=chosen), patch.object(e, "settings", return_value=DEFAULTS):
                with self.assertRaises(FileNotFoundError):
                    e.evaluate(args, data)
            data.load.assert_not_called()

    def test_evaluation_cached_once_for_each_of_90_models(self):
        chosen = dict(configs={g: dict(adam_lr=.001, muon_lr=.003 if g.endswith("_muon_adam") else None)
                              for g in DEFAULTS["groups"]}, steps=5000, final_seeds=DEFAULTS["final_seeds"])
        with TemporaryDirectory() as temp:
            args = SimpleNamespace(out=Path(temp), device=torch.device("cpu"))
            c.write_json(args.out / "selection.json", chosen)
            data = SimpleNamespace(manifest=dict(files={"test.npz": "mock-test-sha"}), load=Mock(return_value=(None, None)))
            for group in DEFAULTS["groups"]:
                lr = chosen["configs"][group]
                for seed in DEFAULTS["final_seeds"]:
                    path = e.run_dir(args.out, "final", group, lr["adam_lr"], lr["muon_lr"], seed)
                    path.mkdir(parents=True)
                    (path / "final.pt").write_bytes(f"mock-{group}-{seed}".encode())
                    c.write_json(path / "result.json", dict(status="complete", actual_steps=5000, protocol={},
                                 checkpoint_sha=c.digest(path / "final.pt"), first_target=None,
                                 target_not_observed=True, parameters=42, training_seconds=10., wall_seconds=12.))
            with patch.object(e, "settings", return_value=DEFAULTS), patch.object(e, "selection", return_value=chosen), \
                 patch.object(e, "protocol", return_value={}), patch.object(c, "build_model"), \
                 patch.object(torch, "load", return_value={}), patch.object(e, "relative_error", return_value=.1) as score, \
                 redirect_stdout(io.StringIO()):
                e.evaluate(args, data)
                e.evaluate(args, data)
            self.assertEqual(score.call_count, 90)
            self.assertTrue((args.out / "paired_effects.csv").exists())

    def test_real_optimizer_checkpoint_resume_matches_uninterrupted_tiny_fixture(self):
        # Synthetic tensors only, two updates: exercises the real serialization
        # and optimizer path without running a heat-equation training experiment.
        c.seed_all(2026)
        x = (torch.randn(4, 1), torch.randn(4, 10), torch.randn(6, 3))
        y = torch.ones(4, 6)
        tiny = copy.deepcopy(DEFAULTS)
        tiny["validation_every"] = 1
        tiny["target_validation_error"] = 1000
        data = SimpleNamespace(load=lambda split, qofn: (x, y))
        for group, muon_lr, interrupt_on_observation in [("deeponet_adam", None, 3), ("deeponet_muon_adam", .003, 3), ("deeponet_adam", None, 2)]:
            with self.subTest(group=group, interrupt_on=interrupt_on_observation), TemporaryDirectory() as temp:
                base = Path(temp)
                args = SimpleNamespace(out=base / "continuous", device=torch.device("cpu"))
                resumed = SimpleNamespace(out=base / "resumed", device=torch.device("cpu"))
                actual_error = e.relative_error
                counter = [0]
                def interrupted_error(*args, **kwargs):
                    counter[0] += 1
                    if counter[0] == interrupt_on_observation:
                        raise RuntimeError("synthetic interruption after the first saved update")
                    return actual_error(*args, **kwargs)
                with patch.object(e, "settings", return_value=tiny), patch.object(e, "protocol", return_value={}), \
                     patch.object(c, "environment", return_value={}), redirect_stdout(io.StringIO()):
                    continuous = e.train_one(args, data, "pilot", group, .001, muon_lr, 289, 2)
                    with patch.object(e, "relative_error", side_effect=interrupted_error):
                        with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
                            e.train_one(resumed, data, "pilot", group, .001, muon_lr, 289, 2)
                    continued = e.train_one(resumed, data, "pilot", group, .001, muon_lr, 289, 2)
                left = e.run_dir(args.out, "pilot", group, .001, muon_lr, 289)
                right = e.run_dir(resumed.out, "pilot", group, .001, muon_lr, 289)
                a = torch.load(left / "final.pt", weights_only=True)
                b = torch.load(right / "final.pt", weights_only=True)
                for key in a:
                    torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
                records = e.read_lines(right / "validation.jsonl")
                self.assertEqual([r["step"] for r in records], [0, 1, 2])
                self.assertEqual(continued["actual_steps"], 2)
                self.assertEqual(continued["first_target"]["step"], 0)
                self.assertGreaterEqual(continued["wall_seconds"], continued["training_seconds"])
                self.assertEqual(continued["final_validation_relative_l2"], continuous["final_validation_relative_l2"])


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
