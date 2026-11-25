"""Check the real Burgers data contract without opening the sealed test arrays."""
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import model_config as m
import runtime as rt
import control as k


class BurgersDataChecks(unittest.TestCase):
    def test_split_and_train_only_preprocessing(self):
        root = m.V2 / "results/data"
        with np.load(root / "split_indices.npz") as f:
            train_ids, val_ids = f["train"], f["val"]
        self.assertEqual(len(train_ids), 1800)
        self.assertEqual(len(val_ids), 200)
        self.assertFalse(set(train_ids).intersection(val_ids))
        self.assertEqual(set(np.concatenate([train_ids, val_ids])), set(range(2000)))
        expected = np.random.default_rng(2026111801).permutation(2000)
        np.testing.assert_array_equal(train_ids, expected[:1800])
        np.testing.assert_array_equal(val_ids, expected[1800:])
        with np.load(root / "train.npz") as f, np.load(root / "preprocessing.npz") as prep:
            self.assertEqual(f["y"].shape, (1800, 2500))
            self.assertTrue(np.all(np.linalg.norm(f["y"], axis=1) > 1e-12))
            for i, name in enumerate(["x1", "x2", "xt"]):
                np.testing.assert_array_equal(prep[f"low{i}"], f[name].min(0))
                np.testing.assert_array_equal(prep[f"high{i}"], f[name].max(0))

    def test_query_target_contract(self):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        import generator as g
        with np.load(m.V2 / "results/data/train.npz") as f:
            x = np.arange(50, dtype=np.float32) / 50
            t = np.linspace(0, 1, 50, dtype=np.float32)
            expected = g.trunk_periodic_features(np.tile(x, 50), np.repeat(t, 50))
            np.testing.assert_allclose(f["xt"], expected, rtol=1e-6, atol=1e-6)
            initial = g.interp_periodic(f["x1"][0], np.arange(13, dtype=np.float32)/13, x)
            np.testing.assert_allclose(f["y"][0, :50], initial, atol=1e-6)

    def test_test_load_is_closed_before_any_array_access(self):
        data = rt.GuardedData()
        with patch.object(np, "load", side_effect=AssertionError("An array was opened")):
            with self.assertRaisesRegex(RuntimeError, "Test split is closed"):
                data.load("test", True)

    def test_both_training_input_encodings_match_models(self):
        data = rt.GuardedData()
        for qofn, dims in [(False, [13, 13, 5]), (True, [14, 14, 6])]:
            x, y = data.load("val", qofn)
            self.assertEqual([a.shape[1] for a in x], dims)
            self.assertEqual(y.shape, (200, 2500))
            self.assertTrue(all(bool(m.c.torch.isfinite(a).all()) for a in x))
            group = "full_adam" if qofn else "mionet_adam"
            net = m.build_model(group)
            tiny = (x[0][:2], x[1][:2], x[2][:7])
            self.assertEqual(net(m.c.inputs_for(group, tiny)).shape, (2, 7))


if __name__ == "__main__":
    unittest.main(verbosity=2)
