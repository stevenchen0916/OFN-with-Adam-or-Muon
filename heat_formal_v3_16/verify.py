"""Bounded checks of the changed architecture and unchanged routing."""
import unittest
import torch
import model_config as m
from run import make_optimizers


class WideModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_default_device("cpu")
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)

    def test_parameter_budget_and_output_rank(self):
        for row in m.parameter_table():
            self.assertLess(abs(row["relative_difference_from_deeponet"]), .01)
        for group in m.c.GROUPS:
            with self.subTest(group=group):
                net = m.build_model(group)
                self.assertEqual(sum(p.numel() for p in net.parameters()),
                                 m.EXPECTED_PARAMETERS[m.architecture(group)])
                if group.startswith("deeponet"):
                    self.assertEqual(net.branch.linears[-1].out_features, 25)
                    self.assertEqual(net.trunk.linears[-1].out_features, 25)
                else:
                    for name in ["branch1", "branch2", "trunk"]:
                        encoder = getattr(net, name)
                        last = encoder.output_layer if hasattr(encoder, "output_layer") else encoder.linears[-1]
                        self.assertEqual(last.out_features, 25)

    def test_forward_backward_and_full_noflow_shared_normalization(self):
        for group in ["deeponet_adam", "mionet_adam", "full_adam", "noflow_adam"]:
            with self.subTest(group=group):
                m.c.seed_all(289)
                net = m.build_model(group)
                qofn = m.c.is_qofn(group)
                inputs = tuple(torch.randn(3 if i < 2 else 7, d) for i,d in enumerate([2,11,4] if qofn else [1,10,3]))
                output = net(m.c.inputs_for(group, inputs))
                self.assertEqual(output.shape, (3,7))
                output.square().mean().backward()
                self.assertTrue(torch.isfinite(output).all())
                for p in net.parameters():
                    if p.grad is not None:
                        self.assertTrue(torch.isfinite(p.grad).all())
        a,b = m.build_model("full_adam"), m.build_model("noflow_adam")
        x = torch.randn(3,50)
        b.branch2.hidden_layers[1].load_state_dict(a.branch2.hidden_layers[1].state_dict())
        a.branch2.h_params[1].data.fill_(100.)
        torch.testing.assert_close(a.branch2.hidden_preactivation(x,1), b.branch2.hidden_preactivation(x,1))

    def test_optimizer_coverage_and_routing(self):
        m.c.require_muon(torch.device("cpu"))
        for group in m.c.GROUPS:
            with self.subTest(group=group):
                net = m.build_model(group)
                hybrid = group.endswith("_muon_adam")
                opts, routing = make_optimizers(net, group, .001, .001 if hybrid else None)
                assigned = [id(p) for opt in opts for pg in opt.param_groups for p in pg["params"]]
                self.assertEqual(len(assigned),len(set(assigned)))
                self.assertEqual(set(assigned), {id(p) for p in net.parameters()})
                if group.startswith(("full_", "noflow_")) and hybrid:
                    self.assertEqual(set(routing["muon"]), {n+".output_layer.weight" for n in ["branch1","branch2","trunk"]})
                if group == "deeponet_output_muon_adam":
                    self.assertEqual(len(routing["muon"]),2)
                for opt in opts:
                    self.assertEqual(opt.defaults["weight_decay"],0)
                if hybrid:
                    self.assertEqual(opts[0].defaults["adjust_lr_fn"],"match_rms_adamw")


if __name__ == "__main__":
    unittest.main(verbosity=2)
