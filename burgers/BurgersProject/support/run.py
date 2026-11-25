"""Optimizer allocation and layer diagnostics shared with the heat protocol."""
import math
import torch
import common as c

def make_optimizers(net, group, adam_lr, muon_lr=None):
    named = dict(net.named_parameters())
    if group.endswith("_muon_adam"):
        if muon_lr is None or not math.isfinite(muon_lr) or muon_lr <= 0:
            raise ValueError("A hybrid requires a separate positive muon_lr.")
        if c.is_qofn(group):
            chosen = [getattr(net, name).output_layer.weight for name in ("branch1", "branch2", "trunk")]
        elif group == "deeponet_output_muon_adam":
            chosen = [net.branch.linears[-1].weight, net.trunk.linears[-1].weight]
        else:
            chosen = [p for p in named.values() if p.ndim == 2]
        ids = {id(p) for p in chosen}
        other = [p for p in named.values() if id(p) not in ids]
        opts = [torch.optim.Muon(chosen, lr=muon_lr, weight_decay=0, adjust_lr_fn="match_rms_adamw"),
                torch.optim.Adam(other, lr=adam_lr, weight_decay=0)]
    else:
        if muon_lr is not None:
            raise ValueError("Pure Adam groups must not receive a Muon LR.")
        ids = set()
        opts = [torch.optim.Adam(net.parameters(), lr=adam_lr, weight_decay=0)]
    return opts, dict(adam_lr=adam_lr, muon_lr=muon_lr,
                      muon=[n for n, p in named.items() if id(p) in ids],
                      adam=[n for n, p in named.items() if id(p) not in ids],
                      defaults=[o.defaults for o in opts])

@torch.no_grad()
def diagnostics(net, x):
    rows = []
    for name, values in zip(("branch1", "branch2", "trunk"), x):
        encoder = getattr(net, name)
        ids = torch.linspace(0, len(values) - 1, min(32, len(values)), device=values.device).long()
        h = values[ids]
        def stats(z):
            n = z.norm(dim=1)
            return dict(mean=float(n.mean()), rms=float(n.square().mean().sqrt()), maximum=float(n.max()))
        for i, layer in enumerate(encoder.hidden_layers):
            pre = encoder.hidden_preactivation(h, i)
            post = encoder.activation(pre)
            rows.append(dict(encoder=name, layer=i,
                             alpha=float(torch.sigmoid(encoder.h_params[i])) if encoder.flow else None,
                             gate_active=bool(encoder.flow and layer.in_features == layer.out_features),
                             input_norm=stats(h), preactivation_norm=stats(pre), feature_norm=stats(post)))
            h = post
        rows.append(dict(encoder=name, layer="output", feature_norm=stats(encoder.output_layer(h))))
    return rows
