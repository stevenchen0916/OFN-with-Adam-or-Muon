"""Short correctness/timing benchmark, never used for model selection."""
import copy
import time
import types

import numpy as np
import torch

import common as c
from models.quantum_compatible_orthogonal_layer import QuantumCompatibleOrthogonalLayer


def stage_reference(layer, x):
    """Original stage-vectorized implementation before composing small matrices."""
    if layer.in_features < layer.out_features:
        x = torch.nn.functional.pad(x, (layer.out_features - layer.in_features, 0))
    cosines, sines = layer.thetas.cos(), layer.thetas.sin()
    for start, end, a, b in zip(layer.starts, layer.ends, layer.theta_starts, layer.theta_ends):
        pairs = x[:, start:end].reshape(len(x), -1, 2)
        x0, x1 = pairs[..., 0], pairs[..., 1]
        rotated = torch.stack((x0 * cosines[a:b] - x1 * sines[a:b],
                               x0 * sines[a:b] + x1 * cosines[a:b]), dim=-1).reshape(len(x), -1)
        x = torch.slice_scatter(x, rotated, dim=1, start=start, end=end)
    if layer.in_features > layer.out_features:
        x = x[:, layer.in_features - layer.out_features:]
    return x + layer.bias if layer.bias is not None else x


def as_stage_reference(net):
    for layer in net.modules():
        if isinstance(layer, QuantumCompatibleOrthogonalLayer):
            layer.forward = types.MethodType(stage_reference, layer)
    return net


def benchmark(args, data):
    from run import make_optimizers, settings, sync, write_csv
    if args.warmup < 1 or args.repeats < 3:
        raise ValueError("Use at least one warmup and three measured updates.")
    s = settings(args)
    c.seed_all(s["pilot"]["seed"])
    new = c.build_model("full_adam").to(args.device)
    old = as_stage_reference(copy.deepcopy(new))
    x, y = data.load("train", True)
    with torch.no_grad():
        a, b = new(x), old(x)
        maximum = float((a - b).abs().max())
        relative = float((a - b).double().norm() / b.double().norm().clamp_min(1e-12))
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
    correctness = dict(seed=s["pilot"]["seed"], functions=len(y), queries=y.shape[1],
                       maximum_absolute_output_difference=maximum, relative_l2_output_difference=relative,
                       tolerance=dict(atol=1e-6, rtol=1e-5),
                       reference="original stage-vectorized rotations, same state_dict")
    del a, b, new, old
    results = []
    cases = [(group, "matrix-composition") for group in s["groups"]] + [("full_adam", "original-stages")]
    for group, implementation in cases:
        c.seed_all(s["pilot"]["seed"])
        net = c.build_model(group).to(args.device)
        if implementation == "original-stages":
            as_stage_reference(net)
        # This benchmark uses fixed nominal rates only to measure update cost.
        # It does not select a learning rate or retain benchmark model weights.
        adam_lr = s["pilot"]["adam_lr"]
        muon_lr = adam_lr if group.endswith("_muon_adam") else None
        optimizers, _ = make_optimizers(net, group, adam_lr, muon_lr)
        bx, by = data.load("train", c.is_qofn(group))
        durations = []
        for i in range(args.warmup + args.repeats):
            sync(args.device)
            tick = time.perf_counter()
            for opt in optimizers:
                opt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(net(c.inputs_for(group, bx)), by)
            if not torch.isfinite(loss):
                raise FloatingPointError("Short timing benchmark became nonfinite.")
            loss.backward()
            for opt in optimizers:
                opt.step()
            sync(args.device)
            if i >= args.warmup:
                durations.append(time.perf_counter() - tick)
        row = dict(group=group, implementation=implementation, batch_functions=len(by),
                   batch_queries=by.shape[1], warmup_updates=args.warmup,
                   measured_updates=args.repeats, mean_step_seconds=float(np.mean(durations)),
                   median_step_seconds=float(np.median(durations)),
                   minimum_step_seconds=min(durations), maximum_step_seconds=max(durations),
                   parameters=sum(p.numel() for p in net.parameters()))
        results.append(row)
        print(f"Benchmark | {group} | {implementation} | median={row['median_step_seconds']:.6f} s/update", flush=True)
        del net, optimizers, loss
    output = args.out / "benchmark.json"
    record = dict(correctness=correctness, timings=results, hardware=c.environment(args.device),
                  dataset_id=data.manifest["dataset_id"], code_sha=c.code_hash(),
                  timing_scope="forward+MSE+backward+optimizer; warmup excluded; validation/diagnostics/I/O excluded",
                  limitation="Short microbenchmark, not sustained thermally steady MacBook training throughput.")
    new_time = next(r["median_step_seconds"] for r in results if r["group"] == "full_adam" and r["implementation"] == "matrix-composition")
    old_time = results[-1]["median_step_seconds"]
    record["full_adam_short_benchmark_speed_ratio_old_over_new"] = old_time / new_time
    c.write_json(output, record)
    write_csv(args.out / "benchmark.csv", results)
    print("Output equivalence:", correctness)
    print("Saved", output, "(no training checkpoints retained).")
