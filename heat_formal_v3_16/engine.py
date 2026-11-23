"""One deterministic run with complete, atomic checkpoints and safe resume."""
from __future__ import annotations
import copy
import json
import math
from pathlib import Path
import random
import signal
import time

import control as k
import model_config as m
import runtime as rt
import numpy as np
import torch
from run import make_optimizers, diagnostics as feature_diagnostics


class Interrupted(Exception):
    pass


def rng_state():
    n = np.random.get_state()
    return dict(python=random.getstate(), torch=torch.get_rng_state(),
                numpy=dict(name=n[0], keys=torch.from_numpy(n[1].astype(np.int64)),
                           position=n[2], has_gauss=n[3], cached_gaussian=n[4]))


def restore_rng(s):
    random.setstate(s["python"])
    torch.set_rng_state(s["torch"])
    n = s["numpy"]
    np.random.set_state((n["name"], n["keys"].numpy().astype(np.uint32),
                         n["position"], n["has_gauss"], n["cached_gaussian"]))


def schedulers(opts, horizon, ratio):
    return [torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=horizon,
              eta_min=o.param_groups[0]["lr"] * ratio) for o in opts]


def lr_check(opts, scheds, step, horizon, ratio):
    factor = ratio + .5 * (1 - ratio) * (1 + math.cos(math.pi * step / horizon))
    for opt, sched in zip(opts, scheds):
        if sched.last_epoch != step:
            raise RuntimeError("Scheduler step counter mismatch.")
        for pg, base in zip(opt.param_groups, sched.base_lrs):
            if not math.isclose(pg["lr"], base * factor, rel_tol=2e-10, abs_tol=1e-15):
                raise RuntimeError("Both optimizers must follow the frozen cosine schedule.")


def finite_parameters(parameters, gradients=False):
    values = [p.grad if gradients else p for p in parameters]
    values = [v.reshape(-1) for v in values if v is not None]
    return not values or bool(torch.isfinite(torch.cat(values)).all())


def update(net, opts, scheds, group, x, y):
    net.train()
    for o in opts:
        o.zero_grad(set_to_none=True)
    prediction = net(m.c.inputs_for(group, x))
    loss = torch.nn.functional.mse_loss(prediction, y)
    # With finite targets, any NaN/Inf prediction makes this MSE nonfinite.
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Nonfinite loss/model output.")
    loss.backward()
    if not finite_parameters(net.parameters(), gradients=True):
        raise FloatingPointError("Nonfinite gradient.")
    for o in opts:
        o.step()
    if not finite_parameters(net.parameters()):
        raise FloatingPointError("Nonfinite model parameter.")
    for s in scheds:
        s.step()
    return float(loss.detach())


@torch.no_grad()
def relative_error(net, group, x, y):
    net.eval()
    prediction = net(m.c.inputs_for(group, x))
    values = (prediction - y).double().norm(dim=1) / y.double().norm(dim=1).clamp_min(1e-12)
    result = float(values.mean())
    if not math.isfinite(result):
        raise FloatingPointError("Nonfinite validation/test model output or relative L2.")
    return result


def assert_tree_close(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)
    elif isinstance(a, dict):
        if a.keys() != b.keys():
            raise AssertionError("Checkpoint dictionary keys differ.")
        for key in a:
            assert_tree_close(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            raise AssertionError("Checkpoint sequence lengths differ.")
        for x, y in zip(a, b):
            assert_tree_close(x, y)
    elif isinstance(a, float):
        if not math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-15):
            raise AssertionError(f"Checkpoint scalar mismatch: {a} / {b}")
    elif a != b:
        raise AssertionError("Checkpoint state differs.")


def check_isolated_routing(net, group, opts):
    """Use copies so diagnostic optimizer steps cannot mutate the real run."""
    _, routing = make_optimizers(net, group, opts[-1].param_groups[0]["lr"],
                                opts[0].param_groups[0]["lr"] if k.hybrid(group) else None)
    original = dict(net.named_parameters())
    assigned = [id(p) for o in opts for pg in o.param_groups for p in pg["params"]]
    if len(assigned) != len(set(assigned)) or set(assigned) != {id(p) for p in original.values()}:
        raise AssertionError("Optimizer ownership overlaps or misses parameters.")
    results = []
    for index, opt in enumerate(opts):
        clone = copy.deepcopy(net)
        cloned_opts, _ = make_optimizers(clone, group, opts[-1].param_groups[0]["lr"],
                                       opts[0].param_groups[0]["lr"] if k.hybrid(group) else None)
        cloned = dict(clone.named_parameters())
        for name, p in original.items():
            cloned[name].grad = p.grad.clone() if p.grad is not None else None
        cloned_opts[index].load_state_dict(copy.deepcopy(opt.state_dict()))
        chosen_ids = {id(p) for pg in opt.param_groups for p in pg["params"]}
        owned_names = {name for name, p in original.items() if id(p) in chosen_ids}
        cloned_opts[index].step()
        changed = []
        for name, p in original.items():
            equal = torch.equal(p.detach(), cloned[name].detach())
            if name not in owned_names and not equal:
                raise AssertionError("Optimizer changed a parameter it does not own: " + name)
            if not equal:
                changed.append(name)
        results.append(dict(optimizer=type(opt).__name__, changed_names=changed,
                            unowned_unchanged=True, owned_count=len(owned_names)))
    return dict(partition_exact=True, isolated_steps=results, routing=routing)


def pack(net, opts, scheds):
    return dict(model=net.state_dict(), optimizers=[o.state_dict() for o in opts],
                schedulers=[s.state_dict() for s in scheds], rng=rng_state())


def load_pack(state, net, opts, scheds):
    net.load_state_dict(state["model"])
    # Schedulers are constructed before loading optimizer LRs; all state is copied.
    for s, saved in zip(scheds, state["schedulers"]):
        s.load_state_dict(copy.deepcopy(saved))
    for o, saved in zip(opts, state["optimizers"]):
        o.load_state_dict(copy.deepcopy(saved))
    restore_rng(state["rng"])


def atomic_torch(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(value, tmp)
    tmp.replace(path)


def train(path, identity, *, data=None, factory=None, stop_at=None,
          prepare_resume_reference=False, verify_resume=False, routing_check=False,
          inject_failure=None, diagnostic_fn=None, handle_signals=True, session_started=None):
    """Injection arguments exist for synthetic tests; real CLI never changes T/data."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    with rt.lock(path / ".run.lock"):
        if (path / "identity.json").exists() and k.read(path / "identity.json") != identity:
            raise RuntimeError("Different run identity in existing directory.")
        k.write(path / "identity.json", identity)
        cached = rt.load_result(path, identity)
        if cached is not None:
            return cached
        session_start = session_started if session_started is not None else time.perf_counter()
        wall_offset = 0.
        stop = [False]
        previous_handlers = {}
        if handle_signals:
            def request_stop(signum, frame):
                stop[0] = True
            for sig in ((signal.SIGINT, signal.SIGTERM, signal.SIGBREAK) if hasattr(signal, "SIGBREAK") else (signal.SIGINT, signal.SIGTERM)):
                previous_handlers[sig] = signal.signal(sig, request_stop)
        log = (path / "run.log").open("a", buffering=1)
        group = identity["group"]
        def emit(message):
            line = f"{identity['stage']} | {group} | seed={identity['seed']} | {message}"
            print(line, flush=True)
            log.write(line + "\n")
        def wall():
            return wall_offset + time.perf_counter() - session_start
        records, step_times, diagnostics, checkpoint_times, observation_times = [], [], [], [], []
        training_seconds, current_step, attempt = 0., 0, 0
        net = opts = scheds = None
        try:
            m.c.seed_all(identity["seed"])
            net = (factory or m.build_model)(group)
            opts, routing = make_optimizers(net, group, identity["adam_lr"], identity["muon_lr"])
            scheds = schedulers(opts, identity["horizon"], identity["minimum_lr_ratio"])
            data = data if data is not None else rt.GuardedData()
            x, y = data.load("train", m.c.is_qofn(group))
            xv, yv = data.load("val", m.c.is_qofn(group))
            if not bool(torch.isfinite(y).all()) or not bool(torch.isfinite(yv).all()):
                raise RuntimeError("Nonfinite frozen target data; not an optimizer divergence.")
            k.write(path / "optimizer.json", routing)
            k.write(path / "hardware.json", m.c.environment(torch.device("cpu")))
            setup_seconds = time.perf_counter() - session_start
            routing_verified = None
            resume_verified = False
            if (path / "latest.pt").exists():
                saved = torch.load(path / "latest.pt", map_location="cpu", weights_only=True)
                if saved["identity"] != identity:
                    raise RuntimeError("Checkpoint identity mismatch.")
                load_pack(saved, net, opts, scheds)
                current_step, training_seconds = saved["step"], saved["training_seconds"]
                wall_offset = saved["wall_seconds"]
                timing = path / "latest_timing.json"
                if timing.exists():
                    t = k.read(timing)
                    if t["step"] == current_step and t["checkpoint_sha"] == k.sha(path / "latest.pt"):
                        wall_offset = max(wall_offset, t["wall_seconds"])
                records, step_times = saved["records"], saved["step_times"]
                diagnostics = saved["diagnostics"]
                checkpoint_times = saved["checkpoint_times"]
                observation_times = saved["observation_times"]
                routing_verified = saved.get("routing_verified")
                resume_verified = saved.get("resume_verified", False)
                lr_check(opts, scheds, current_step, identity["horizon"], identity["minimum_lr_ratio"])
                emit(f"resuming complete checkpoint at step {current_step}")

            def save_logs():
                (path / "validation.jsonl").write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in records))
                k.csv_write(path / "validation.csv", records)
                k.csv_write(path / "step_times.csv", step_times)
                (path / "features.jsonl").write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in diagnostics))

            def checkpoint():
                tick = time.perf_counter()
                state = dict(**pack(net, opts, scheds), identity=identity, step=current_step,
                             training_seconds=training_seconds, wall_seconds=wall(), records=records,
                             step_times=step_times, diagnostics=diagnostics,
                             checkpoint_times=checkpoint_times, observation_times=observation_times,
                             routing_verified=routing_verified, resume_verified=resume_verified)
                atomic_torch(path / "latest.pt", state)
                save_logs()
                checkpoint_times.append(time.perf_counter() - tick)
                k.write(path / "latest_timing.json", dict(step=current_step, wall_seconds=wall(),
                                                          checkpoint_sha=k.sha(path / "latest.pt")))

            def observe(loss):
                tick = time.perf_counter()
                value = relative_error(net, group, xv, yv)
                if m.c.is_qofn(group):
                    feature = (diagnostic_fn or feature_diagnostics)(net, xv)
                    # Strict JSON encoding also catches nonfinite feature norms.
                    try:
                        json.dumps(feature, allow_nan=False)
                    except ValueError as e:
                        raise FloatingPointError("Nonfinite QOFN diagnostic feature.") from e
                    diagnostics.append(dict(step=current_step, layers=feature))
                observation_times.append(time.perf_counter() - tick)
                usage = rt.machine_usage()
                row = dict(step=current_step, train_mse=loss, validation_relative_l2=value,
                           training_seconds=training_seconds, wall_seconds=wall(),
                           learning_rates=[o.param_groups[0]["lr"] for o in opts], **usage)
                records.append(row)
                save_logs()
                recent = [r["seconds"] for r in step_times[-500:]]
                first = [r["seconds"] for r in step_times[20:520]]
                k.write(path / "progress.json", dict(status="running", step=current_step,
                    validation_relative_l2=value, wall_seconds=wall(), training_seconds=training_seconds,
                    recent_mean_step_seconds=float(np.mean(recent)) if recent else None,
                    timing_drift_ratio=float(np.mean(recent)/np.mean(first)) if first else None, **usage))
                emit(f"step={current_step}/{identity['total_steps']} val={value:.8g} train_s={training_seconds:.1f} wall_s={wall():.1f}")

            save_logs()  # Remove log rows beyond a recovered complete checkpoint.
            if current_step == 0 and not records:
                observe(None)
                checkpoint()
            # A power loss can occur after the step-100 checkpoint but before
            # committing its reference. Recreate it without bypassing the check.
            if stop_at is not None and current_step == stop_at:
                checkpoint()
                if prepare_resume_reference:
                    update(net, opts, scheds, group, x, y)
                    atomic_torch(path / "resume_reference.pt", pack(net, opts, scheds))
                k.write(path / "progress.json", dict(status="paused_for_resume_check", step=current_step))
                return dict(status="paused", step=current_step)
            for attempt in range(current_step + 1, identity["total_steps"] + 1):
                if stop[0]:
                    checkpoint()
                    raise Interrupted("Stop requested; complete checkpoint saved.")
                tick = time.perf_counter()
                try:
                    if inject_failure is not None:
                        inject_failure(attempt, net)
                    # The dry check uses the first gradient on copies before real update.
                    if routing_check and routing_verified is None:
                        for o in opts:
                            o.zero_grad(set_to_none=True)
                        dry_loss = torch.nn.functional.mse_loss(net(m.c.inputs_for(group, x)), y)
                        dry_loss.backward()
                        routing_verified = check_isolated_routing(net, group, opts)
                        tick = time.perf_counter()
                    loss = update(net, opts, scheds, group, x, y)
                except FloatingPointError:
                    training_seconds += time.perf_counter() - tick
                    raise
                duration = time.perf_counter() - tick
                training_seconds += duration
                current_step = attempt
                step_times.append(dict(step=current_step, seconds=duration))
                if verify_resume and not resume_verified and current_step == 101:
                    reference = torch.load(path / "resume_reference.pt", map_location="cpu", weights_only=True)
                    assert_tree_close(pack(net, opts, scheds), reference)
                    resume_verified = True
                    k.write(path / "resume_verified.json", dict(passed=True, checkpoint_step=100,
                              compared_step=101, tolerance=dict(atol=1e-6, rtol=1e-6),
                              states=["model", "all optimizers", "all schedulers", "Python/NumPy/torch RNG"]))
                if current_step % identity["validation_every"] == 0 or current_step == identity["total_steps"]:
                    lr_check(opts, scheds, current_step, identity["horizon"], identity["minimum_lr_ratio"])
                    observe(loss)
                if current_step % identity["checkpoint_every"] == 0 or current_step == identity["total_steps"]:
                    checkpoint()
                if stop_at is not None and current_step == stop_at:
                    checkpoint()
                    if prepare_resume_reference:
                        update(net, opts, scheds, group, x, y)
                        atomic_torch(path / "resume_reference.pt", pack(net, opts, scheds))
                    k.write(path / "progress.json", dict(status="paused_for_resume_check", step=current_step))
                    return dict(status="paused", step=current_step)
                if stop[0]:
                    checkpoint()
                    raise Interrupted("Stop requested after a complete update; checkpoint saved.")
            atomic_torch(path / "final.pt", net.state_dict())
            lr_check(opts, scheds, current_step, identity["horizon"], identity["minimum_lr_ratio"])
            measured = [r["seconds"] for r in step_times if r["step"] > 20]
            result = dict(identity=identity, status="completed", actual_steps=current_step,
                          records=records, training_seconds=training_seconds, wall_seconds=wall(),
                          parameters=sum(p.numel() for p in net.parameters()), checkpoint_sha=k.sha(path / "final.pt"),
                          final_validation_relative_l2=records[-1]["validation_relative_l2"],
                          mean_step_seconds=float(np.mean(measured)) if measured else None,
                          median_step_seconds=float(np.median(measured)) if measured else None,
                          last_100_mean_seconds=float(np.mean([r["seconds"] for r in step_times[-100:]])),
                          mean_observation_seconds=float(np.mean(observation_times)),
                          mean_checkpoint_seconds=float(np.mean(checkpoint_times)) if checkpoint_times else 0.,
                          setup_seconds=setup_seconds, routing_verified=routing_verified,
                          resume_verified=resume_verified, final_learning_rates=[o.param_groups[0]["lr"] for o in opts])
        except FloatingPointError as error:
            result = dict(identity=identity, status="diverged", actual_steps=current_step,
                          attempted_step=attempt, records=records, reason=str(error),
                          final_validation_relative_l2=None, selection_score=None,
                          selection_score_status="positive_infinity", training_seconds=training_seconds,
                          wall_seconds=wall(), parameters=sum(p.numel() for p in net.parameters()) if net else None)
            emit("DIVERGED: " + str(error))
        except BaseException as error:
            # Unexpected exceptions do not turn into a numerical-failure score.
            k.write(path / "progress.json", dict(status="interrupted", step=current_step,
                                                 reason=type(error).__name__ + ": " + str(error)))
            raise
        finally:
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)
            log.close()
        result["targets"] = k.target_events(result)
        k.write(path / "result.json", result)
        k.write(path / "progress.json", dict(status=result["status"], step=current_step,
                                             wall_seconds=result["wall_seconds"]))
        return result
