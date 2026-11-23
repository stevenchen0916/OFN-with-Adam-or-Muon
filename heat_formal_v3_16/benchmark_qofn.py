"""One sustained 1000-update timing run; no validation/test or model selection.

Only the training split is loaded. Benchmark weights are discarded. Data and
previous experiments are never overwritten. A completed result is cached.
"""
import argparse
import json
import time
from pathlib import Path

import model_config as m
import numpy as np
import torch

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE/"timing_results")
    args = parser.parse_args()
    out = args.out.resolve()
    with m.c.exclusive(out):
        if (out/"result.json").exists():
            print("Reusing completed timing result", (out/"result.json").read_text(), flush=True)
            return
        torch.set_default_device("cpu")
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        m.c.seed_all(289)
        data = m.c.Dataset(m.V2/"results", torch.device("cpu"))
        net = m.build_model("full_adam")
        optimizer = torch.optim.Adam(net.parameters(),lr=.001,weight_decay=0)
        x,y = data.load("train",True)
        hardware = m.c.environment(torch.device("cpu"))
        m.c.write_json(out/"hardware.json",hardware)
        m.c.write_json(out/"models.json",dict(layers=m.LAYERS,parameters=m.parameter_table(),
                                               matching="within 1% of DeepONet, not exactly equal"))
        identity = dict(purpose="sustained throughput only; not a formal training run",group="full_adam",seed=289,
                        updates=1000,warmup_updates=20,threads=4,device="cpu",batch="full:1800x8000",
                        learning_rate=.001,scheduler=None,weight_decay=0,
                        dataset_id=data.manifest["dataset_id"],code_sha={p.name:m.c.digest(p) for p in HERE.glob("*.py")},
                        helper_sha=m.c.code_hash(),parameters=sum(p.numel() for p in net.parameters()),
                        validation_evaluations=0,test_evaluations=0,benchmark_weights_retained=False)
        m.c.write_json(out/"identity.json",identity)
        m.c.write_json(out/"status.json",dict(state="running",step=0))
        log = (out/"progress.log").open("w",buffering=1)
        def emit(s):
            print(s,flush=True)
            log.write(s+"\n")
        emit("Full QOFN, 29131 parameters, CPU 4 threads, 1000 updates, first 20 warmup; no accuracy evaluation.")
        started = time.perf_counter()
        rows=[]
        try:
            for step in range(1,1001):
                tick=time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                loss=torch.nn.functional.mse_loss(net(x),y)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite timing update; no accuracy result retained.")
                loss.backward()
                optimizer.step()
                duration=time.perf_counter()-tick
                rows.append(dict(step=step,seconds=duration,warmup=step<=20))
                if step%100==0:
                    measured=[r['seconds'] for r in rows if not r['warmup']]
                    m.c.write_csv(out/"step_times.csv",rows)
                    m.c.write_json(out/"status.json",dict(state="running",step=step,
                                       mean_seconds_per_update=float(np.mean(measured))))
                    emit(f"Timing only | {step}/1000 | mean={np.mean(measured):.5f} s/update | wall={time.perf_counter()-started:.1f} s")
            measured=np.array([r["seconds"] for r in rows if not r["warmup"]])
            recent=np.array([r["seconds"] for r in rows if r["step"]>500])
            mean=float(measured.mean())
            sustained=float(recent.mean())
            proxy=max(mean,sustained)
            result=dict(**identity,status="complete",measured_updates=len(measured),
                        mean_step_seconds=mean,median_step_seconds=float(np.median(measured)),
                        p90_step_seconds=float(np.quantile(measured,.9)),last_500_mean_seconds=sustained,
                        wall_seconds=time.perf_counter()-started,
                        timing_scope="forward+MSE+backward+Adam; validation/logging/checkpoints excluded",
                        cost_proxy_seconds_per_update=proxy,
                        estimated_seconds_per_20000_update_run=proxy*20000,
                        qofn_runs=dict(sweep=60,additional_reusing_3_sweep_seeds=28,additional_using_10_new_seeds=40),
                        proxy_qofn_hours_reuse_3_sweep_seeds=proxy*20000*88/3600,
                        proxy_qofn_hours_10_new_seeds=proxy*20000*100/3600,
                        caveats=["Only Full QOFN+Adam measured; No-Flow and Muon times not measured.",
                                 "1000 steps do not establish convergence or all-day thermal throughput.",
                                 "20000-step cost excludes validation, checkpoints, final evaluation and other architectures."])
            m.c.write_json(out/"result.json",result)
            m.c.write_json(out/"status.json",dict(state="completed",step=1000))
            emit(json.dumps(result,indent=2))
        except BaseException as error:
            m.c.write_csv(out/"step_times.csv",rows)
            m.c.write_json(out/"status.json",dict(state="stopped",step=len(rows),reason=str(error)))
            raise
        finally:
            log.close()


if __name__=="__main__":
    main()
