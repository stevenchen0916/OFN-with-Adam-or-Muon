"""Optional <=10-minute resource test; training data only, no retained weights."""
import argparse
from pathlib import Path
import signal
import subprocess
import sys
import time

import control as k
import model_config as m
import runtime as rt
import engine
import torch


def worker(root, index, threads):
    root=Path(root)
    rt.configure(threads)
    m.c.seed_all(289)
    net=m.build_model("full_adam")
    opts=[torch.optim.Adam(net.parameters(),lr=.001,weight_decay=0)]
    data=rt.GuardedData()
    x,y=data.load("train",True)
    # No schedule is necessary for a discarded throughput-only model.
    for _ in range(20):
        engine.update(net,opts,[],"full_adam",x,y)
    k.write(root/f"ready_{index}.json",dict(ready=True,pid=__import__('os').getpid()))
    gate=root/"gate.json"
    while not gate.exists():
        time.sleep(.02)
    timing=k.read(gate)
    n=0
    start=time.perf_counter()
    while time.perf_counter()<timing["deadline"]:
        engine.update(net,opts,[],"full_adam",x,y)
        n+=1
    k.write(root/f"worker_{index}.json",dict(updates=n,start=start,end=time.perf_counter(),
             threads=threads,machine=rt.machine_usage(),validation_evaluations=0,test_evaluations=0,
             weights_retained=False))


def compare(out, seconds):
    if not 90 <= seconds <= 600:
        raise ValueError("Resource-test total budget must be between 90 and 600 seconds.")
    root=Path(out)/"throughput"/str(time.time_ns())
    root.mkdir(parents=True)
    overall=time.perf_counter()
    hard_deadline=overall+seconds
    measure=(seconds-60)/2
    arms=[]
    processes=[]
    try:
        for count,threads in [(1,4),(2,2)]:
            arm=root/f"workers_{count}_threads_{threads}"
            arm.mkdir()
            processes=[rt.spawn([sys.executable,"-u",str(Path(__file__).resolve()),
                         "--root",str(arm),"--index",str(i),"--threads",str(threads)]) for i in range(count)]
            setup_deadline=min(hard_deadline-measure,time.perf_counter()+30)
            while not all((arm/f"ready_{i}.json").exists() for i in range(count)):
                if any(p.poll() is not None for p in processes) or time.perf_counter()>setup_deadline:
                    raise RuntimeError("Throughput setup failed/exceeded allotted warmup; no resource choice made.")
                time.sleep(.05)
            start=time.perf_counter()
            k.write(arm/"gate.json",dict(start=start,deadline=start+measure))
            while any(p.poll() is None for p in processes):
                if time.perf_counter()>hard_deadline+2:
                    raise RuntimeError("Throughput budget exceeded; test stopped.")
                time.sleep(.1)
            if any(p.returncode!=0 for p in processes):
                raise RuntimeError("Resource worker failed.")
            values=[k.read(arm/f"worker_{i}.json") for i in range(count)]
            elapsed=max(v["end"] for v in values)-start
            value=dict(workers=count,threads_per_worker=threads,updates=sum(v["updates"] for v in values),
                       measurement_wall_seconds=elapsed,worker_results=values)
            value["aggregate_updates_per_second"]=value["updates"]/elapsed
            arms.append(value)
            print(f"Resource {count}x{threads}: {value['aggregate_updates_per_second']:.5f} total updates/s",flush=True)
        comparison=dict(arms=arms,two_by_two_throughput_ratio=arms[1]["aggregate_updates_per_second"]/arms[0]["aggregate_updates_per_second"],
                        actual_wall_seconds=time.perf_counter()-overall,
                        limitation="Representative Full+Adam workload, not all nine configurations; inspect swap and memory too.",
                        resources_automatically_changed=False)
        k.write(root/"report.json",comparison)
        k.write(Path(out)/"throughput_latest.json",dict(report=str(root/"report.json"),**comparison))
        print(json_text(comparison),flush=True)
        return comparison
    finally:
        for proc in processes:
            if proc.poll() is None:
                rt.interrupt_process(proc)
        for proc in processes:
            proc.wait()


def json_text(value):
    import json
    return json.dumps(value,ensure_ascii=False,indent=2)


if __name__=="__main__":
    a=argparse.ArgumentParser();a.add_argument("--root",type=Path,required=True)
    a.add_argument("--index",type=int,required=True);a.add_argument("--threads",type=int,required=True)
    args=a.parse_args()
    worker(args.root,args.index,args.threads)
