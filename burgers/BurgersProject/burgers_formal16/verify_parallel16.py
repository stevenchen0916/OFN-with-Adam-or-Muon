"""Synthetic checks for the execution-only 16 CPU amendment."""
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import control as k
import runtime as rt
import runner


class ParallelExecution(unittest.TestCase):
    def test_four_workers_four_threads_resource_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);rt.initialize(out)
            k.write(out/'resources.json',dict(workers=4,threads_per_worker=4,device='cpu'))
            ctx=rt.context(out)
            self.assertEqual(ctx['resources']['workers']*ctx['resources']['threads_per_worker'],16)
            self.assertEqual(ctx['protocol_id'],k.FROZEN_ID)
            k.write(out/'resources.json',dict(workers=4,threads_per_worker=2,device='cpu'))
            with self.assertRaises(RuntimeError):rt.resources(out)

    def test_four_worker_budget_retains_whole_seed_barrier(self):
        p=k.protocol();durations={g:1. for g in p['groups']}
        budget=k.simulate_budget(durations,p,4)
        self.assertEqual(budget['baseline_runs'],225)
        self.assertEqual(budget['formal_seconds'],30.)
        self.assertLess(budget['total_seconds'],225.)
        order=k.final_order(p)
        for i,seed in enumerate(p['formal']['seeds']):
            self.assertEqual({s for s,g in order[i*9:(i+1)*9]},{seed})

    def test_dispatch_has_four_slots_and_never_duplicates_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);rt.initialize(out)
            k.write(out/'resources.json',dict(workers=4,threads_per_worker=4,device='cpu'))
            jobs=[dict(path=str(out/f'job_{i}'),identity=dict(group='synthetic',seed=i)) for i in range(9)]
            done=set();seen=[];counter=dict(active=0,peak=0)
            class Process:
                def __init__(self,command):
                    job=k.read(command[command.index('--spec')+1]);self.name=job['path']
                    self.pid=100+len(seen);self.calls=0;self.finished=False
                    seen.append(self.name);counter['active']+=1
                    counter['peak']=max(counter['peak'],counter['active'])
                def poll(self):
                    self.calls+=1
                    if self.calls>=2 and not self.finished:
                        self.finished=True;done.add(self.name);counter['active']-=1
                    return 0 if self.finished else None
                def wait(self):
                    while self.poll() is None:pass
                    return 0
                def send_signal(self,sig):pass
            with patch('runtime.spawn',side_effect=Process),patch('runner.time.sleep'):
                runner.dispatch(out,lambda:[j for j in jobs if j['path'] not in done])
            self.assertEqual(counter['peak'],4)
            self.assertEqual(len(seen),9)
            self.assertEqual(len(set(seen)),9)

    def test_backup_restores_isolated_16cpu_layout(self):
        with tempfile.TemporaryDirectory() as tmp,tempfile.TemporaryDirectory() as backup_dir:
            root=Path(tmp);rt.initialize(root)
            runner.backup(root,Path(backup_dir))
            archive=next(Path(backup_dir).glob('*.zip'))
            with zipfile.ZipFile(archive) as z:
                names=set(z.namelist())
                self.assertIn('burgers_formal16/protocol.json',names)
                self.assertIn('results16/manifest.json',names)
                self.assertIn('run16.sh',names)
                self.assertNotIn('heat_formal_v3/protocol.json',names)


if __name__=='__main__':unittest.main(verbosity=2)
