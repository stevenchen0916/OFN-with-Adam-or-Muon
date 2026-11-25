"""Synthetic-data checks only. Never opens the real train/val/test NPZ files."""
import copy
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import control as k
import model_config as m
import runtime as rt
import engine
import runner
import analysis
import numpy as np
import torch
from deepxde.nn.pytorch.deeponet import DeepONetCartesianProd
from deepxde.nn.pytorch.mionet import MIONetCartesianProd
from models.qofn import QOFN


def tiny(group):
    if group.startswith("deeponet"):
        return DeepONetCartesianProd([11,5,4],[3,5,4],"relu","Glorot normal")
    if group.startswith("mionet"):
        return MIONetCartesianProd([1,3,4],[10,5,4],[3,5,4],"relu","Glorot normal")
    return QOFN([2,4,4],[11,4,4],[4,4,4],"relu",flow=group.startswith("full"))


class SyntheticData:
    def __init__(self):
        self.manifest={"dataset_id":k.protocol()["data"]["dataset_id"]}
        self.opened=[]
        g=torch.Generator().manual_seed(123)
        self.raw=(torch.rand(3,1,generator=g),torch.rand(3,10,generator=g),torch.rand(7,3,generator=g))
        self.augmented=tuple(torch.cat([x,torch.ones(len(x),1)],1) for x in self.raw)
        self.target=.2+self.raw[0]@(.5+self.raw[2][:,0]).reshape(1,-1)

    def load(self,split,qofn):
        self.opened.append(split)
        return (self.augmented if qofn else self.raw),self.target


def unit_identity(group,seed=289,total=20):
    i=rt.run_identity({"synthetic":True},"unit",group,seed,.001,.001 if k.hybrid(group) else None,total)
    i.update(horizon=total,validation_every=5,checkpoint_every=5)
    return i


def fake_terminal(out,job,status="completed",value=.05,real_model=False):
    path=Path(job["path"]);path.mkdir(parents=True,exist_ok=True)
    identity=job["identity"]
    k.write(path/"identity.json",identity)
    records=[dict(step=s,validation_relative_l2=value,training_seconds=float(s),wall_seconds=float(s)+1)
             for s in [19000,19500,20000]]
    r=dict(identity=identity,status=status,actual_steps=20000 if status=="completed" else 7,
           records=records,training_seconds=20000.,wall_seconds=20001.,parameters=100,
           targets=[],final_validation_relative_l2=value if status=="completed" else None)
    if status=="completed":
        if real_model:
            m.c.seed_all(identity["seed"])
            torch.save(tiny(identity["group"]).state_dict(),path/"final.pt")
        else:
            (path/"final.pt").write_bytes(b"synthetic-control-fixture-not-a-real-training-checkpoint")
        r["checkpoint_sha"]=k.sha(path/"final.pt")
    else:
        r.update(reason="synthetic injected NaN",selection_score=None,selection_score_status="positive_infinity")
    r["targets"]=k.target_events(r)
    k.write(path/"result.json",r)
    return r


class Correctness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rt.configure(1)
        cls.original_protocol_sha=k.sha(k.HERE/"protocol.json")

    @classmethod
    def tearDownClass(cls):
        assert k.sha(k.HERE/"protocol.json")==cls.original_protocol_sha

    def test_frozen_id_and_seed_barrier(self):
        p=k.protocol();order=k.final_order(p)
        self.assertEqual(len(order),90)
        for i,seed in enumerate(range(292,302)):
            block=order[9*i:9*i+9]
            self.assertEqual({s for s,g in block},{seed})
            self.assertEqual({g for s,g in block},set(p["groups"]))

    def test_lr_selection_seed_median_then_mean(self):
        rows=[]
        for seed,vals in zip([289,290,291],[[.1,.2,.001],[.3,.2,.001],[.4,.3,.001]]):
            rows.append(dict(identity={"seed":seed},status="completed",records=[
                dict(step=s,validation_relative_l2=v) for s,v in zip([19000,19500,20000],vals)]))
        self.assertAlmostEqual(k.candidate_score(rows)["score"],.2)
        with self.assertRaises(RuntimeError):
            k.candidate_score(rows[:2])
        rows[0]["records"].pop()
        with self.assertRaises(RuntimeError):
            k.candidate_score(rows)

    def test_infinite_failure_is_not_missing_or_omitted(self):
        rs=[dict(identity={"seed":s},status="diverged" if s==290 else "completed") for s in [289,290,291]]
        score=k.candidate_score(rs)
        self.assertIsNone(score["score"])
        self.assertEqual(score["score_status"],"positive_infinity")
        rs[1]["status"]="interrupted"
        with self.assertRaises(RuntimeError):
            k.candidate_score(rs)

    def test_endpoint_extensions_both_directions_and_ties(self):
        def table(vals):
            return [dict(rate=r,score=v,score_status="finite" if v is not None else "positive_infinity") for r,v in vals]
        self.assertEqual(k.choose_candidate(table([(1.,2.),(3.,1.)]))[1],9.)
        self.assertAlmostEqual(k.choose_candidate(table([(1.,1.),(3.,2.)]))[1],1/3)
        best,ext=k.choose_candidate(table([(1.,None),(3.,1.),(9.,None)]))
        self.assertEqual(best["rate"],3.);self.assertIsNone(ext)
        self.assertIsNone(k.choose_candidate(table([(1.,None),(3.,None)]))[0])
        self.assertEqual(k.choose_candidate(table([(1.,1.),(3.,1.),(9.,2.)]))[0]["rate"],1.)

    def test_guard_closes_test_before_provider_load(self):
        provider=SyntheticData();data=rt.GuardedData(provider=provider)
        with self.assertRaises(RuntimeError):
            data.load("test",False)
        self.assertEqual(provider.opened,[])

    def test_all_nine_small_models_resume_and_routing(self):
        for group in k.protocol()["groups"]:
            with self.subTest(group=group),tempfile.TemporaryDirectory() as tmp:
                identity=rt.run_identity({"synthetic":True},"dry_run/synthetic",group,289,.001,
                                        .001 if k.hybrid(group) else None,200)
                provider=SyntheticData();guard=rt.GuardedData(provider=provider)
                first=engine.train(tmp,identity,data=guard,factory=tiny,stop_at=100,
                    prepare_resume_reference=True,routing_check=True,handle_signals=False)
                self.assertEqual(first["status"],"paused")
                if group=="full_muon_adam":
                    proc=subprocess.run([sys.executable,str(Path(__file__).resolve()),"--toy-child",tmp],
                                        capture_output=True,text=True)
                    self.assertEqual(proc.returncode,0,proc.stdout+proc.stderr)
                    result=k.read(Path(tmp)/"result.json")
                else:
                    result=engine.train(tmp,identity,data=guard,factory=tiny,verify_resume=True,
                                        routing_check=True,handle_signals=False)
                self.assertTrue(result["resume_verified"])
                self.assertTrue(result["routing_verified"]["partition_exact"])
                self.assertEqual(result["actual_steps"],200)
                self.assertNotIn("test",provider.opened)
                state=torch.load(Path(tmp)/"latest.pt",weights_only=True)
                self.assertTrue(all(s["last_epoch"]==200 for s in state["schedulers"]))
                before=list(provider.opened)
                engine.train(tmp,identity,data=guard,factory=tiny,handle_signals=False)
                self.assertEqual(provider.opened,before)

    def test_full_cosine_endpoint_both_optimizers(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=engine.train(tmp,unit_identity("deeponet_muon_adam"),factory=tiny,
                           data=rt.GuardedData(provider=SyntheticData()),handle_signals=False)
            for lr in r["final_learning_rates"]:
                self.assertAlmostEqual(lr,1e-5,places=14)

    def test_numerical_failure_is_terminal_for_formal_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            identity=unit_identity("deeponet_adam",292)
            identity["checkpoint_every"]=1
            calls=[]
            def inject(step,net):
                calls.append(step)
                if step==3:
                    raise FloatingPointError("injected NaN")
            kwargs=dict(factory=tiny,data=rt.GuardedData(provider=SyntheticData()),handle_signals=False)
            r=engine.train(tmp,identity,inject_failure=inject,**kwargs)
            self.assertEqual(r["status"],"diverged")
            self.assertEqual(r["identity"]["seed"],292)
            self.assertFalse((Path(tmp)/"final.pt").exists())
            old=list(calls)
            engine.train(tmp,identity,inject_failure=inject,**kwargs)
            self.assertEqual(old,calls)
            saved=torch.load(Path(tmp)/"latest.pt",weights_only=True)
            self.assertEqual(saved["step"],2)

    def test_interruption_resume_matches_uninterrupted(self):
        with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b:
            i=unit_identity("deeponet_muon_adam")
            engine.train(a,i,factory=tiny,data=rt.GuardedData(provider=SyntheticData()),handle_signals=False)
            def interrupt(step,net):
                if step==7:
                    raise engine.Interrupted("synthetic power interruption")
            with self.assertRaises(engine.Interrupted):
                engine.train(b,i,factory=tiny,data=rt.GuardedData(provider=SyntheticData()),
                             inject_failure=interrupt,handle_signals=False)
            engine.train(b,i,factory=tiny,data=rt.GuardedData(provider=SyntheticData()),handle_signals=False)
            sa=torch.load(Path(a)/"latest.pt",weights_only=True)
            sb=torch.load(Path(b)/"latest.pt",weights_only=True)
            for key in ["model","optimizers","schedulers","rng"]:
                engine.assert_tree_close(sa[key],sb[key])

    def test_target_times_censor_and_failure_separate(self):
        r=dict(status="completed",wall_seconds=200.,records=[dict(step=0,validation_relative_l2=.07,wall_seconds=1.,training_seconds=0.),
            dict(step=500,validation_relative_l2=.055,wall_seconds=100.,training_seconds=90.)])
        events=k.target_events(r)
        self.assertEqual(events[0]["first_step"],500)
        self.assertEqual(events[1]["censor_wall_seconds"],200.)
        r["status"]="diverged"
        failed=k.target_events(r)
        self.assertIsNone(failed[1]["censor_wall_seconds"])
        self.assertEqual(failed[1]["failure_wall_seconds"],200.)

    def test_log_interaction_and_exact_wilcoxon(self):
        self.assertAlmostEqual(math.log(.072/.08)-math.log(.036/.04),0.)
        self.assertEqual(analysis.signed_rank_exact([0.,0.])["pvalue"],1.)
        self.assertEqual(analysis.signed_rank_exact([1.,1.,0.])["pvalue"],.5)
        from scipy.stats import wilcoxon,ttest_1samp
        x=[.1,.23,-.31,.4,-.52,.67,.72,-.89,.91,1.07]
        self.assertAlmostEqual(analysis.signed_rank_exact(x)["pvalue"],float(wilcoxon(x,method="exact").pvalue))
        self.assertAlmostEqual(analysis.contrast_summary(x)["paired_t_p"],float(ttest_1samp(x,0).pvalue))

    def test_failure_cannot_be_silently_dropped_from_primary(self):
        p=k.protocol();rows=[]
        for seed in p["formal"]["seeds"]:
            for g in p["groups"]:
                bad=seed==292 and g=="full_muon_adam"
                rows.append(dict(seed=seed,group=g,status="diverged" if bad else "completed",
                    test_relative_l2=None if bad else .04+.0001*(seed-292)+(0.001 if g.endswith("_adam") else 0),
                    parameters=100,training_seconds=1.,wall_seconds=2.,targets=[]))
        with tempfile.TemporaryDirectory() as tmp:
            primary=analysis.analyze(tmp,p,rows)
            self.assertEqual(primary["n"],9)
            self.assertIsNone(primary["confirmatory_t_p"])
            self.assertEqual(primary["missing_seeds"],[292])

    def test_budget_and_architecture_local_dependency(self):
        p=k.protocol();d={g:1. for g in p["groups"]}
        self.assertEqual(k.simulate_budget(d,p,1)["total_seconds"],225.)
        self.assertEqual(k.simulate_budget(d,p,2)["formal_seconds"],50.)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rt.initialize(root)
            jobs,_=runner.refresh_search(root,["deeponet"])
            self.assertEqual(len(jobs),15)
            for job in jobs:
                rate=job["identity"]["adam_lr"]
                fake_terminal(root,job,value=.02+.01*math.log(rate/.001)**2)
            ready,_=runner.refresh_search(root,["deeponet"])
            self.assertEqual(len(ready),30)
            self.assertEqual({j["identity"]["group"] for j in ready},{"deeponet_muon_adam","deeponet_output_muon_adam"})
            self.assertFalse((root/"search/mionet_adam.selected.json").exists())

    def test_all_invalid_search_remains_unresolved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rt.initialize(root)
            jobs,_=runner.refresh_search(root,["mionet"])
            for job in jobs:
                fake_terminal(root,job,status="diverged")
            ready,unresolved=runner.refresh_search(root,["mionet"])
            self.assertFalse(ready);self.assertEqual(unresolved,["mionet_adam"])
            self.assertIsNone(runner.search_choice(root,"mionet_adam"))

    def test_evaluation_gate_cache_and_formal_seed_barrier(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);rt.initialize(out)
            p=k.protocol();configs={g:dict(adam_lr=.001,muon_lr=.001 if k.hybrid(g) else None) for g in p["groups"]}
            selected=dict(configs=configs)
            k.write(out/"selection.json",selected)
            fake=SyntheticData()
            with patch("runner.selection",return_value=selected):
                with self.assertRaises(RuntimeError):
                    runner.evaluate(out,provider=fake,factory=tiny)
                self.assertEqual(fake.opened,[])
                for g in p["groups"][:-1]:
                    lr=configs[g]
                    job=runner.spec(out,"formal",g,292,lr["adam_lr"],lr["muon_lr"])
                    fake_terminal(out,job,real_model=True)
                with self.assertRaises(RuntimeError):
                    runner.evaluate(out,True,"synthetic interruption",provider=fake,factory=tiny)
                self.assertEqual(fake.opened,[])
                g=p["groups"][-1];lr=configs[g]
                fake_terminal(out,runner.spec(out,"formal",g,292,lr["adam_lr"],lr["muon_lr"]),real_model=True)
                next_jobs=runner.formal_jobs(out,selected)
                self.assertEqual({j["identity"]["seed"] for j in next_jobs},{293})
                rows=runner.evaluate(out,True,"synthetic interruption",provider=fake,factory=tiny)
                self.assertEqual(len(rows),9)
                self.assertIn("test",fake.opened)
                second=SyntheticData()
                again=runner.evaluate(out,True,"synthetic interruption",provider=second,factory=tiny)
                self.assertEqual(rows,again)
                self.assertEqual(second.opened,[])
                first_group=p["groups"][0];lr=configs[first_group]
                path=Path(runner.spec(out,"formal",first_group,292,lr["adam_lr"],lr["muon_lr"])["path"])
                (path/"final.pt").write_bytes(b"tampered checkpoint")
                unopened=SyntheticData()
                with self.assertRaises(RuntimeError):
                    runner.evaluate(out,True,"synthetic interruption",provider=unopened,factory=tiny)
                self.assertEqual(unopened.opened,[])

    def test_actual_search_extension_then_diverged_outer_point(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);rt.initialize(out)
            jobs,_=runner.refresh_search(out,["deeponet"])
            for job in jobs:
                fake_terminal(out,job,value=1/job["identity"]["adam_lr"])
            extra,_=runner.refresh_search(out,["deeponet"])
            self.assertEqual(len(extra),3)
            self.assertTrue(all(j["identity"]["adam_lr"]==.03 for j in extra))
            for job in extra:
                fake_terminal(out,job,status="diverged")
            ready,_=runner.refresh_search(out,["deeponet"])
            self.assertEqual(len(ready),30)
            self.assertEqual(runner.search_choice(out,"deeponet_adam")["adam_lr"],.01)

    def test_missing_synthetic_verification_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);_,manifest=rt.initialize(out)
            with self.assertRaises(RuntimeError):
                runner.require_unit_verification(out)
            value=dict(passed=True,code=manifest["code"],protocol_id=k.FROZEN_ID,
                       test_source_sha={name:k.sha(k.HERE/name) for name in ["verify.py","verify_runner.py","verify_platform.py","verify_parallel16.py","verify_burgers_data.py"]})
            k.write(out/"unit_verification.json",value)
            runner.require_unit_verification(out)
            value["code"]={"changed":True};k.write(out/"unit_verification.json",value)
            with self.assertRaises(RuntimeError):
                runner.require_unit_verification(out)

    def test_diverged_formal_seed_block_is_not_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);rt.initialize(out);p=k.protocol()
            configs={g:dict(adam_lr=.001,muon_lr=.001 if k.hybrid(g) else None) for g in p["groups"]}
            chosen=dict(configs=configs)
            for group,lr in configs.items():
                fake_terminal(out,runner.spec(out,"formal",group,292,lr["adam_lr"],lr["muon_lr"]),status="diverged")
            pending=runner.formal_jobs(out,chosen)
            self.assertEqual(len(pending),9)
            self.assertEqual({j["identity"]["seed"] for j in pending},{293})

    def test_power_window_between_checkpoint_and_dry_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            identity=rt.run_identity({"synthetic":True},"dry_run/synthetic","deeponet_muon_adam",289,.001,.001,200)
            def train(**kwargs):
                return engine.train(tmp,identity,factory=tiny,
                                    data=rt.GuardedData(provider=SyntheticData()),handle_signals=False,**kwargs)
            train(stop_at=100)
            self.assertFalse((Path(tmp)/"resume_reference.pt").exists())
            paused=train(stop_at=100,prepare_resume_reference=True)
            self.assertEqual(paused["status"],"paused")
            self.assertTrue((Path(tmp)/"resume_reference.pt").exists())
            self.assertTrue(train(verify_resume=True)["resume_verified"])


if __name__=="__main__":
    if len(sys.argv)>1 and sys.argv[1]=="--toy-child":
        rt.configure(1)
        path=Path(sys.argv[2]);identity=k.read(path/"identity.json")
        engine.train(path,identity,data=rt.GuardedData(provider=SyntheticData()),factory=tiny,
                     verify_resume=True,routing_check=True,handle_signals=False)
    else:
        unittest.main(verbosity=2)
