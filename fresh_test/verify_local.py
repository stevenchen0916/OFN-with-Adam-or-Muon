"""Local checks use original data/dry-run models, never the new replication seeds."""
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory

import replicate as r


def main():
    workspace = Path(sys.argv[1]).resolve()
    with TemporaryDirectory(prefix="heat-fresh-verification-") as temporary:
        temp = Path(temporary)
        (temp / "heat_formal_v3_16").symlink_to(workspace / "heat_formal_v3", target_is_directory=True)
        (temp / "heat_fair_v2").symlink_to(workspace / "heat_fair_v2", target_is_directory=True)
        m, a, k, torch = r.imports(temp)
        import numpy as np
        import engine
        out = temp / "verification_output"
        (out / "data").mkdir(parents=True)
        shutil.copy2(workspace / "heat_fair_v2/results/data/val.npz", out / "data/test.npz")
        data_sha = r.digest(out / "data/test.npz")

        # Reconstruct an original known sample, including physical coordinate order,
        # boundary projection and its training-only PCA, using the old data seeds.
        _, _, _, kappa, initial, _, queries = r.raw_inputs(temp, 200, 101234, 105678)
        _, numerical = r.solve_one((str(temp), 0, float(kappa[0, 0]), initial[0]))
        with np.load(workspace / "heat_fair_v2/results/data/preprocessing.npz") as fitted:
            coeff = ((initial-fitted["pca_mean"]) @ fitted["pca_components"].T).astype(np.float32)
        with np.load(workspace / "heat_fair_v2/results/data/test.npz") as old:
            np.testing.assert_array_equal(kappa, old["x1"])
            np.testing.assert_allclose(coeff, old["x2"], atol=2e-6, rtol=2e-6)
            np.testing.assert_array_equal(queries, old["xt"])
            np.testing.assert_array_equal(numerical, old["y"][0])
        r.validate_arrays(queries, numerical[None, :])
        try:
            r.validate_arrays(queries[:, [0, 2, 1]], numerical[None, :])
        except RuntimeError:
            pass
        else:
            raise AssertionError("The physical coordinate-order check did not detect y/t exchange.")

        paths = list((workspace / "heat_formal_v3/results/dry_run/d0720dd93bd6").glob("*/**/final.pt"))
        assert len(paths) == 9
        jobs, expected = [], {}
        data = m.c.Dataset(m.V2 / "results", torch.device("cpu"))
        for path in paths:
            group = path.relative_to(workspace / "heat_formal_v3/results/dry_run/d0720dd93bd6").parts[0]
            job = dict(group=group, seed=289, checkpoint=str(path), checkpoint_sha=r.digest(path))
            jobs.append(job)
            m.c.seed_all(289)
            net = m.build_model(group)
            net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            x, y = data.load("val", m.c.is_qofn(group))
            expected[group] = engine.relative_error(net, group, x, y)
        with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context("spawn"),
                                 initializer=r.init_evaluator, initargs=(str(temp), str(out), "fixture", data_sha)) as pool:
            rows = list(pool.map(r.score_one, jobs))
        assert len(rows) == 9
        for row in rows:
            assert abs(row["test_relative_l2"] - expected[row["group"]]) < 1e-12
            assert len(row["per_solution_relative_l2"]) == 200

        # A second evaluation must reuse all 90 committed scores. A mismatched
        # checkpoint identity must fail rather than silently reuse a score.
        fixture = dict(jobs=[dict(job, seed=seed) for seed in range(292, 302) for job in jobs])
        fixture_key = r.object_hash(fixture)
        sample_rows = {row["group"]: row for row in rows}
        for job in fixture["jobs"]:
            row = dict(sample_rows[job["group"]], seed=job["seed"], frozen_inputs_sha=fixture_key)
            r.write(out / "scores" / f"{job['group']}_seed_{job['seed']}.json", row)
        cached = r.evaluate(temp, out, fixture, dict(test_sha=data_sha))
        assert len(cached) == 90
        first = fixture["jobs"][0]
        bad_path = out / "scores" / f"{first['group']}_seed_{first['seed']}.json"
        bad = r.read(bad_path)
        bad["checkpoint_sha"] = "changed"
        r.write(bad_path, bad)
        try:
            r.evaluate(temp, out, fixture, dict(test_sha=data_sha))
        except RuntimeError:
            pass
        else:
            raise AssertionError("Mismatched score cache was accepted.")

        # Small non-replication-seed data fixture tests parallel generation and
        # committed-data reuse without opening any fresh final-test inputs.
        fixture_out = temp / "generation_fixture"
        fixture_out.mkdir()
        fixture_frozen = dict(protocol=dict(n_test=4, kappa_seed=2026111701,
                                           initial_field_seed=2026111702, workers=4),
                              preprocessing_sha=r.digest(workspace / "heat_fair_v2/results/data/preprocessing.npz"))
        generated = r.generate(temp, fixture_out, fixture_frozen)
        original_mtime = (fixture_out / "data/test.npz").stat().st_mtime_ns
        assert r.generate(temp, fixture_out, fixture_frozen) == generated
        assert (fixture_out / "data/test.npz").stat().st_mtime_ns == original_mtime

        # Check the statistical helper against the manuscript's rounded seed data.
        full_a = [5.806,5.483,5.658,6.232,5.844,5.839,5.599,5.894,4.964,5.206]
        full_m = [3.712,3.573,3.197,3.984,3.579,3.335,3.311,3.263,3.344,3.081]
        nf_a = [5.455,5.147,5.175,5.236,4.997,5.034,4.928,5.129,5.059,4.467]
        nf_m = [3.978,3.697,3.552,3.925,3.259,3.693,3.437,3.616,3.332,3.323]
        contrasts = np.log(np.array(full_m)/full_a)-np.log(np.array(nf_m)/nf_a)
        summary = a.contrast_summary(contrasts)
        assert abs(summary["mean"] + .1509) < .0001
        assert abs(summary["wilcoxon"]["pvalue"] - .00390625) < 1e-12

        # Freeze must reject missing/ineligible originals before any fresh data.
        incomplete = temp / "incomplete"
        incomplete.mkdir()
        try:
            r.freeze(temp, incomplete, out)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("Missing original formal outputs were accepted.")

        report = dict(status="passed", new_replication_data_generated=False,
                      known_old_sample_and_pca_reproduced=True,
                      physical_y_t_exchange_detected=True,
                      nine_dry_run_models_parallel_scores_match_original_evaluator=True,
                      per_solution_scores=200, original_statistical_result_reproduced=True,
                      all_90_committed_score_caches_reused=True,
                      mismatched_score_cache_rejected=True,
                      four_sample_parallel_generation_and_data_resume_passed=True,
                      missing_original_outputs_rejected=True,
                      scope="Local data and dry-run checkpoints; not the server's formal 90-checkpoint replication")
        r.write(Path(__file__).parent / "LOCAL_VERIFICATION.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
