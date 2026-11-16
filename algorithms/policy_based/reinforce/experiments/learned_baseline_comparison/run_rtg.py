"""Add reward-to-go to the frozen paired REINFORCE experiment.

Only five new RTG+TD-baseline policies are trained. Existing vanilla and
full-return baseline controls retain their original training logs and selected
checkpoints. All three variants receive a new, common held-out test set.
"""

from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import time
from zoneinfo import ZoneInfo

import gymnasium as gym
import torch

import core
import run as runner


HERE = Path(__file__).resolve().parent
VARIANTS = ["vanilla", "baseline", "rtg_baseline"]
CONTROL_FILES = ["training.csv", "validation.jsonl", "validation_episodes.jsonl", "best.pt", "last.pt"]


def validate_control(control):
    """Validate the reusable evidence before creating any new experiment."""
    protocol = runner.read_json(control / "protocol.json")
    assert protocol["variants"] == ["vanilla", "baseline"], "Expected the original two-variant control run"
    assert protocol["config"] == core.DEFAULT_CONFIG, "Control configuration differs"
    assert protocol["validation_interval"] > 0
    runner.verify_sources(protocol)
    versions = {name: importlib.metadata.version(name) for name in protocol["versions"]}
    assert versions == protocol["versions"], "Dependency versions differ from the controls"
    assert platform.python_version() == protocol["python"], "Python version differs from controls"
    env = gym.make(protocol["env_name"])
    try:
        assert env.spec.max_episode_steps == protocol["environment"]["max_episode_steps"]
        assert env.action_space.low.tolist() == protocol["environment"]["action_low"]
        assert env.action_space.high.tolist() == protocol["environment"]["action_high"]
    finally:
        env.close()
    frozen = runner.read_json(control / "frozen_models.json")["sha256"]
    artifacts = {}
    for seed in protocol["seeds"]:
        for variant in protocol["variants"]:
            folder = control / f"{variant}_seed_{seed}"
            trial = runner.read_json(folder / "trial.json")
            assert trial["status"] == "complete" and trial["episodes"] == protocol["episodes"]
            with (folder / "training.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            assert [int(row["episode"]) for row in rows] == list(range(1, protocol["episodes"] + 1))
            steps = 0
            for row in rows:
                steps += int(row["length"])
                assert steps == int(row["env_steps"])
            assert steps == trial["env_steps"]
            for name in ("best.pt", "last.pt"):
                path = folder / name
                assert runner.digest(path) == frozen[str(path.relative_to(control))], f"Changed control model: {path}"
            for name in CONTROL_FILES + ["trial.json"]:
                path = folder / name
                artifacts[str(path.relative_to(control))] = runner.digest(path)
    artifacts["protocol.json"] = runner.digest(control / "protocol.json")
    artifacts["frozen_models.json"] = runner.digest(control / "frozen_models.json")
    # The archived implementation remains available for auditing historical
    # controls. Behavioral parity tests cover the current full-return branch.
    for name in ("core.py", "run.py"):
        artifacts[f"snapshots/{name}"] = runner.digest(control / "snapshots" / name)
    return protocol, artifacts


def prepare(control, output, workers):
    old, hashes = validate_control(control)
    protocol = copy.deepcopy(old)
    protocol.update({
        "started_at": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "variants": VARIANTS, "trained_variants": ["rtg_baseline"], "probe_variants": ["rtg_baseline"],
        "control_run": str(control), "workers": workers,
        "test_seeds": list(range(80_000, 80_000 + len(old["test_seeds"]))),
        "probe_seeds": list(range(90_000, 90_000 + len(old["probe_seeds"]))),
        "previous_test_seeds_not_reused": old["test_seeds"],
        "source_hashes": {key: runner.digest(core.SOURCE_FILES[key]) for key in VARIANTS},
        "source_paths": {key: str(core.SOURCE_FILES[key].relative_to(runner.ROOT)) for key in VARIANTS},
        "algorithm": "rtg_baseline changes ONLY actor weights from full trajectory R-V(s) to finite-trajectory G_t-V(s); TD critic, gamma=1, losses, optimizers, initialization and action handling unchanged",
        "primary_endpoint": "Final checkpoint stochastic test mean; paired RTG+baseline minus full-return baseline",
        "probe": "Same frozen RTG-trained actor and critic, same 64 trajectories: RTG, RTG-V, and full-return R-V gradient covariance traces",
        "sampling_efficiency": "Previous-validation step interpolation, no extrapolation; compare all variants over the shared environment-step range and report first two-consecutive >=950 attainment with failures retained",
        "control_reuse": "Training trajectories and checkpoint selection reused without retraining or selecting again; all models reevaluated on the new common test set",
    })
    assert set(protocol["test_seeds"]).isdisjoint(old["test_seeds"])
    assert set(protocol["test_seeds"]).isdisjoint(protocol["validation_seeds"])
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "snapshots"
    snapshot.mkdir()
    for key in VARIANTS:
        shutil.copy2(core.SOURCE_FILES[key], snapshot / core.SOURCE_FILES[key].name)
    for path in HERE.glob("*.py"):
        shutil.copy2(path, snapshot / path.name)
    for seed in protocol["seeds"]:
        for variant in ("vanilla", "baseline"):
            origin = control / f"{variant}_seed_{seed}"
            destination = output / origin.name
            destination.mkdir()
            for name in CONTROL_FILES:
                shutil.copy2(origin / name, destination / name)
            trial = runner.read_json(origin / "trial.json")
            trial["reused_from"] = str(origin)
            runner.write_json(destination / "trial.json", trial)
    checks = []
    for seed in protocol["seeds"]:
        before = torch.get_rng_state().clone()
        baseline = core.make_agent("baseline", seed, protocol["config"])
        rtg = core.make_agent("rtg_baseline", seed, protocol["config"])
        same_actor = all(torch.equal(a, b) for a, b in zip(baseline.policy.parameters(), rtg.policy.parameters())) and torch.equal(baseline.log_std, rtg.log_std)
        same_critic = all(torch.equal(a, b) for a, b in zip(baseline.critic.parameters(), rtg.critic.parameters()))
        isolated = torch.equal(before, torch.get_rng_state())
        assert same_actor and same_critic and isolated
        checks.append({"seed": seed, "same_actor_initialization": same_actor,
                       "same_critic_initialization": same_critic, "initialization_rng_isolated": isolated})
    runner.write_json(output / "protocol.json", protocol)
    runner.write_json(output / "control_provenance.json", {"control_run": str(control), "sha256": hashes})
    runner.write_json(output / "initialization_checks.json", checks)
    return protocol


def execute(args):
    torch.set_num_threads(1)
    output, control = args.output.resolve(), args.control_run.resolve()
    started = time.perf_counter()
    protocol = prepare(control, output, args.workers)
    runner.event("rtg_experiment_started", output=str(output), control_run=str(control),
                 episodes=protocol["episodes"], seeds=protocol["seeds"], workers=args.workers)
    tasks = [(str(output), "rtg_baseline", seed, protocol) for seed in protocol["seeds"]]
    trials = runner.execute_tasks(runner.train_trial, tasks, args.workers)
    runner.verify_sources(protocol)
    runner.freeze_models(output, protocol)
    runner.final_evaluation(output, args.workers)
    # Reuse is read-only: verify original evidence again after all new work.
    for relative, expected in runner.read_json(output / "control_provenance.json")["sha256"].items():
        assert runner.digest(control / relative) == expected, f"Original control artifact changed: {relative}"
    runner.write_json(output / "execution.json", {
        "elapsed_seconds": time.perf_counter() - started,
        "completed_runs": 2 * len(protocol["seeds"]) + sum(t["status"] == "complete" for t in trials),
        "total_runs": 3 * len(protocol["seeds"]), "new_training_runs": len(tasks),
        "reused_training_runs": 2 * len(protocol["seeds"]),
        "finished_at": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
    })
    from report_rtg import generate_report
    generate_report(output)
    runner.event("rtg_experiment_finished", output=str(output), elapsed_seconds=round(time.perf_counter() - started, 2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    start = commands.add_parser("run")
    start.add_argument("--control-run", type=Path, default=runner.ROOT / "runs/learned_baseline_comparison/20261116_standard")
    start.add_argument("--output", type=Path, default=runner.ROOT / "runs/learned_baseline_comparison" / datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d-%H%M%S_rtg"))
    start.add_argument("--workers", type=int, default=3)
    report = commands.add_parser("report")
    report.add_argument("output", type=Path)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("output", type=Path)
    evaluate.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if args.command == "report":
        from report_rtg import generate_report
        generate_report(args.output.resolve())
        return
    if args.workers < 1:
        parser.error("workers must be positive")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    if args.command == "run":
        execute(args)
    else:
        runner.final_evaluation(args.output.resolve(), args.workers)


if __name__ == "__main__":
    main()
