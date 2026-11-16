"""Reproducible, paired REINFORCE comparison; run with .venv/bin/python.

The run command trains all candidates before freezing checkpoints and opening
the test set. The original algorithm files are never edited.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback
from zoneinfo import ZoneInfo

import gymnasium as gym
import torch

import core


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
TRAIN_FIELDS = ["episode", "return", "length", "env_steps", "actor_loss", "critic_loss", "std",
                "grad_norm", "value_mean", "terminated", "truncated", "train_seconds"]


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_sources(protocol):
    """Reject post-hoc evaluation under different algorithm source code."""
    for variant in protocol["source_hashes"]:
        path = core.SOURCE_FILES[variant]
        if digest(path) != protocol["source_hashes"][variant]:
            raise RuntimeError(f"Algorithm source differs from the frozen protocol: {path}")


def event(kind, **kwargs):
    print(json.dumps({"event": kind, **kwargs}, ensure_ascii=False, allow_nan=False), flush=True)


def train_trial(task):
    output, variant, seed, protocol = task
    torch.set_num_threads(1)
    folder = Path(output) / f"{variant}_seed_{seed}"
    folder.mkdir(exist_ok=False)
    agent = core.make_agent(variant, seed, protocol["config"])
    torch.manual_seed(20_000 + seed)
    env = gym.make(protocol["env_name"])
    trial = {"variant": variant, "seed": seed, "status": "running", "episodes": 0, "env_steps": 0,
             "train_seconds": 0., "evaluation_seconds": 0., "best_validation_mean": None,
             "best_episode": None, "last_validation_mean": None, "first_pass_episode": None,
             "first_pass_steps": None, "stop_reason": None}
    streak = 0
    started = time.perf_counter()
    with (folder / "training.csv").open("w", newline="") as csv_handle, (folder / "validation.jsonl").open("w") as val_handle, (folder / "validation_episodes.jsonl").open("w") as detail_handle:
        writer = csv.DictWriter(csv_handle, fieldnames=TRAIN_FIELDS)
        writer.writeheader()

        def validate():
            nonlocal streak
            eval_start = time.perf_counter()
            evaluation = core.evaluate(agent, protocol["validation_seeds"])
            trial["evaluation_seconds"] += time.perf_counter() - eval_start
            score = evaluation["summary"]["mean"]
            row = {"episode": trial["episodes"], "env_steps": trial["env_steps"],
                   "train_seconds": trial["train_seconds"], **evaluation["summary"]}
            val_handle.write(json.dumps(row, allow_nan=False) + "\n")
            val_handle.flush()
            detail_handle.write(json.dumps({"episode": trial["episodes"], **evaluation}, allow_nan=False) + "\n")
            detail_handle.flush()
            trial["last_validation_mean"] = score
            if trial["best_validation_mean"] is None or score > trial["best_validation_mean"]:
                trial["best_validation_mean"] = score
                trial["best_episode"] = trial["episodes"]
                core.save_checkpoint(folder / "best.pt", agent, trial["episodes"], trial["env_steps"])
            streak = streak + 1 if score >= 950 else 0
            if streak >= 2 and trial["first_pass_episode"] is None:
                trial["first_pass_episode"] = trial["episodes"]
                trial["first_pass_steps"] = trial["env_steps"]
            write_json(folder / "trial.json", trial)
            event("validation", variant=variant, seed=seed, episode=trial["episodes"],
                  steps=trial["env_steps"], mean=score, best=trial["best_validation_mean"],
                  train_seconds=round(trial["train_seconds"], 2))

        try:
            validate()
            for episode in range(1, protocol["episodes"] + 1):
                train_start = time.perf_counter()
                env_seed = 1_000_000 + seed * 10_000 + episode
                trajectory = core.collect_episode(agent, env, env_seed)
                metrics = core.update(agent, core.tensor_batch(trajectory))
                trial["train_seconds"] += time.perf_counter() - train_start
                trial["episodes"] = episode
                trial["env_steps"] += trajectory["length"]
                writer.writerow({"episode": episode, "return": trajectory["return"],
                                 "length": trajectory["length"], "env_steps": trial["env_steps"],
                                 **metrics, "terminated": bool(trajectory["terminated"][-1]),
                                 "truncated": trajectory["truncated"], "train_seconds": trial["train_seconds"]})
                if episode % protocol["validation_interval"] == 0 or episode == protocol["episodes"]:
                    csv_handle.flush()
                    validate()
            core.save_checkpoint(folder / "last.pt", agent, trial["episodes"], trial["env_steps"])
            trial["status"] = "complete"
            trial["stop_reason"] = "episode_limit"
        except Exception as exc:
            trial["status"] = "failed"
            trial["stop_reason"] = "exception"
            trial["error"] = f"{type(exc).__name__}: {exc}"
            (folder / "error.txt").write_text(traceback.format_exc())
            # A failed run is not silently substituted or evaluated as a full run.
        finally:
            env.close()
            trial["wall_seconds"] = time.perf_counter() - started
            write_json(folder / "trial.json", trial)
    event("training_complete", **trial)
    return trial


def evaluate_trial(task):
    output, variant, seed, protocol, hashes = task
    torch.set_num_threads(1)
    folder = Path(output) / f"{variant}_seed_{seed}"
    result = {"reload_reproduced": True}
    started = time.perf_counter()
    try:
        for name in ("last", "best"):
            path = folder / f"{name}.pt"
            relative = str(path.relative_to(output))
            if digest(path) != hashes[relative]:
                raise RuntimeError(f"Frozen checkpoint changed: {relative}")
            agent, _ = core.load_checkpoint(path)
            stochastic = core.evaluate(agent, protocol["test_seeds"])
            deterministic = core.evaluate(agent, protocol["test_seeds"], deterministic=True)
            reloaded, _ = core.load_checkpoint(path)
            repeated = core.evaluate(reloaded, protocol["test_seeds"][:3])
            reproduced = repeated["episodes"] == stochastic["episodes"][:3]
            result["reload_reproduced"] &= reproduced
            if not reproduced:
                raise AssertionError(f"Checkpoint reload mismatch: {relative}")
            result[name] = {"stochastic": stochastic, "deterministic": deterministic}
        result["test_seconds"] = time.perf_counter() - started
        write_json(folder / "test.json", result)
        if variant in protocol.get("probe_variants", ["baseline"]):
            agent, _ = core.load_checkpoint(folder / "last.pt")
            probe_start = time.perf_counter()
            probe = core.gradient_probe(agent, protocol["probe_seeds"])
            probe["seconds"] = time.perf_counter() - probe_start
            write_json(folder / "gradient_probe.json", probe)
        event("test_complete", variant=variant, seed=seed,
              final_mean=result["last"]["stochastic"]["summary"]["mean"],
              best_mean=result["best"]["stochastic"]["summary"]["mean"])
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        write_json(folder / "test_error.json", result)
        (folder / "test_error.txt").write_text(traceback.format_exc())
        event("test_failed", variant=variant, seed=seed, error=result["error"])
    return result


def execute_tasks(function, tasks, workers):
    context = multiprocessing.get_context("spawn")
    results = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        futures = {pool.submit(function, task): task for task in tasks}
        for future in as_completed(futures):
            results.append(future.result())
    return results


def verify_initialization(seeds, config):
    checks = []
    for seed in seeds:
        before = torch.get_rng_state().clone()
        vanilla = core.make_agent("vanilla", seed, config)
        baseline = core.make_agent("baseline", seed, config)
        same = all(torch.equal(a, b) for a, b in zip(vanilla.policy.parameters(), baseline.policy.parameters())) and torch.equal(vanilla.log_std, baseline.log_std)
        isolated = torch.equal(before, torch.get_rng_state())
        assert same and isolated
        checks.append({"seed": seed, "same_actor_initialization": same, "initialization_rng_isolated": isolated})
    return checks


def prepare_protocol(args, output):
    torch.set_num_threads(1)
    env = gym.make(core.DEFAULT_CONFIG["env_name"])
    try:
        env.reset(seed=0)
        environment = {"max_episode_steps": env.spec.max_episode_steps,
                       "reward_threshold": env.spec.reward_threshold,
                       "action_low": env.action_space.low.tolist(), "action_high": env.action_space.high.tolist()}
    finally:
        env.close()
    protocol = {
        "started_at": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "env_name": core.DEFAULT_CONFIG["env_name"], "environment": environment,
        "episodes": args.episodes, "seeds": args.seeds, "variants": ["vanilla", "baseline"],
        "workers": args.workers, "config": dict(core.DEFAULT_CONFIG),
        "validation_interval": args.validation_interval,
        "validation_seeds": list(range(50_000, 50_000 + args.validation_episodes)),
        "test_seeds": list(range(60_000, 60_000 + args.test_episodes)),
        "probe_seeds": list(range(70_000, 70_000 + args.probe_episodes)),
        "source_hashes": {key: digest(core.SOURCE_FILES[key]) for key in ("vanilla", "baseline")},
        "source_paths": {key: str(core.SOURCE_FILES[key].relative_to(ROOT)) for key in ("vanilla", "baseline")},
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "numpy", "gymnasium", "mujoco", "matplotlib")},
        "python": platform.python_version(), "platform": platform.platform(),
        "rng": {"actor_init": "train_seed", "critic_init": "10000 + train_seed",
                "training_actions": "torch.manual_seed(20000 + train_seed) after initialization",
                "training_environment": "1000000 + 10000 * train_seed + episode (1-based)",
                "evaluation_actions": "100000 + environment_seed; isolated with fork_rng"},
        "algorithm": "Original full-return, single-episode REINFORCE; baseline adds one detached gamma=1 TD MSE update per episode; raw Gaussian actions unchanged",
        "primary_endpoint": "Final checkpoint stochastic test mean, compared within each training seed",
        "attainment": "First second consecutive validation mean >= 950; no early stopping",
        "checkpoint_selection": "Best validation mean; ties retain earliest; tests opened only after all models frozen",
        "probe": "At each final baseline policy, identical held-out trajectories compare covariance traces of sum-per-trajectory gradients with/without the frozen baseline",
    }
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "snapshots"
    snapshot.mkdir()
    for variant in protocol["variants"]:
        path = core.SOURCE_FILES[variant]
        shutil.copy2(path, snapshot / path.name)
    for path in HERE.glob("*.py"):
        shutil.copy2(path, snapshot / path.name)
    write_json(output / "protocol.json", protocol)
    write_json(output / "initialization_checks.json", verify_initialization(args.seeds, protocol["config"]))
    return protocol


def freeze_models(output, protocol):
    frozen = {}
    for seed in protocol["seeds"]:
        for variant in protocol["variants"]:
            folder = output / f"{variant}_seed_{seed}"
            if read_json(folder / "trial.json")["status"] == "complete":
                for name in ("last", "best"):
                    path = folder / f"{name}.pt"
                    frozen[str(path.relative_to(output))] = digest(path)
    write_json(output / "frozen_models.json", {"frozen_at": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(), "sha256": frozen})
    return frozen


def final_evaluation(output, workers):
    protocol = read_json(output / "protocol.json")
    verify_sources(protocol)
    frozen = read_json(output / "frozen_models.json")["sha256"]
    tasks = [(str(output), variant, seed, protocol, frozen)
             for seed in protocol["seeds"] for variant in protocol["variants"]
             if read_json(output / f"{variant}_seed_{seed}" / "trial.json")["status"] == "complete"]
    return execute_tasks(evaluate_trial, tasks, workers)


def run(args):
    output = args.output.resolve()
    started = time.perf_counter()
    protocol = prepare_protocol(args, output)
    event("experiment_started", output=str(output), episodes=args.episodes, seeds=args.seeds, workers=args.workers)
    tasks = [(str(output), variant, seed, protocol) for seed in args.seeds for variant in protocol["variants"]]
    trials = execute_tasks(train_trial, tasks, args.workers)
    verify_sources(protocol)
    freeze_models(output, protocol)
    final_evaluation(output, args.workers)
    write_json(output / "execution.json", {"elapsed_seconds": time.perf_counter() - started,
                                         "completed_runs": sum(t["status"] == "complete" for t in trials),
                                         "total_runs": len(tasks),
                                         "finished_at": datetime.now(ZoneInfo("Asia/Seoul")).isoformat()})
    from report import generate_report
    generate_report(output)
    event("experiment_finished", output=str(output), elapsed_seconds=round(time.perf_counter() - started, 2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    start = commands.add_parser("run", help="Train, freeze, test, and report a new paired experiment")
    start.add_argument("--output", type=Path, default=ROOT / "runs" / "learned_baseline_comparison" / datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d-%H%M%S"))
    start.add_argument("--seeds", nargs="+", type=int, default=[101, 202, 303, 404, 505])
    start.add_argument("--episodes", type=int, default=5000)
    start.add_argument("--workers", type=int, default=3)
    start.add_argument("--validation-interval", type=int, default=100)
    start.add_argument("--validation-episodes", type=int, default=20)
    start.add_argument("--test-episodes", type=int, default=100)
    start.add_argument("--probe-episodes", type=int, default=64)
    report = commands.add_parser("report", help="Regenerate plots and the report without training")
    report.add_argument("output", type=Path)
    evaluate = commands.add_parser("evaluate", help="Reevaluate the already frozen models")
    evaluate.add_argument("output", type=Path)
    evaluate.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if args.command == "run":
        if min(args.episodes, args.workers, args.validation_interval, args.validation_episodes, args.test_episodes) < 1 or args.probe_episodes < 2:
            parser.error("Counts must be positive; the gradient probe needs at least two trajectories")
        if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
            parser.error("Training seeds must be unique nonnegative integers")
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[name] = "1"
        run(args)
    elif args.command == "report":
        from report import generate_report
        generate_report(args.output.resolve())
    else:
        final_evaluation(args.output.resolve(), args.workers)


if __name__ == "__main__":
    main()
