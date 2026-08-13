"""Vanilla REINFORCE for bounded continuous actions on Pendulum-v1.

The policy uses a tanh-squashed Gaussian. Each episode produces reward-to-go
values G_t, and the strict discounted start-state objective is optimized with
loss = -sum_t gamma**t * G_t * log pi(a_t | s_t).

This deliberately remains vanilla REINFORCE: one complete trajectory per
update and no learned baseline. Optional within-trajectory return
standardization is provided as an explicitly recorded experiment switch.
"""

import argparse
import csv
import json
import math
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass,field
from datetime import datetime
from pathlib import Path

import numpy as np

# Required by deterministic CUDA matrix multiplications. It must be set before
# PyTorch creates a cuBLAS handle; an existing user choice is preserved.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gymnasium as gym

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SUPPORTED_LOG_STD_MIN = -20.0
SUPPORTED_LOG_STD_MAX = 2.0

DEFAULT_CONFIG = {
    "hidden_dim": 64,
    "n_training_episodes": 10000,
    "n_evaluation_episodes": 100,
    "max_steps": 200,
    "gamma": 0.99,
    "lr": 5e-4,
    "env_id": "Pendulum-v1",
    "seed": 42,
    "experiment_name": "reinforce_pendulum",
    "checkpoint_every": 500,
    "eval_every": 500,
    "print_every": 100,
    "runs_root": "runs",
    "objective": "discounted_start_state",
    "normalize_returns": False,
    "log_std_min": -5.0,
    "log_std_max": 1.0,
    "best_checkpoint_metric": "eval/deterministic_mean_reward",
}


def set_seed(seed):
    if (
        torch.cuda.is_available()
        and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8",":16:8")
    ):
        raise RuntimeError(
            "deterministic CUDA training requires "
            "CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8"
        )
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def create_summary_writer(**kwargs):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "TensorBoard is required for training; install it with "
            "`python -m pip install tensorboard`"
        ) from error
    return SummaryWriter(**kwargs)


def collect_runtime_metadata():
    metadata = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": str(np.__version__),
        "torch": str(torch.__version__),
        "gymnasium": str(gym.__version__),
        "cuda_available": torch.cuda.is_available(),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    repository_root = Path(__file__).resolve().parents[3]
    try:
        commit = subprocess.run(
            ["git","rev-parse","HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git","status","--porcelain"],
                cwd=repository_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError,subprocess.CalledProcessError):
        commit = None
        dirty = None
    metadata["git_commit"] = commit
    metadata["git_dirty"] = dirty
    return metadata


def write_json_atomic(path,data):
    path = Path(path)
    temporary_path = path.with_suffix(path.suffix+".tmp")
    with temporary_path.open("w",encoding="utf-8") as output_file:
        json.dump(data,output_file,ensure_ascii=False,indent=2)
    temporary_path.replace(path)


def validate_config(config):
    positive_integer_fields = (
        "hidden_dim",
        "n_training_episodes",
        "n_evaluation_episodes",
        "max_steps",
        "eval_every",
        "print_every",
    )
    for field in positive_integer_fields:
        if config[field] <= 0:
            raise ValueError(f"{field} must be greater than zero")
    if not 0 <= config["seed"] <= np.iinfo(np.uint32).max:
        raise ValueError("seed must be between 0 and 2**32-1")
    if not math.isfinite(config["gamma"]) or not 0 <= config["gamma"] <= 1:
        raise ValueError("gamma must be finite and between 0 and 1")
    if not math.isfinite(config["lr"]) or config["lr"] <= 0:
        raise ValueError("lr must be finite and greater than zero")
    if (
        not math.isfinite(config["log_std_min"])
        or not math.isfinite(config["log_std_max"])
        or config["log_std_min"] >= config["log_std_max"]
    ):
        raise ValueError("log_std_min must be less than log_std_max")
    if (
        config["log_std_min"] < SUPPORTED_LOG_STD_MIN
        or config["log_std_max"] > SUPPORTED_LOG_STD_MAX
    ):
        raise ValueError(
            "log_std bounds must stay within "
            f"[{SUPPORTED_LOG_STD_MIN}, {SUPPORTED_LOG_STD_MAX}]"
        )
    if config["checkpoint_every"] < 0:
        raise ValueError("checkpoint_every must be zero or greater")
    if not config["experiment_name"].strip():
        raise ValueError("experiment_name must not be empty")
    if not config["env_id"].strip():
        raise ValueError("env_id must not be empty")
    if not config["runs_root"].strip():
        raise ValueError("runs_root must not be empty")


def validate_environment(env):
    observation_space = env.observation_space
    action_space = env.action_space
    if not isinstance(observation_space,gym.spaces.Box):
        raise TypeError("observation space must be gymnasium.spaces.Box")
    if len(observation_space.shape) != 1 or observation_space.shape[0] <= 0:
        raise ValueError("observation space must be a non-empty one-dimensional Box")
    if not isinstance(action_space,gym.spaces.Box):
        raise TypeError("action space must be gymnasium.spaces.Box")
    if len(action_space.shape) != 1 or action_space.shape[0] <= 0:
        raise ValueError("action space must be a non-empty one-dimensional Box")
    if not np.isfinite(action_space.low).all() or not np.isfinite(action_space.high).all():
        raise ValueError("action space bounds must be finite")
    if not np.less(action_space.low,action_space.high).all():
        raise ValueError("every action-space upper bound must exceed its lower bound")


def create_run_directory(experiment_name, seed, runs_root="runs"):
    safe_experiment_name = re.sub(r"[^\w.-]+","-",experiment_name).strip("-._")
    safe_experiment_name = safe_experiment_name or "experiment"
    run_time = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S%z")
    run_id = f"{safe_experiment_name}__{run_time}__seed-{seed}"
    runs_root = Path(runs_root)
    suffix = 1
    run_directory = runs_root/run_id
    while True:
        try:
            run_directory.mkdir(parents=True,exist_ok=False)
            break
        except FileExistsError:
            run_directory = runs_root/f"{run_id}__{suffix:02d}"
            suffix += 1
    return run_directory,run_directory.name,run_time


def save_run_config(run_directory,config,run_id,run_time):
    run_config = {
        **config,
        "run_id": run_id,
        "run_time": run_time,
        "device": str(DEVICE),
    }
    config_path = run_directory/"config.json"
    write_json_atomic(config_path,run_config)
    return run_config


def write_run_summary(
    run_directory,
    status,
    started_at,
    started_monotonic,
    error=None,
):
    run_directory = Path(run_directory)
    metrics_path = run_directory/"metrics.csv"
    eval_metrics_path = run_directory/"eval_metrics.csv"
    metric_rows = []
    evaluation_rows = []
    if metrics_path.is_file():
        with metrics_path.open(encoding="utf-8",newline="") as metrics_file:
            metric_rows = list(csv.DictReader(metrics_file))
    if eval_metrics_path.is_file():
        with eval_metrics_path.open(encoding="utf-8",newline="") as eval_metrics_file:
            evaluation_rows = list(csv.DictReader(eval_metrics_file))

    latest_path = run_directory/"checkpoints"/"latest.pt"
    latest_checkpoint = (
        load_training_checkpoint(latest_path,map_location="cpu")
        if latest_path.is_file()
        else None
    )
    finished_at = datetime.now().astimezone()
    best_eval_reward = (
        latest_checkpoint["best_eval_reward"]
        if latest_checkpoint is not None
        else None
    )
    if best_eval_reward is not None and not math.isfinite(best_eval_reward):
        best_eval_reward = None
    summary = {
        "run_id": run_directory.name,
        "status": status,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat() if status != "running" else None,
        "duration_seconds": round(time.monotonic()-started_monotonic,6),
        "episodes_completed": (
            latest_checkpoint["episode"] if latest_checkpoint is not None else 0
        ),
        "total_environment_steps": sum(
            int(row["episode_length"]) for row in metric_rows
        ),
        "best_eval_reward": best_eval_reward,
        "best_eval_episode": (
            latest_checkpoint["best_eval_episode"]
            if latest_checkpoint is not None
            else None
        ),
        "final_evaluation": (
            {
                "episode": int(evaluation_rows[-1]["episode"]),
                "deterministic_mean_reward": float(
                    evaluation_rows[-1]["deterministic_mean_reward"]
                ),
                "deterministic_std_reward": float(
                    evaluation_rows[-1]["deterministic_std_reward"]
                ),
                "stochastic_mean_reward": float(
                    evaluation_rows[-1]["stochastic_mean_reward"]
                ),
                "stochastic_std_reward": float(
                    evaluation_rows[-1]["stochastic_std_reward"]
                ),
                "n_eval_episodes": int(evaluation_rows[-1]["n_eval_episodes"]),
            }
            if evaluation_rows
            else None
        ),
        "error": (
            {"type": type(error).__name__,"message": str(error)}
            if error is not None
            else None
        ),
    }
    write_json_atomic(run_directory/"summary.json",summary)


def save_training_checkpoint(
    checkpoint_path,
    policy,
    optimizer,
    episode,
    config,
    scores,
    evaluation_history,
    best_eval_reward,
    best_eval_episode,
    run_id,
):
    numpy_random_state = np.random.get_state()
    checkpoint = {
        "checkpoint_version": 2,
        "run_id": run_id,
        "episode": episode,
        "policy_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "scores": [float(score) for score in scores],
        "evaluation_history": [
            {
                "episode": int(evaluation["episode"]),
                "deterministic_mean_reward": float(
                    evaluation["deterministic_mean_reward"]
                ),
                "deterministic_std_reward": float(
                    evaluation["deterministic_std_reward"]
                ),
                "stochastic_mean_reward": float(
                    evaluation["stochastic_mean_reward"]
                ),
                "stochastic_std_reward": float(
                    evaluation["stochastic_std_reward"]
                ),
                "n_eval_episodes": int(evaluation["n_eval_episodes"]),
            }
            for evaluation in evaluation_history
        ],
        "best_eval_reward": float(best_eval_reward),
        "best_eval_episode": (
            int(best_eval_episode) if best_eval_episode is not None else None
        ),
        "random_state": random.getstate(),
        "numpy_random_state": {
            "bit_generator": numpy_random_state[0],
            "state": torch.from_numpy(
                numpy_random_state[1].astype(np.int64,copy=True)
            ),
            "position": int(numpy_random_state[2]),
            "has_gauss": int(numpy_random_state[3]),
            "cached_gaussian": float(numpy_random_state[4]),
        },
        "torch_random_state": torch.get_rng_state(),
        "cuda_random_state": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True,exist_ok=True)
    temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix+".tmp")
    torch.save(checkpoint,temporary_path)
    temporary_path.replace(checkpoint_path)


def copy_checkpoint_atomic(source_path,destination_path):
    source_path = Path(source_path)
    destination_path = Path(destination_path)
    temporary_path = destination_path.with_suffix(destination_path.suffix+".tmp")
    shutil.copyfile(source_path,temporary_path)
    temporary_path.replace(destination_path)


def load_training_checkpoint(checkpoint_path,map_location=DEVICE):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=True,
    )
    if not isinstance(checkpoint,dict):
        raise ValueError("checkpoint root must be a dictionary")
    if checkpoint.get("checkpoint_version") != 2:
        raise ValueError("unsupported checkpoint version")
    required_keys = {
        "run_id",
        "episode",
        "policy_state_dict",
        "optimizer_state_dict",
        "config",
        "scores",
        "evaluation_history",
        "best_eval_reward",
        "best_eval_episode",
        "random_state",
        "numpy_random_state",
        "torch_random_state",
        "cuda_random_state",
    }
    missing_keys = required_keys-checkpoint.keys()
    if missing_keys:
        raise ValueError(f"checkpoint is missing keys: {sorted(missing_keys)}")
    episode = checkpoint["episode"]
    scores = checkpoint["scores"]
    evaluation_history = checkpoint["evaluation_history"]
    if not isinstance(checkpoint["run_id"],str) or not checkpoint["run_id"]:
        raise ValueError("checkpoint run_id must be a non-empty string")
    if type(episode) is not int or episode < 0:
        raise ValueError("checkpoint episode must be a non-negative integer")
    if not isinstance(checkpoint["config"],dict):
        raise ValueError("checkpoint config must be a dictionary")
    if checkpoint["config"].get("run_id") != checkpoint["run_id"]:
        raise ValueError("checkpoint run_id does not match its config")
    if not isinstance(checkpoint["policy_state_dict"],dict):
        raise ValueError("checkpoint policy state must be a dictionary")
    if not isinstance(checkpoint["optimizer_state_dict"],dict):
        raise ValueError("checkpoint optimizer state must be a dictionary")
    if not isinstance(scores,list) or len(scores) != episode:
        raise ValueError("checkpoint scores must contain one value per episode")
    if not isinstance(evaluation_history,list):
        raise ValueError("checkpoint evaluation history must be a list")
    evaluation_episodes = []
    for evaluation in evaluation_history:
        if not isinstance(evaluation,dict):
            raise ValueError("checkpoint evaluation entries must be dictionaries")
        try:
            evaluation_episode = evaluation["episode"]
            deterministic_mean = float(evaluation["deterministic_mean_reward"])
            deterministic_std = float(evaluation["deterministic_std_reward"])
            stochastic_mean = float(evaluation["stochastic_mean_reward"])
            stochastic_std = float(evaluation["stochastic_std_reward"])
            n_eval_episodes = evaluation["n_eval_episodes"]
        except (KeyError,TypeError,ValueError) as error:
            raise ValueError("checkpoint contains an invalid evaluation entry") from error
        if (
            type(evaluation_episode) is not int
            or not 1 <= evaluation_episode <= episode
            or type(n_eval_episodes) is not int
            or n_eval_episodes <= 0
            or not all(
                math.isfinite(value)
                for value in (deterministic_mean,stochastic_mean)
            )
            or not all(
                math.isfinite(value) and value >= 0
                for value in (deterministic_std,stochastic_std)
            )
        ):
            raise ValueError("checkpoint contains an invalid evaluation entry")
        evaluation_episodes.append(evaluation_episode)
    if evaluation_episodes != sorted(set(evaluation_episodes)):
        raise ValueError("checkpoint evaluation episodes must be strictly increasing")
    best_eval_episode = checkpoint["best_eval_episode"]
    best_eval_reward = checkpoint["best_eval_reward"]
    if evaluation_history:
        best_evaluation = max(
            evaluation_history,
            key=lambda evaluation: evaluation["deterministic_mean_reward"],
        )
        if (
            best_eval_episode != best_evaluation["episode"]
            or best_eval_reward != best_evaluation["deterministic_mean_reward"]
        ):
            raise ValueError("checkpoint best evaluation metadata is inconsistent")
    elif best_eval_episode is not None or best_eval_reward != -np.inf:
        raise ValueError("checkpoint best evaluation metadata must be empty")
    return checkpoint


def restore_training_state(checkpoint,policy,optimizer):
    policy.load_state_dict(checkpoint["policy_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    random.setstate(checkpoint["random_state"])
    numpy_random_state = checkpoint["numpy_random_state"]
    np.random.set_state(
        (
            numpy_random_state["bit_generator"],
            numpy_random_state["state"].cpu().numpy().astype(
                np.uint32,
                copy=True,
            ),
            numpy_random_state["position"],
            numpy_random_state["has_gauss"],
            numpy_random_state["cached_gaussian"],
        )
    )
    torch.set_rng_state(checkpoint["torch_random_state"].cpu())
    if torch.cuda.is_available():
        cuda_random_state = checkpoint["cuda_random_state"]
        if (
            cuda_random_state is None
            or len(cuda_random_state) != torch.cuda.device_count()
        ):
            raise ValueError(
                "checkpoint CUDA RNG state does not match the visible GPU count"
            )
        torch.cuda.set_rng_state_all(
            [random_state.cpu() for random_state in cuda_random_state]
        )


def prepare_csv_alignment(csv_path,expected_episodes,checkpoint_episode):
    with csv_path.open(encoding="utf-8",newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if (
        fieldnames is None
        or "episode" not in fieldnames
        or len(fieldnames) != len(set(fieldnames))
    ):
        raise ValueError(f"{csv_path.name} has an invalid header")
    expected_count = len(expected_episodes)
    if len(rows) < expected_count:
        raise ValueError(f"{csv_path.name} ends before the checkpoint")
    for expected_episode,row in zip(expected_episodes,rows[:expected_count]):
        if None in row or any(row.get(field) is None for field in fieldnames):
            raise ValueError(f"{csv_path.name} has a damaged committed row")
        try:
            episode = int(row["episode"])
        except (TypeError,ValueError) as error:
            raise ValueError(f"{csv_path.name} has an invalid episode column") from error
        if episode != expected_episode:
            raise ValueError(f"{csv_path.name} is not aligned with the checkpoint")
    for row in rows[expected_count:]:
        try:
            trailing_episode = int(row.get("episode"))
        except (TypeError,ValueError):
            continue
        if trailing_episode <= checkpoint_episode:
            raise ValueError(f"{csv_path.name} contains duplicate or reordered episodes")
    return fieldnames,rows[:expected_count],len(rows) > expected_count


def write_csv_atomic(csv_path,fieldnames,rows):
    temporary_path = csv_path.with_suffix(csv_path.suffix+".tmp")
    with temporary_path.open("w",encoding="utf-8",newline="") as csv_file:
        writer = csv.DictWriter(csv_file,fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(csv_path)


def validate_resume_artifacts(run_directory,checkpoint):
    checkpoint_episode = checkpoint["episode"]
    metrics_path = Path(run_directory)/"metrics.csv"
    eval_metrics_path = Path(run_directory)/"eval_metrics.csv"
    if not metrics_path.is_file() or not eval_metrics_path.is_file():
        raise ValueError("resume requires the original metrics CSV files")

    expected_metric_episodes = list(range(1,checkpoint_episode+1))
    checkpoint_eval_episodes = [
        int(evaluation["episode"])
        for evaluation in checkpoint["evaluation_history"]
    ]
    metrics_alignment = prepare_csv_alignment(
        metrics_path,
        expected_metric_episodes,
        checkpoint_episode,
    )
    eval_metrics_alignment = prepare_csv_alignment(
        eval_metrics_path,
        checkpoint_eval_episodes,
        checkpoint_episode,
    )

    if checkpoint["best_eval_episode"] is not None:
        best_path = Path(run_directory)/"checkpoints"/"best.pt"
        best_checkpoint = (
            load_training_checkpoint(best_path,map_location="cpu")
            if best_path.is_file()
            else None
        )
        best_is_aligned = (
            best_checkpoint is not None
            and best_checkpoint["episode"] == checkpoint["best_eval_episode"]
            and best_checkpoint["best_eval_reward"] == checkpoint["best_eval_reward"]
        )
        if not best_is_aligned and checkpoint["best_eval_episode"] == checkpoint_episode:
            temporary_path = best_path.with_suffix(best_path.suffix+".tmp")
            torch.save(checkpoint,temporary_path)
            temporary_path.replace(best_path)
        elif not best_is_aligned:
            raise ValueError("best.pt is not aligned with the resumed checkpoint")

    for csv_path,alignment in (
        (metrics_path,metrics_alignment),
        (eval_metrics_path,eval_metrics_alignment),
    ):
        fieldnames,rows,needs_repair = alignment
        if needs_repair:
            write_csv_atomic(csv_path,fieldnames,rows)


def validate_resume_checkpoint_choice(resume_path,checkpoint):
    if resume_path.name not in ("latest.pt","interrupted.pt"):
        raise ValueError(
            "resume only supports checkpoints/latest.pt or "
            "checkpoints/interrupted.pt; use a new run for older snapshots"
        )
    latest_path = resume_path.parent/"latest.pt"
    if resume_path.name == "latest.pt":
        return
    if not latest_path.is_file():
        raise ValueError("interrupted.pt requires an aligned latest.pt")
    latest_checkpoint = load_training_checkpoint(latest_path,map_location="cpu")
    if (
        checkpoint["run_id"] != latest_checkpoint["run_id"]
        or checkpoint["episode"] != latest_checkpoint["episode"]
        or not torch.equal(
            checkpoint["torch_random_state"].cpu(),
            latest_checkpoint["torch_random_state"].cpu(),
        )
    ):
        raise ValueError("interrupted.pt is stale; resume from latest.pt instead")


def compute_reward_to_go(rewards,gamma,dtype=torch.float32,device=None):
    """Compute G_t = r_t + gamma*r_(t+1) + ... for every timestep."""
    rewards = torch.as_tensor(rewards,dtype=dtype,device=device)
    if rewards.ndim != 1 or rewards.numel() == 0:
        raise ValueError("rewards must be a non-empty one-dimensional sequence")
    returns = torch.empty_like(rewards)
    next_return = torch.zeros((),dtype=dtype,device=device)
    for timestep in reversed(range(len(rewards))):
        next_return = rewards[timestep]+gamma*next_return
        returns[timestep] = next_return
    return returns


def standardize_returns(returns):
    """Standardize returns across one trajectory for an optional experiment."""
    if returns.ndim != 1 or returns.numel() == 0:
        raise ValueError("returns must be a non-empty one-dimensional tensor")
    return (returns-returns.mean())/(returns.std(unbiased=False)+torch.finfo(returns.dtype).eps)


def compute_reinforce_loss(saved_log_probs,returns,gamma):
    """REINFORCE loss for J=E[sum_t gamma**t * reward_t]."""
    if not saved_log_probs:
        raise ValueError("saved_log_probs must not be empty")
    if returns.ndim != 1 or len(saved_log_probs) != returns.numel():
        raise ValueError("saved_log_probs and returns must have equal lengths")
    log_probs = torch.stack(saved_log_probs).reshape(len(saved_log_probs),-1).sum(dim=-1)
    returns = returns.to(dtype=log_probs.dtype,device=log_probs.device)
    timestep_discounts = torch.pow(
        torch.as_tensor(gamma,dtype=log_probs.dtype,device=log_probs.device),
        torch.arange(len(returns),dtype=log_probs.dtype,device=log_probs.device),
    )
    return -(timestep_discounts*log_probs*returns).sum()


def compute_gradient_norm(parameters):
    squared_norm = None
    for parameter in parameters:
        if parameter.grad is None:
            continue
        parameter_squared_norm = parameter.grad.detach().pow(2).sum()
        squared_norm = (
            parameter_squared_norm
            if squared_norm is None
            else squared_norm+parameter_squared_norm
        )
    return squared_norm.sqrt().item() if squared_norm is not None else 0.0


@dataclass
class TrainingState:
    start_episode: int = 1
    scores: list = field(default_factory=list)
    evaluation_history: list = field(default_factory=list)
    best_eval_reward: float = -np.inf
    best_eval_episode: int | None = None
    append_metrics: bool = False


class Policy(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim,
        action_low,
        action_high,
        log_std_min=-5.0,
        log_std_max=1.0,
    ):
        super().__init__()
        self.hidden_layer = nn.Linear(state_dim,hidden_dim)
        self.mean_head = nn.Linear(hidden_dim,action_dim)
        self.log_std_head = nn.Linear(hidden_dim,action_dim)
        nn.init.zeros_(self.log_std_head.weight)
        nn.init.constant_(self.log_std_head.bias,-0.5)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.register_buffer(
            "action_scale",
            torch.as_tensor((action_high-action_low)/2.0,dtype=torch.float32),
        )
        self.register_buffer(
            "action_bias",
            torch.as_tensor((action_high+action_low)/2.0,dtype=torch.float32),
        )

    def forward(self,x):
        x = F.relu(self.hidden_layer(x))
        mu = self.mean_head(x)
        log_std = self.log_std_head(x)
        return mu,log_std

    def act(self,state,deterministic=False,return_diagnostics=False):
        state = torch.as_tensor(
            state,
            dtype=self.action_scale.dtype,
            device=self.action_scale.device,
        ).unsqueeze(0)
        mu,log_std = self.forward(state)
        log_std = torch.clamp(log_std,self.log_std_min,self.log_std_max)
        std = log_std.exp()
        dist = torch.distributions.Normal(mu,std)
        raw_action = mu if deterministic else dist.sample()
        squashed_action = torch.tanh(raw_action)
        action = self.action_bias+self.action_scale*squashed_action
        log_prob = None
        if not deterministic:
            log_tanh_jacobian = 2*(
                math.log(2)-raw_action-F.softplus(-2*raw_action)
            )
            log_prob = (
                dist.log_prob(raw_action)
                - torch.log(self.action_scale)
                - log_tanh_jacobian
            ).sum(dim=-1)
        result = action.squeeze(0).detach().cpu().numpy(),log_prob
        if not return_diagnostics:
            return result
        diagnostics = {
            "policy_std": std.detach().mean().item(),
            "action_saturation": (squashed_action.detach().abs()>0.99).float().mean().item(),
        }
        return result[0],result[1],diagnostics

def reinforce(
    env,
    eval_env,
    policy,
    optimizer,
    config,
    run_directory,
    writer,
    training_state=None,
):
    training_state = training_state or TrainingState()
    n_training_episodes = config["n_training_episodes"]
    max_steps = config["max_steps"]
    gamma = config["gamma"]
    print_every = config["print_every"]
    seed = config["seed"]
    normalize_returns = config["normalize_returns"]
    checkpoint_every = config["checkpoint_every"]
    eval_every = config["eval_every"]
    n_evaluation_episodes = config["n_evaluation_episodes"]
    run_id = config["run_id"]
    run_directory = Path(run_directory)
    metrics_path = run_directory/"metrics.csv"
    eval_metrics_path = run_directory/"eval_metrics.csv"
    checkpoint_directory = run_directory/"checkpoints"
    start_episode = training_state.start_episode
    append_metrics = training_state.append_metrics
    scores = list(training_state.scores)
    recent_scores = deque(scores[-100:],maxlen=100)
    evaluation_history = list(training_state.evaluation_history)
    best_eval_reward = training_state.best_eval_reward
    best_eval_episode = training_state.best_eval_episode
    metrics_file = None
    eval_metrics_file = None
    metric_names = [
        "episode",
        "episode_reward",
        "reward_rolling_mean_100",
        "episode_length",
        "policy_loss",
        "discounted_return_mean",
        "discounted_return_std",
        "policy_return_mean",
        "policy_return_std",
        "gradient_norm",
        "policy_std_mean",
        "action_saturation_fraction",
        "learning_rate",
    ]
    try:
        write_metrics_header = not append_metrics or not metrics_path.exists()
        metrics_file = metrics_path.open(
            "a" if append_metrics else "w",
            encoding="utf-8",
            newline="",
        )
        metrics_writer = csv.DictWriter(metrics_file,fieldnames=metric_names)
        if write_metrics_header:
            metrics_writer.writeheader()

        write_eval_metrics_header = (
            not append_metrics or not eval_metrics_path.exists()
        )
        eval_metrics_file = eval_metrics_path.open(
            "a" if append_metrics else "w",
            encoding="utf-8",
            newline="",
        )
        eval_metrics_writer = csv.DictWriter(
            eval_metrics_file,
            fieldnames=[
                "episode",
                "deterministic_mean_reward",
                "deterministic_std_reward",
                "stochastic_mean_reward",
                "stochastic_std_reward",
                "n_eval_episodes",
            ],
        )
        if write_eval_metrics_header:
            eval_metrics_writer.writeheader()

        save_training_checkpoint(
            checkpoint_directory/"latest.pt",
            policy,
            optimizer,
            start_episode-1,
            config,
            scores,
            evaluation_history,
            best_eval_reward,
            best_eval_episode,
            run_id,
        )

        for i_episode in range(start_episode,n_training_episodes+1):
            is_new_best = False
            saved_log_probs = []
            rewards = []
            policy_stds = []
            action_saturations = []
            state,_ = env.reset(seed=seed+i_episode-1)
            for _ in range(max_steps):
                action,log_prob,action_diagnostics = policy.act(
                    state,
                    return_diagnostics=True,
                )
                saved_log_probs.append(log_prob)
                policy_stds.append(action_diagnostics["policy_std"])
                action_saturations.append(action_diagnostics["action_saturation"])
                state,reward,terminated,truncated,_ = env.step(action)
                rewards.append(reward)
                if terminated or truncated:
                    break
            episode_reward = sum(rewards)
            recent_scores.append(episode_reward)
            scores.append(episode_reward)
            n_steps = len(rewards)
            policy_parameter = next(policy.parameters())
            returns = compute_reward_to_go(
                rewards,
                gamma,
                dtype=policy_parameter.dtype,
                device=policy_parameter.device,
            )
            policy_returns = (
                standardize_returns(returns) if normalize_returns else returns
            )

            policy_loss = compute_reinforce_loss(saved_log_probs,policy_returns,gamma)

            optimizer.zero_grad()
            policy_loss.backward()
            gradient_norm = compute_gradient_norm(policy.parameters())
            optimizer.step()

            metrics = {
                "episode": i_episode,
                "episode_reward": episode_reward,
                "reward_rolling_mean_100": np.mean(recent_scores),
                "episode_length": n_steps,
                "policy_loss": policy_loss.item(),
                "discounted_return_mean": returns.mean().item(),
                "discounted_return_std": returns.std(unbiased=False).item(),
                "policy_return_mean": policy_returns.mean().item(),
                "policy_return_std": policy_returns.std(unbiased=False).item(),
                "gradient_norm": gradient_norm,
                "policy_std_mean": np.mean(policy_stds),
                "action_saturation_fraction": np.mean(action_saturations),
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            if writer is not None:
                for metric_name,metric_value in metrics.items():
                    if metric_name != "episode":
                        writer.add_scalar(f"train/{metric_name}",metric_value,i_episode)
            metrics_writer.writerow(metrics)
            metrics_file.flush()

            should_evaluate = (
                i_episode % eval_every == 0 or i_episode == n_training_episodes
            )
            if should_evaluate:
                was_training = policy.training
                policy.eval()
                deterministic_mean,deterministic_std = evaluate_agent(
                    eval_env,
                    max_steps,
                    n_evaluation_episodes,
                    policy,
                    seed,
                    deterministic=True,
                )
                stochastic_mean,stochastic_std = evaluate_agent(
                    eval_env,
                    max_steps,
                    n_evaluation_episodes,
                    policy,
                    seed,
                    deterministic=False,
                )
                if was_training:
                    policy.train()
                eval_metrics = {
                    "episode": i_episode,
                    "deterministic_mean_reward": deterministic_mean,
                    "deterministic_std_reward": deterministic_std,
                    "stochastic_mean_reward": stochastic_mean,
                    "stochastic_std_reward": stochastic_std,
                    "n_eval_episodes": n_evaluation_episodes,
                }
                evaluation_history.append(eval_metrics)
                if writer is not None:
                    for metric_name,metric_value in eval_metrics.items():
                        if metric_name not in ("episode","n_eval_episodes"):
                            writer.add_scalar(
                                f"eval/{metric_name}",
                                metric_value,
                                i_episode,
                            )
                eval_metrics_writer.writerow(eval_metrics)
                eval_metrics_file.flush()
                if deterministic_mean > best_eval_reward:
                    best_eval_reward = deterministic_mean
                    best_eval_episode = i_episode
                    is_new_best = True

            if writer is not None:
                writer.flush()
            save_training_checkpoint(
                checkpoint_directory/"latest.pt",
                policy,
                optimizer,
                i_episode,
                config,
                scores,
                evaluation_history,
                best_eval_reward,
                best_eval_episode,
                run_id,
            )
            if checkpoint_every and i_episode % checkpoint_every == 0:
                copy_checkpoint_atomic(
                    checkpoint_directory/"latest.pt",
                    checkpoint_directory/f"episode_{i_episode:06d}.pt",
                )
            if is_new_best:
                copy_checkpoint_atomic(
                    checkpoint_directory/"latest.pt",
                    checkpoint_directory/"best.pt",
                )

            if i_episode % print_every == 0:
                print(
                    f"Episode {i_episode}\t"
                    f"Rolling Average Score: {np.mean(recent_scores):.2f}"
                )
        copy_checkpoint_atomic(
            checkpoint_directory/"latest.pt",
            checkpoint_directory/"final.pt",
        )
    except BaseException:
        if metrics_file is not None:
            metrics_file.close()
            metrics_file = None
        if eval_metrics_file is not None:
            eval_metrics_file.close()
            eval_metrics_file = None
        latest_path = checkpoint_directory/"latest.pt"
        if latest_path.is_file():
            latest_checkpoint = load_training_checkpoint(
                latest_path,
                map_location="cpu",
            )
            try:
                validate_resume_artifacts(run_directory,latest_checkpoint)
            except Exception as recovery_error:
                print(
                    f"Failed to reconcile interrupted run artifacts: {recovery_error}",
                    file=sys.stderr,
                )
            copy_checkpoint_atomic(
                latest_path,
                checkpoint_directory/"interrupted.pt",
            )
        raise
    finally:
        if metrics_file is not None:
            metrics_file.close()
        if eval_metrics_file is not None:
            eval_metrics_file.close()

    return scores,evaluation_history

@torch.no_grad()
def evaluate_agent(
    env,
    max_steps,
    n_eval_episodes,
    policy,
    seed,
    deterministic=True,
):
    """Evaluate fixed initial states without changing the training RNG stream."""
    policy_device = next(policy.parameters()).device
    cuda_devices = (
        list(range(torch.cuda.device_count()))
        if policy_device.type == "cuda"
        else []
    )
    episode_rewards = []
    with torch.random.fork_rng(devices=cuda_devices):
        if not deterministic:
            torch.manual_seed(seed+200_000)
            if policy_device.type == "cuda":
                torch.cuda.manual_seed_all(seed+200_000)

        for episode in range(n_eval_episodes):
            state,_ = env.reset(seed=seed+100_000+episode)
            episode_reward = 0.0
            for _ in range(max_steps):
                action,_ = policy.act(state,deterministic=deterministic)
                state,reward,terminated,truncated,_ = env.step(action)
                episode_reward += reward
                if terminated or truncated:
                    break
            episode_rewards.append(episode_reward)

    return float(np.mean(episode_rewards)),float(np.std(episode_rewards))


def parse_args(argv=None):
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="Train REINFORCE on Pendulum-v1")
    parser.add_argument("--experiment-name",default=DEFAULT_CONFIG["experiment_name"])
    parser.add_argument("--seed",type=int,default=DEFAULT_CONFIG["seed"])
    parser.add_argument("--env-id",default=DEFAULT_CONFIG["env_id"])
    parser.add_argument(
        "--hidden-dim",
        "--h-size",
        dest="hidden_dim",
        type=int,
        default=DEFAULT_CONFIG["hidden_dim"],
    )
    parser.add_argument(
        "--n-training-episodes",
        type=int,
        default=None,
        help="total training episodes; the only configurable setting on resume",
    )
    parser.add_argument(
        "--n-evaluation-episodes",
        type=int,
        default=DEFAULT_CONFIG["n_evaluation_episodes"],
    )
    parser.add_argument(
        "--max-steps",
        "--max-t",
        dest="max_steps",
        type=int,
        default=DEFAULT_CONFIG["max_steps"],
    )
    parser.add_argument("--gamma",type=float,default=DEFAULT_CONFIG["gamma"])
    parser.add_argument("--lr",type=float,default=DEFAULT_CONFIG["lr"])
    parser.add_argument(
        "--log-std-min",
        type=float,
        default=DEFAULT_CONFIG["log_std_min"],
        help=f"minimum log standard deviation (>= {SUPPORTED_LOG_STD_MIN:g})",
    )
    parser.add_argument(
        "--log-std-max",
        type=float,
        default=DEFAULT_CONFIG["log_std_max"],
        help=f"maximum log standard deviation (<= {SUPPORTED_LOG_STD_MAX:g})",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=DEFAULT_CONFIG["checkpoint_every"],
        help="periodic checkpoint interval; use 0 to disable periodic checkpoints",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=DEFAULT_CONFIG["eval_every"],
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=DEFAULT_CONFIG["print_every"],
    )
    parser.add_argument("--runs-root",default=DEFAULT_CONFIG["runs_root"])
    parser.add_argument(
        "--normalize-returns",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_CONFIG["normalize_returns"],
        help="standardize reward-to-go values within each trajectory",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help=(
            "resume trusted latest.pt/interrupted.pt in its original run directory; "
            "only --n-training-episodes may also be supplied"
        ),
    )
    args = parser.parse_args(raw_argv)
    if args.resume is None:
        if args.n_training_episodes is None:
            args.n_training_episodes = DEFAULT_CONFIG["n_training_episodes"]
        try:
            validate_config(vars(args))
        except ValueError as error:
            parser.error(str(error))
    else:
        supplied_options = {
            argument.split("=",1)[0]
            for argument in raw_argv
            if argument.startswith("--")
        }
        incompatible_options = supplied_options-{
            "--resume",
            "--n-training-episodes",
        }
        if incompatible_options:
            parser.error(
                "--resume uses all experiment settings from the checkpoint; "
                "remove these incompatible options: "
                + ", ".join(sorted(incompatible_options))
            )
        if args.n_training_episodes is not None and args.n_training_episodes <= 0:
            parser.error("n_training_episodes must be greater than zero")
    return args


def main(argv=None):
    started_at = datetime.now().astimezone()
    started_monotonic = time.monotonic()
    args = parse_args(argv)
    resume_checkpoint = None
    if args.resume is None:
        config = vars(args).copy()
        config.pop("resume")
        config["objective"] = DEFAULT_CONFIG["objective"]
        config["best_checkpoint_metric"] = DEFAULT_CONFIG["best_checkpoint_metric"]
        start_episode = 1
        initial_scores = None
        initial_evaluation_history = None
        initial_best_eval_reward = -np.inf
        initial_best_eval_episode = None
        append_metrics = False
        run_dir = None
        run_id = None
        run_time = None
        config["runtime"] = collect_runtime_metadata()
    else:
        resume_path = Path(args.resume).expanduser().resolve()
        resume_checkpoint = load_training_checkpoint(resume_path,map_location=DEVICE)
        config = dict(resume_checkpoint["config"])
        saved_target_episode = config["n_training_episodes"]
        if args.n_training_episodes is not None:
            config["n_training_episodes"] = args.n_training_episodes
        validate_config(config)
        run_dir = resume_path.parent.parent
        run_id = resume_checkpoint["run_id"]
        if run_dir.name != run_id:
            raise ValueError("checkpoint must remain inside its original run directory")
        if config.get("device") != str(DEVICE):
            raise ValueError(
                f"resume device mismatch: checkpoint uses {config.get('device')!r}, "
                f"current device is {str(DEVICE)!r}"
            )
        validate_resume_checkpoint_choice(resume_path,resume_checkpoint)
        validate_resume_artifacts(run_dir,resume_checkpoint)
        checkpoint_episode = resume_checkpoint["episode"]
        target_episode = config["n_training_episodes"]
        final_checkpoint_exists = (run_dir/"checkpoints"/"final.pt").is_file()
        if target_episode < checkpoint_episode or (
            target_episode == checkpoint_episode and final_checkpoint_exists
        ):
            raise ValueError(
                "n_training_episodes must be greater than the checkpoint episode "
                "unless final.pt is missing and only finalization remains"
            )
        finalize_only = target_episode == checkpoint_episode
        run_time = config["run_time"]
        start_episode = checkpoint_episode+1
        initial_scores = resume_checkpoint["scores"]
        initial_evaluation_history = resume_checkpoint["evaluation_history"]
        initial_best_eval_reward = resume_checkpoint["best_eval_reward"]
        initial_best_eval_episode = resume_checkpoint["best_eval_episode"]
        append_metrics = True
        retained_final_only_evaluation = False
        if finalize_only:
            resume_mode = "finalize_only"
        elif checkpoint_episode >= saved_target_episode:
            resume_mode = "extension"
            retained_final_only_evaluation = (
                checkpoint_episode % config["eval_every"] != 0
            )
            if retained_final_only_evaluation:
                print(
                    "Extending a completed run: its previous final-only evaluation "
                    "is retained, so the evaluation history and best checkpoint "
                    "can differ from a run originally targeting the new count.",
                    file=sys.stderr,
                )
        elif config["n_training_episodes"] != saved_target_episode:
            resume_mode = "recovery_with_target_change"
        else:
            resume_mode = "recovery"
        config.setdefault("resume_history",[]).append(
            {
                "mode": resume_mode,
                "resumed_at": started_at.isoformat(),
                "checkpoint": str(resume_path),
                "from_episode": resume_checkpoint["episode"],
                "previous_target_episode": saved_target_episode,
                "target_episode": config["n_training_episodes"],
                "retained_final_only_evaluation": (
                    retained_final_only_evaluation
                ),
                "runtime": collect_runtime_metadata(),
            }
        )

    set_seed(config["seed"])
    print(DEVICE)

    env = None
    eval_env = None

    try:
        env = gym.make(config["env_id"])
        eval_env = gym.make(config["env_id"])
        validate_environment(env)
        validate_environment(eval_env)
        config["state_dim"] = env.observation_space.shape[0]
        config["action_dim"] = env.action_space.shape[0]

        policy = Policy(
            config["state_dim"],
            config["action_dim"],
            config["hidden_dim"],
            env.action_space.low,
            env.action_space.high,
            config["log_std_min"],
            config["log_std_max"],
        ).to(DEVICE)
        optimizer = optim.Adam(policy.parameters(),lr=config["lr"])

        if resume_checkpoint is None:
            run_dir,run_id,run_time = create_run_directory(
                config["experiment_name"],
                config["seed"],
                config["runs_root"],
            )
        else:
            restore_training_state(resume_checkpoint,policy,optimizer)
        run_config = save_run_config(run_dir,config,run_id,run_time)
        write_run_summary(
            run_dir,
            "running",
            started_at,
            started_monotonic,
        )
        writer_kwargs = {"log_dir": str(run_dir)}
        if append_metrics:
            writer_kwargs["purge_step"] = start_episode
        tensorboard_writer = None
        run_status = "running"
        run_error = None

        try:
            tensorboard_writer = create_summary_writer(**writer_kwargs)
            if append_metrics:
                tensorboard_writer.add_text(
                    "run/resume",
                    json.dumps(config["resume_history"][-1],ensure_ascii=False,indent=2),
                    start_episode-1,
                )
            else:
                tensorboard_writer.add_text(
                    "run/config",
                    f"```json\n{json.dumps(run_config,ensure_ascii=False,indent=2)}\n```",
                    0,
                )
            training_state = TrainingState(
                start_episode=start_episode,
                scores=list(initial_scores or []),
                evaluation_history=list(initial_evaluation_history or []),
                best_eval_reward=initial_best_eval_reward,
                best_eval_episode=initial_best_eval_episode,
                append_metrics=append_metrics,
            )
            scores,evaluation_history = reinforce(
                env,
                eval_env,
                policy,
                optimizer,
                run_config,
                run_dir,
                tensorboard_writer,
                training_state,
            )
            final_evaluation = evaluation_history[-1]
            mean_reward = final_evaluation["deterministic_mean_reward"]
            std_reward = final_evaluation["deterministic_std_reward"]
        except KeyboardInterrupt as error:
            run_status = "interrupted"
            run_error = error
            raise
        except BaseException as error:
            run_status = "failed"
            run_error = error
            raise
        else:
            run_status = "completed"
        finally:
            if tensorboard_writer is not None:
                tensorboard_writer.close()
            try:
                write_run_summary(
                    run_dir,
                    run_status,
                    started_at,
                    started_monotonic,
                    error=run_error,
                )
            except Exception as summary_error:
                print(
                    f"Failed to write run summary: {summary_error}",
                    file=sys.stderr,
                )
    finally:
        if env is not None:
            env.close()
        if eval_env is not None:
            eval_env.close()

    print(f"Run directory: {run_dir}")
    print(f"Deterministic evaluation reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    return run_dir,scores,mean_reward,std_reward


if __name__ == "__main__":
    main()
