"""Faithful, instrumented comparison of the REINFORCE teaching scripts.

Only reproducibility and observation are added. Actions are sent to the
environment exactly as sampled, as in the source scripts. The rtg_baseline
variant changes only actor weights to reward-to-go minus the TD baseline;
the two original variants retain their full-return weights unchanged.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
import types

import gymnasium as gym
import numpy as np
import torch


SOURCE_DIR = Path(__file__).resolve().parents[2]
SOURCE_FILES = {
    "vanilla": SOURCE_DIR / "my_minimal_continuous.py",
    "baseline": SOURCE_DIR / "learned_baseline_version.py",
    "rtg_baseline": SOURCE_DIR / "rtg_learned_baseline_version.py",
}
DEFAULT_CONFIG = {
    "env_name": "InvertedPendulum-v5", "hidden_size": [32, 32],
    "lr": 0.001, "obs_dim": 4, "act_dim": 1,
}


def load_source(variant):
    path = SOURCE_FILES[variant]
    module = types.ModuleType(f"source_{variant}")
    module.__file__ = str(path)
    exec(compile(path.read_text(), str(path), "exec"), module.__dict__)
    return module


@dataclass
class Agent:
    variant: str
    seed: int
    config: dict
    policy: torch.nn.Module
    log_std: torch.nn.Parameter
    critic: torch.nn.Module | None
    optimizer: torch.optim.Optimizer
    critic_optimizer: torch.optim.Optimizer | None
    source: types.ModuleType


def make_agent(variant, seed, config=None):
    config = {**DEFAULT_CONFIG, **(config or {})}
    source = load_source(variant)
    sizes = [config["obs_dim"]] + list(config["hidden_size"])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        policy = source.network(sizes + [config["act_dim"]])
        log_std = torch.nn.Parameter(torch.zeros(config["act_dim"]))
        critic = None
        if variant in ("baseline", "rtg_baseline"):
            torch.manual_seed(10_000 + seed)
            critic = source.network(sizes + [1])
    optimizer = source.Adam(list(policy.parameters()) + [log_std], config["lr"])
    critic_optimizer = source.Adam(critic.parameters(), config["lr"]) if critic is not None else None
    return Agent(variant, seed, config, policy, log_std, critic, optimizer, critic_optimizer, source)


def tensor_batch(trajectory):
    return {key: torch.as_tensor(np.asarray(trajectory[key]), dtype=torch.float32)
            for key in ("obs", "actions", "rewards", "next_obs", "terminated")}


def td_targets(critic, batch):
    return (batch["rewards"] + (1 - batch["terminated"]) *
            critic(batch["next_obs"]).squeeze(-1)).detach()


def reward_to_go(rewards):
    """Undiscounted finite-trajectory returns, preserving shape and dtype."""
    if rewards.ndim != 1:
        raise ValueError("Rewards must have shape [T]")
    return rewards.flip(0).cumsum(0).flip(0)


def _finite(tensors, what):
    if not all(t is not None and torch.isfinite(t).all().item() for t in tensors):
        raise FloatingPointError(f"Non-finite or missing {what}")


def update(agent, batch):
    # Python sum matches the originals' sum(ep_rew), including conversion to
    # float32 only after the total has been computed. Rewards here are exactly
    # representable integers in InvertedPendulum-v5.
    total = sum(batch["rewards"].tolist())
    returns = (reward_to_go(batch["rewards"]) if agent.variant == "rtg_baseline"
               else torch.full_like(batch["rewards"], total))
    baseline = agent.critic(batch["obs"]).squeeze(-1).detach() if agent.critic is not None else torch.zeros_like(returns)
    logprob = agent.source.get_distribution(agent.policy, batch["obs"], agent.log_std.exp()).log_prob(batch["actions"]).sum(dim=-1)
    assert logprob.shape == returns.shape == baseline.shape
    actor_loss = -(logprob * (returns - baseline)).sum()
    _finite([actor_loss], "actor loss")
    agent.optimizer.zero_grad()
    actor_loss.backward()
    params = list(agent.policy.parameters()) + [agent.log_std]
    _finite([p.grad for p in params], "actor gradients")
    grad_norm = torch.linalg.vector_norm(torch.cat([p.grad.flatten() for p in params])).item()
    agent.optimizer.step()
    _finite(params, "actor parameters")

    critic_loss_value = None
    if agent.critic is not None:
        agent.critic_optimizer.zero_grad()
        target = td_targets(agent.critic, batch)
        values = agent.critic(batch["obs"]).squeeze(-1)
        assert values.shape == target.shape == returns.shape
        critic_loss = ((target - values) ** 2).mean()
        _finite([critic_loss], "critic loss")
        critic_loss.backward()
        _finite([p.grad for p in agent.critic.parameters()], "critic gradients")
        agent.critic_optimizer.step()
        _finite(list(agent.critic.parameters()), "critic parameters")
        critic_loss_value = critic_loss.item()
    return {"actor_loss": actor_loss.item(), "critic_loss": critic_loss_value,
            "std": agent.log_std.detach().exp().item(), "grad_norm": grad_norm,
            "value_mean": baseline.mean().item() if agent.critic is not None else None}


@torch.no_grad()
def collect_episode(agent, env, env_seed, action_seed=None, deterministic=False):
    context = torch.random.fork_rng(devices=[]) if action_seed is not None else nullcontext()
    with context:
        if action_seed is not None:
            torch.manual_seed(action_seed)
        obs, _ = env.reset(seed=int(env_seed))
        rows = {key: [] for key in ("obs", "actions", "rewards", "next_obs", "terminated")}
        while True:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
            action = (agent.policy(obs_tensor) if deterministic else
                      agent.source.get_action(agent.policy, obs_tensor, agent.log_std.exp()))
            action = action.cpu().numpy()
            if not np.isfinite(action).all():
                raise FloatingPointError("Non-finite sampled action")
            next_obs, reward, terminated, truncated, _ = env.step(action)
            rows["obs"].append(obs)
            rows["actions"].append(action)
            rows["rewards"].append(reward)
            rows["next_obs"].append(next_obs)
            rows["terminated"].append(terminated)
            obs = next_obs
            if terminated or truncated:
                break
        rows = {key: np.asarray(value) for key, value in rows.items()}
        rows.update({"return": float(sum(rows["rewards"])), "length": len(rows["rewards"]),
                     "truncated": bool(truncated)})
        return rows


def summarize(episodes):
    values = np.asarray([r["return"] for r in episodes], dtype=np.float64)
    return {"episodes": len(episodes), "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.,
            "min": float(values.min()), "median": float(np.median(values)), "max": float(values.max()),
            "full_length_fraction": float(np.mean([r["length"] == 1000 for r in episodes]))}


def evaluate(agent, seeds, deterministic=False):
    env = gym.make(agent.config["env_name"])
    was_training = agent.policy.training
    agent.policy.eval()
    episodes = []
    try:
        for seed in seeds:
            trajectory = collect_episode(agent, env, seed, action_seed=100_000 + seed, deterministic=deterministic)
            episodes.append({"seed": int(seed), "return": trajectory["return"], "length": trajectory["length"]})
    finally:
        env.close()
        agent.policy.train(was_training)
    return {"summary": summarize(episodes), "episodes": episodes}


def save_checkpoint(path, agent, episode, env_steps):
    path = Path(path)
    state = {"variant": agent.variant, "seed": agent.seed, "config": agent.config,
             "episode": episode, "env_steps": env_steps,
             "policy_state": agent.policy.state_dict(), "log_std": agent.log_std.detach(),
             "critic_state": agent.critic.state_dict() if agent.critic is not None else None,
             "optimizer_state": agent.optimizer.state_dict(),
             "critic_optimizer_state": agent.critic_optimizer.state_dict() if agent.critic_optimizer is not None else None,
             "torch_rng_state": torch.get_rng_state()}
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def load_checkpoint(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    agent = make_agent(state["variant"], state["seed"], state["config"])
    agent.policy.load_state_dict(state["policy_state"])
    with torch.no_grad():
        agent.log_std.copy_(state["log_std"])
    agent.optimizer.load_state_dict(state["optimizer_state"])
    if agent.critic is not None:
        agent.critic.load_state_dict(state["critic_state"])
        agent.critic_optimizer.load_state_dict(state["critic_optimizer_state"])
    return agent, state


def gradient_probe(agent, seeds):
    """Trace of empirical covariance of per-trajectory actor gradients.

    Both estimators use the SAME fixed policy and identical trajectories.
    Parameters include the mean network and log_std. No training occurs.
    Welford's vector update avoids storing all gradient vectors.
    """
    if agent.critic is None:
        raise ValueError("A learned baseline is required for this probe")
    params = list(agent.policy.parameters()) + [agent.log_std]
    size = sum(p.numel() for p in params)
    estimators = 3 if agent.variant == "rtg_baseline" else 2
    means = [torch.zeros(size, dtype=torch.float64) for _ in range(estimators)]
    m2 = [torch.zeros_like(means[0]) for _ in range(estimators)]
    total_returns = []
    env = gym.make(agent.config["env_name"])
    try:
        for count, seed in enumerate(seeds, 1):
            trajectory = collect_episode(agent, env, seed, action_seed=100_000 + seed)
            batch = tensor_batch(trajectory)
            total_returns.append(trajectory["return"])
            with torch.no_grad():
                baseline = agent.critic(batch["obs"]).squeeze(-1)
            returns = (reward_to_go(batch["rewards"]) if agent.variant == "rtg_baseline"
                       else torch.full_like(baseline, trajectory["return"]))
            for index in range(estimators):
                logprob = agent.source.get_distribution(agent.policy, batch["obs"], agent.log_std.exp()).log_prob(batch["actions"]).sum(dim=-1)
                weights = returns if index == 0 else (returns - baseline if index == 1 else trajectory["return"] - baseline)
                objective = (logprob * weights).sum()
                grad = torch.cat([g.flatten() for g in torch.autograd.grad(objective, params)]).double()
                _finite([grad], "probe gradient")
                delta = grad - means[index]
                means[index] += delta / count
                m2[index] += delta * (grad - means[index])
    finally:
        env.close()
    n = len(total_returns)
    if n < 2:
        raise ValueError("At least two trajectories are required")
    traces = [max(0., value.sum().item() / (n - 1)) for value in m2]
    result = {"trajectory_count": n, "without_baseline_variance_trace": traces[0],
            "with_baseline_variance_trace": traces[1],
            "variance_ratio": traces[1] / traces[0] if traces[0] > 0 else None,
            "mean_return": float(np.mean(total_returns)),
            "without_baseline_mean_gradient_norm": means[0].norm().item(),
            "with_baseline_mean_gradient_norm": means[1].norm().item()}
    if agent.variant == "rtg_baseline":
        result.update({"actor_return_type": "reward_to_go",
                       "full_return_baseline_variance_trace": traces[2],
                       "rtg_vs_full_baseline_variance_ratio": traces[1] / traces[2] if traces[2] > 0 else None})
    return result
