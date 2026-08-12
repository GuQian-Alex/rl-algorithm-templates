"""
This is partial (modified) code from https://huggingface.co/learn/deep-rl-course/en/unit4/hands-on
"""

import argparse
import csv
import json
import random
import re
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import device
from torch.utils.tensorboard import SummaryWriter

import gymnasium as gym

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DEFAULT_CONFIG = {
    "h_size": 64,
    "n_training_episodes": 10000,
    "n_evaluation_episodes": 100,
    "max_t": 200,
    "gamma": 0.99,
    "lr": 5e-4,
    "env_id": "Pendulum-v1",
    "seed": 42,
    "experiment_name": "reinforce_pendulum",
    "checkpoint_every": 500,
    "print_every": 100,
    "runs_root": "runs",
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def create_run_directory(experiment_name, seed, runs_root="runs"):
    safe_experiment_name = re.sub(r"[^A-Za-z0-9._-]+","-",experiment_name).strip("-._")
    safe_experiment_name = safe_experiment_name or "experiment"
    run_time = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S%z")
    run_id = f"{safe_experiment_name}__{run_time}__seed-{seed}"
    run_directory = Path(runs_root)/run_id

    suffix = 1
    while run_directory.exists():
        run_directory = Path(runs_root)/f"{run_id}__{suffix:02d}"
        suffix += 1

    run_directory.mkdir(parents=True)
    return run_directory,run_directory.name,run_time


def save_run_config(run_directory,config,run_id,run_time):
    run_config = {
        **config,
        "run_id": run_id,
        "run_time": run_time,
        "device": str(device),
    }
    config_path = run_directory/"config.json"
    with config_path.open("w",encoding="utf-8") as config_file:
        json.dump(run_config,config_file,ensure_ascii=False,indent=2)
    return run_config


def save_training_checkpoint(
    checkpoint_path,
    policy,
    optimizer,
    episode,
    config,
    scores,
    best_score,
    run_id,
):
    checkpoint = {
        "run_id": run_id,
        "episode": episode,
        "policy_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "scores": list(scores),
        "best_score": best_score,
        "random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
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


class Policy(nn.Module):
    def __init__(self,s_size,a_size,h_size,action_low,action_high):
        super(Policy,self).__init__()
        self.fc1 = nn.Linear(s_size,h_size)
        self.fc2 = nn.Linear(h_size,a_size)
        self.fc2_ = nn.Linear(h_size,a_size)
        self.register_buffer(
            "action_scale",
            torch.as_tensor((action_high-action_low)/2.0,dtype=torch.float32),
        )
        self.register_buffer(
            "action_bias",
            torch.as_tensor((action_high+action_low)/2.0,dtype=torch.float32),
        )

    def forward(self,x):
        x = F.relu(self.fc1(x))
        mu = self.fc2(x)
        log_std = self.fc2_(x)
        return mu,log_std

    def act(self,state,deterministic=False):
        state = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.action_scale.device,
        ).unsqueeze(0)
        mu,log_std = self.forward(state)
        log_std = torch.clamp(log_std,-20,2)
        std = log_std.exp()
        dist = torch.distributions.Normal(mu,std)
        raw_action = mu if deterministic else dist.sample()
        squashed_action = torch.tanh(raw_action)
        action = self.action_bias+self.action_scale*squashed_action
        log_prob = dist.log_prob(raw_action)
        log_prob -= torch.log(
            self.action_scale*(1-squashed_action.pow(2))+1e-6
        )
        log_prob = log_prob.sum(dim=-1)
        return action.squeeze(0).detach().cpu().numpy(),log_prob

def reinforce(
    env,
    policy,
    optimizer,
    n_training_episodes,
    max_t,
    gamma,
    print_every,
    seed,
    writer=None,
    metrics_path=None,
    checkpoint_directory=None,
    checkpoint_every=None,
    checkpoint_config=None,
    run_id=None,
):
    scores_deque = deque(maxlen=100)
    scores = []
    best_score = -np.inf
    metrics_file = None
    metrics_writer = None
    metric_names = [
        "episode",
        "episode_reward",
        "reward_mean_100",
        "episode_length",
        "policy_loss",
        "discounted_return_mean",
        "discounted_return_std",
        "learning_rate",
    ]
    if metrics_path is not None:
        metrics_file = Path(metrics_path).open("w",encoding="utf-8",newline="")
        metrics_writer = csv.DictWriter(metrics_file,fieldnames=metric_names)
        metrics_writer.writeheader()

    try:
        for i_episode in range(1,n_training_episodes+1):
            saved_log_probs = []
            rewards = []
            state,obs = env.reset(seed=seed+i_episode)
            for t in range(max_t):
                action,log_prob = policy.act(state)
                saved_log_probs.append(log_prob)
                state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                rewards.append(reward)
                if done:
                    break
            episode_reward = sum(rewards)
            scores_deque.append(episode_reward)
            scores.append(episode_reward)
            returns =deque(maxlen=max_t)
            n_steps = len(rewards)

            for t in range(n_steps)[::-1]:
                disc_return_t = returns[0] if len(returns)>0 else 0
                returns.appendleft(gamma*disc_return_t + rewards[t])
            eps = np.finfo(np.float32).eps.item()
            returns = torch.tensor(returns)
            # returns = (returns-returns.mean())/(returns.std()+eps)

            policy_loss = []

            for log_probs,disc_returns in zip(saved_log_probs,returns):
                policy_loss.append(-log_probs*disc_returns)
            policy_loss = torch.cat(policy_loss).sum()

            optimizer.zero_grad()
            policy_loss.backward()
            optimizer.step()

            metrics = {
                "episode": i_episode,
                "episode_reward": episode_reward,
                "reward_mean_100": np.mean(scores_deque),
                "episode_length": n_steps,
                "policy_loss": policy_loss.item(),
                "discounted_return_mean": returns.mean().item(),
                "discounted_return_std": returns.std(unbiased=False).item(),
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            if writer is not None:
                for metric_name,metric_value in metrics.items():
                    if metric_name != "episode":
                        writer.add_scalar(f"train/{metric_name}",metric_value,i_episode)
            if metrics_writer is not None:
                metrics_writer.writerow(metrics)
                metrics_file.flush()

            if checkpoint_directory is not None:
                checkpoint_directory = Path(checkpoint_directory)
                if metrics["reward_mean_100"] > best_score:
                    best_score = metrics["reward_mean_100"]
                    save_training_checkpoint(
                        checkpoint_directory/"best.pt",
                        policy,
                        optimizer,
                        i_episode,
                        checkpoint_config,
                        scores,
                        best_score,
                        run_id,
                    )
                if checkpoint_every and i_episode % checkpoint_every == 0:
                    save_training_checkpoint(
                        checkpoint_directory/f"episode_{i_episode:06d}.pt",
                        policy,
                        optimizer,
                        i_episode,
                        checkpoint_config,
                        scores,
                        best_score,
                        run_id,
                    )

            if i_episode % print_every == 0:
                print("Episode {}\tAverage Score: {:.2f}".format(i_episode, np.mean(scores_deque)))
                if writer is not None:
                    writer.flush()
    finally:
        if metrics_file is not None:
            metrics_file.close()

    if checkpoint_directory is not None:
        save_training_checkpoint(
            checkpoint_directory/"final.pt",
            policy,
            optimizer,
            n_training_episodes,
            checkpoint_config,
            scores,
            best_score,
            run_id,
        )

    return scores

@torch.no_grad()
def evaluate_agent(env, max_steps, n_eval_episodes, policy, seed):
    """
    Evaluate the agent for ``n_eval_episodes`` episodes and returns average reward and std of reward.
    :param env: The evaluation environment
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param policy: The Reinforce agent
    """
    episode_rewards = []
    for episode in range(n_eval_episodes):
        state, info = env.reset(seed=seed+100_000+episode)
        step = 0
        done = False
        total_rewards_ep = 0

        for step in range(max_steps):
            action, _ = policy.act(state,deterministic=True)
            new_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_rewards_ep += reward

            if done:
                break
            state = new_state
        episode_rewards.append(total_rewards_ep)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train REINFORCE on Pendulum-v1")
    parser.add_argument("--experiment-name",default=DEFAULT_CONFIG["experiment_name"])
    parser.add_argument("--seed",type=int,default=DEFAULT_CONFIG["seed"])
    parser.add_argument("--env-id",default=DEFAULT_CONFIG["env_id"])
    parser.add_argument("--h-size",type=int,default=DEFAULT_CONFIG["h_size"])
    parser.add_argument(
        "--n-training-episodes",
        type=int,
        default=DEFAULT_CONFIG["n_training_episodes"],
    )
    parser.add_argument(
        "--n-evaluation-episodes",
        type=int,
        default=DEFAULT_CONFIG["n_evaluation_episodes"],
    )
    parser.add_argument("--max-t",type=int,default=DEFAULT_CONFIG["max_t"])
    parser.add_argument("--gamma",type=float,default=DEFAULT_CONFIG["gamma"])
    parser.add_argument("--lr",type=float,default=DEFAULT_CONFIG["lr"])
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=DEFAULT_CONFIG["checkpoint_every"],
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=DEFAULT_CONFIG["print_every"],
    )
    parser.add_argument("--runs-root",default=DEFAULT_CONFIG["runs_root"])
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = vars(args).copy()
    set_seed(config["seed"])
    print(device)

    env = gym.make(config["env_id"])
    eval_env = gym.make(config["env_id"])
    env.action_space.seed(config["seed"])
    eval_env.action_space.seed(config["seed"]+100_000)
    config["state_space"] = env.observation_space.shape[0]
    config["action_space"] = env.action_space.shape[0]

    try:
        policy = Policy(
            config["state_space"],
            config["action_space"],
            config["h_size"],
            env.action_space.low,
            env.action_space.high,
        ).to(device)
        optimizer = optim.Adam(policy.parameters(),lr=config["lr"])

        run_dir,run_id,run_time = create_run_directory(
            config["experiment_name"],
            config["seed"],
            config["runs_root"],
        )
        run_config = save_run_config(run_dir,config,run_id,run_time)
        tensorboard_writer = SummaryWriter(log_dir=str(run_dir))
        tensorboard_writer.add_text(
            "run/config",
            f"```json\n{json.dumps(run_config,ensure_ascii=False,indent=2)}\n```",
            0,
        )

        try:
            scores = reinforce(
                env,
                policy,
                optimizer,
                config["n_training_episodes"],
                config["max_t"],
                config["gamma"],
                config["print_every"],
                config["seed"],
                writer=tensorboard_writer,
                metrics_path=run_dir/"metrics.csv",
                checkpoint_directory=run_dir/"checkpoints",
                checkpoint_every=config["checkpoint_every"],
                checkpoint_config=run_config,
                run_id=run_id,
            )
            mean_reward,std_reward = evaluate_agent(
                eval_env,
                config["max_t"],
                config["n_evaluation_episodes"],
                policy,
                config["seed"],
            )
            tensorboard_writer.add_scalar(
                "eval/mean_reward",
                mean_reward,
                config["n_training_episodes"],
            )
            tensorboard_writer.add_scalar(
                "eval/std_reward",
                std_reward,
                config["n_training_episodes"],
            )
        finally:
            tensorboard_writer.close()
    finally:
        env.close()
        eval_env.close()

    print(f"Run directory: {run_dir}")
    print(f"Evaluation reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    return run_dir,scores,mean_reward,std_reward


if __name__ == "__main__":
    main()
