"""与 Agent 配套的纯 REINFORCE：每批采集完整回合，以整回合总奖励作为权重。"""

import argparse

import gymnasium as gym
import numpy as np
import torch
from torch.optim import Adam
if __package__:
    from .agent import Agent
else:
    from agent import Agent



class REINFORCE:
    """支持向量 Box 观测，以及 Discrete 或向量 Box 动作。

    batch_num 是每次更新采集的回合数，epoch 是更新次数，sizes 是隐藏层宽度。
    act_type 默认从环境推断。特殊游戏的数据预处理放在外部 wrapper 中，
    再通过 env 参数传入；不传 env 时使用 env_name 创建环境。
    """

    def __init__(
        self,
        env_name="Pendulum-v1",
        lr=0.001,
        batch_num=10,
        epoch=1000,
        sizes=(64,),
        act_type=None,
        *,
        env=None,
    ):
        if not isinstance(batch_num, int) or batch_num < 1:
            raise ValueError("batch_num must be a positive integer")
        if not isinstance(epoch, int) or epoch < 1:
            raise ValueError("epoch must be a positive integer")

        self.batch_num = batch_num
        self.epoch = epoch
        self.env = gym.make(env_name) if env is None else env
        try:
            obs_space = self.env.observation_space
            act_space = self.env.action_space
            if not isinstance(obs_space, gym.spaces.Box) or len(obs_space.shape) != 1:
                raise ValueError("Use an observation wrapper to provide a 1D Box")
            self.obs_dim = obs_space.shape[0]

            if isinstance(act_space, gym.spaces.Discrete):
                inferred_type = "discrete"
                self.act_dim = act_space.n
            elif (
                isinstance(act_space, gym.spaces.Box)
                and len(act_space.shape) == 1
                and np.issubdtype(act_space.dtype, np.floating)
            ):
                inferred_type = "continuous"
                self.act_dim = act_space.shape[0]
            else:
                raise ValueError("Action space must be Discrete or a 1D floating-point Box")

            if act_type is not None and act_type != inferred_type:
                raise ValueError(f"act_type must be {inferred_type!r} for this environment")
            self.act_type = inferred_type
            self.agent = Agent([self.obs_dim] + list(sizes) + [self.act_dim], self.act_type)
            # agent.parameters() 同时包含策略网络参数和连续动作的 log_std。
            self.optimizer = Adam(self.agent.parameters(), lr=lr)
        except Exception:
            self.env.close()
            raise

    def collect_data(self):
        """收集一批完整回合，返回训练 Tensor 和各回合统计。

        obs: [N, obs_dim]；act: 离散 [N] / 连续 [N, act_dim]；weights: [N]。
        每个回合的所有时间步共享该回合的总回报，不使用 reward-to-go。
        环境必须能终止回合，或在外部用 TimeLimit 设置步数上限。
        """
        batch_obs, batch_act, batch_weights = [], [], []
        ep_returns, ep_lengths = [], []
        device = next(self.agent.parameters()).device

        for _ in range(self.batch_num):
            obs, _ = self.env.reset()
            ep_return, ep_length = 0.0, 0

            while True:
                # 先复制再 step，防止环境复用并修改同一块观测数组。
                batch_obs.append(np.array(obs, dtype=np.float32, copy=True))
                obs_tensor = torch.as_tensor(batch_obs[-1], dtype=torch.float32, device=device)
                action = self.agent.get_action(obs_tensor)

                if self.act_type == "discrete":
                    raw_action = int(action.item())
                    env_action = raw_action + int(self.env.action_space.start)
                else:
                    raw_action = action.cpu().numpy().copy()
                    space = self.env.action_space
                    env_action = np.clip(raw_action, space.low, space.high).astype(space.dtype)

                # 裁剪仅用于环境交互；高斯 log_prob 必须对应原始采样动作。
                batch_act.append(raw_action)
                obs, reward, terminated, truncated, _ = self.env.step(env_action)
                ep_return += float(reward)
                ep_length += 1
                if terminated or truncated:
                    break

            ep_returns.append(ep_return)
            ep_lengths.append(ep_length)
            batch_weights.extend([ep_return] * ep_length)

        act_dtype = torch.long if self.act_type == "discrete" else torch.float32
        return {
            "obs": torch.as_tensor(np.asarray(batch_obs), dtype=torch.float32, device=device),
            "act": torch.as_tensor(np.asarray(batch_act), dtype=act_dtype, device=device),
            "weights": torch.as_tensor(batch_weights, dtype=torch.float32, device=device),
            "ep_returns": ep_returns,
            "ep_lengths": ep_lengths,
        }

    def train(self):
        """每批更新一次，返回各轮统计；训练结束或出现异常时关闭环境。"""
        history = []
        self.agent.train()
        try:
            for epoch in range(1, self.epoch + 1):
                batch = self.collect_data()
                # 采样不保留计算图，训练时重新计算 log_prob 来建立梯度。
                logprob = self.agent.get_logprob(batch["obs"], batch["act"])
                # 每条轨迹内部求和，再对轨迹取平均，而不是对所有时间步取平均。
                loss = -(logprob * batch["weights"]).sum() / self.batch_num

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                metrics = {
                    "epoch": epoch,
                    "loss": loss.item(),
                    "mean_return": float(np.mean(batch["ep_returns"])),
                    "mean_length": float(np.mean(batch["ep_lengths"])),
                }
                history.append(metrics)
                print(
                    f"epoch {epoch}/{self.epoch} | "
                    f"return {metrics['mean_return']:.2f} | "
                    f"length {metrics['mean_length']:.1f} | "
                    f"loss {metrics['loss']:.4f}"
                )
        finally:
            self.env.close()
        return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="Pendulum-v1", help="Gymnasium 环境 ID")
    parser.add_argument("--epochs", type=int, default=1000, help="策略更新次数")
    parser.add_argument("--batch-num", type=int, default=10, help="每次更新采集的回合数")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--sizes", type=int, nargs="+", default=[64], help="隐藏层宽度")
    args = parser.parse_args()
    REINFORCE(
        env_name=args.env,
        lr=args.lr,
        batch_num=args.batch_num,
        epoch=args.epochs,
        sizes=args.sizes,
    ).train()
