"""Behavioral checks for the paired REINFORCE experiment.

Run from this directory or with:
    .venv/bin/python -m unittest discover -s algorithms/policy_based/reinforce/experiments/learned_baseline_comparison -p 'test_*.py' -v

The two original ``train`` functions are the oracle for one-step update parity.
No source file is modified and these checks do not run the full experiment.
"""

from __future__ import annotations

import contextlib
import io
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import gymnasium as gym
import numpy as np
import torch

import core
import run as runner


def fixed_trajectory(*, terminal: bool = True, single_step: bool = False):
    states = np.asarray(
        [[0.0, 0.0, 0.0, 0.0],
         [0.1, 0.2, -0.1, 0.3],
         [-0.2, 0.1, 0.2, -0.1],
         [0.3, -0.2, 0.1, 0.0]],
        dtype=np.float32,
    )
    count = 1 if single_step else 3
    rewards = np.asarray([1.2, -0.1, 3.0][:count], dtype=np.float32)
    terminated = np.zeros(count, dtype=bool)
    terminated[-1] = terminal
    return {
        "obs": states[:count].copy(),
        "next_obs": states[1:count + 1].copy(),
        "actions": np.asarray([[0.2], [-0.4], [0.1]][:count], dtype=np.float32),
        "rewards": rewards,
        "terminated": terminated,
        "truncated": not terminal,
        "return": float(sum(float(value) for value in rewards)),
        "length": count,
    }


class FixedTrajectoryEnv:
    """Feed a known complete trajectory to an unmodified source trainer."""

    def __init__(self, trajectory):
        self.trajectory = trajectory
        self.action_space = gym.spaces.Box(-3.0, 3.0, (1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (4,), dtype=np.float64)
        self.index = 0
        self.closed = False

    def reset(self, **kwargs):
        self.index = 0
        return self.trajectory["obs"][0].copy(), {}

    def step(self, action):
        index = self.index
        np.testing.assert_array_equal(action, self.trajectory["actions"][index])
        self.index += 1
        last = self.index == self.trajectory["length"]
        return (
            self.trajectory["next_obs"][index].copy(),
            float(self.trajectory["rewards"][index]),
            bool(self.trajectory["terminated"][index]),
            bool(last and self.trajectory["truncated"]),
            {},
        )

    def close(self):
        self.closed = True


def parameters(agent):
    result = {f"policy.{key}": value.detach().clone()
              for key, value in agent.policy.state_dict().items()}
    result["log_std"] = agent.log_std.detach().clone()
    if agent.critic is not None:
        result.update({f"critic.{key}": value.detach().clone()
                       for key, value in agent.critic.state_dict().items()})
    return result


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def assert_parameters_equal(self, left, right, *, exact=True):
        left, right = parameters(left), parameters(right)
        self.assertEqual(left.keys(), right.keys())
        for name in left:
            with self.subTest(parameter=name):
                torch.testing.assert_close(
                    left[name], right[name],
                    rtol=0.0 if exact else 1e-6,
                    atol=0.0 if exact else 1e-7,
                )

    def test_paired_initialization_and_rng_isolation(self):
        torch.manual_seed(9876)
        state = torch.get_rng_state().clone()
        vanilla = core.make_agent("vanilla", 101)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        baseline = core.make_agent("baseline", 101)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        for name, value in vanilla.policy.state_dict().items():
            torch.testing.assert_close(value, baseline.policy.state_dict()[name], rtol=0, atol=0)
        torch.testing.assert_close(vanilla.log_std, baseline.log_std, rtol=0, atol=0)
        self.assertIsNone(vanilla.critic)
        self.assertIsNotNone(baseline.critic)
        self.assert_parameters_equal(baseline, core.make_agent("baseline", 101))

    def test_single_step_shapes_and_terminal_bootstrap(self):
        critic = torch.nn.Linear(4, 1)
        with torch.no_grad():
            critic.weight.zero_()
            critic.bias.fill_(7.0)
        for terminal in (False, True):
            with self.subTest(terminal=terminal):
                trajectory = fixed_trajectory(terminal=terminal, single_step=True)
                batch = core.tensor_batch(trajectory)
                target = core.td_targets(critic, batch)
                self.assertEqual(tuple(target.shape), (1,))
                self.assertFalse(target.requires_grad)
                expected = batch["rewards"] if terminal else batch["rewards"] + 7.0
                torch.testing.assert_close(target, expected, rtol=0, atol=0)
                for variant in ("vanilla", "baseline"):
                    agent = core.make_agent(variant, 101)
                    core.update(agent, batch)
                    for value in parameters(agent).values():
                        self.assertTrue(torch.isfinite(value).all().item())

    def test_one_update_matches_actual_source_train(self):
        # Both termination cases matter: the source TD target distinguishes
        # an actual terminal state from an externally truncated trajectory.
        for variant in ("vanilla", "baseline"):
            for terminal in (False, True):
                with self.subTest(variant=variant, terminal=terminal):
                    trajectory = fixed_trajectory(terminal=terminal)
                    reference = core.make_agent(variant, 303)
                    actual = core.make_agent(variant, 303)
                    source = core.load_source(variant)
                    env = FixedTrajectoryEnv(trajectory)
                    models = [reference.policy]
                    if reference.critic is not None:
                        models.append(reference.critic)
                    actions = [torch.as_tensor(value.copy()) for value in trajectory["actions"]]
                    with mock.patch.object(source.gym, "make", return_value=env), \
                         mock.patch.object(source, "network", side_effect=models), \
                         mock.patch.object(source, "Parameter", return_value=reference.log_std), \
                         mock.patch.object(source, "get_action", side_effect=actions), \
                         contextlib.redirect_stdout(io.StringIO()):
                        source.train(epochs=1)
                    self.assertEqual(env.index, trajectory["length"])
                    self.assertTrue(env.closed)
                    core.update(actual, core.tensor_batch(trajectory))
                    self.assert_parameters_equal(reference, actual, exact=False)

    def test_evaluation_preserves_rng_parameters_and_mode(self):
        agent = core.make_agent("baseline", 101)
        before_parameters = parameters(agent)
        torch.manual_seed(1234)
        np.random.seed(5678)
        random.seed(9012)
        before_torch = torch.get_rng_state().clone()
        before_numpy = np.random.get_state()
        before_python = random.getstate()
        was_training = agent.policy.training
        first = core.evaluate(agent, [50123, 50124])
        second = core.evaluate(agent, [50123, 50124])
        self.assertEqual(first, second)
        self.assertEqual(agent.policy.training, was_training)
        self.assertTrue(torch.equal(before_torch, torch.get_rng_state()))
        after_numpy = np.random.get_state()
        self.assertEqual(before_numpy[0], after_numpy[0])
        np.testing.assert_array_equal(before_numpy[1], after_numpy[1])
        self.assertEqual(before_numpy[2:], after_numpy[2:])
        self.assertEqual(before_python, random.getstate())
        for name, value in parameters(agent).items():
            torch.testing.assert_close(before_parameters[name], value, rtol=0, atol=0)

    def _short_training(self, variant, *, insert_evaluation=False):
        agent = core.make_agent(variant, 101)
        torch.manual_seed(20101)
        trajectories = []
        env = gym.make("InvertedPendulum-v5")
        try:
            for episode in range(1, 5):
                trajectory = core.collect_episode(agent, env, 2_010_000 + episode)
                core.update(agent, core.tensor_batch(trajectory))
                trajectories.append(trajectory)
                if insert_evaluation and episode == 2:
                    core.evaluate(agent, [50123, 50124])
        finally:
            env.close()
        return agent, trajectories

    def test_short_training_reproduces_with_or_without_evaluation(self):
        for variant in ("vanilla", "baseline"):
            with self.subTest(variant=variant):
                reference, expected = self._short_training(variant)
                actual, observed = self._short_training(variant, insert_evaluation=True)
                self.assert_parameters_equal(reference, actual)
                for left, right in zip(expected, observed):
                    for name in ("obs", "actions", "rewards", "next_obs", "terminated"):
                        np.testing.assert_array_equal(left[name], right[name])
                    self.assertEqual(left["return"], right["return"])
                    self.assertEqual(left["length"], right["length"])

    def test_private_collection_seed_preserves_training_rng(self):
        agent = core.make_agent("baseline", 202)
        torch.manual_seed(4321)
        state = torch.get_rng_state().clone()
        env = gym.make("InvertedPendulum-v5")
        try:
            first = core.collect_episode(agent, env, 70000, action_seed=170000)
            second = core.collect_episode(agent, env, 70000, action_seed=170000)
        finally:
            env.close()
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        for name in ("obs", "actions", "rewards", "next_obs", "terminated"):
            np.testing.assert_array_equal(first[name], second[name])

    def test_checkpoint_reload_reproduces_evaluation(self):
        for variant in ("vanilla", "baseline"):
            with self.subTest(variant=variant):
                original, trajectories = self._short_training(variant)
                steps = sum(trajectory["length"] for trajectory in trajectories)
                with tempfile.TemporaryDirectory(prefix="reinforce-test-") as temporary:
                    path = Path(temporary) / "model.pt"
                    core.save_checkpoint(path, original, episode=4, env_steps=steps)
                    loaded, metadata = core.load_checkpoint(path)
                self.assertEqual(metadata["episode"], 4)
                self.assertEqual(metadata["env_steps"], steps)
                self.assert_parameters_equal(original, loaded)
                self.assertEqual(
                    core.evaluate(original, [60123, 60124]),
                    core.evaluate(loaded, [60123, 60124]),
                )
                self.assertEqual(
                    core.evaluate(original, [60123, 60124], deterministic=True),
                    core.evaluate(loaded, [60123, 60124], deterministic=True),
                )

    def test_source_hash_verification_blocks_changed_code_before_evaluation(self):
        hashes = {variant: runner.digest(path)
                  for variant, path in core.SOURCE_FILES.items()}
        runner.verify_sources({"source_hashes": hashes})
        for variant in hashes:
            with self.subTest(changed_variant=variant):
                changed = {"source_hashes": dict(hashes, **{variant: "0" * 64})}
                with self.assertRaisesRegex(RuntimeError, "differs from the frozen protocol"):
                    runner.verify_sources(changed)
                output = Path("unused-experiment-path")
                with mock.patch.object(runner, "read_json", return_value=changed) as read_json, \
                     mock.patch.object(runner, "execute_tasks") as execute_tasks:
                    with self.assertRaisesRegex(RuntimeError, "differs from the frozen protocol"):
                        runner.final_evaluation(output, workers=1)
                read_json.assert_called_once_with(output / "protocol.json")
                execute_tasks.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
