"""Verify the actor-only reward-to-go extension and its source-code parity."""

from __future__ import annotations

import contextlib
import io
import unittest
from unittest import mock

import numpy as np
import torch

import core
from test_experiment import FixedTrajectoryEnv, fixed_trajectory, parameters


class RewardToGoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_reward_to_go_values_dtype_shape_and_input_unchanged(self):
        source = core.load_source("rtg_baseline")
        cases = [
            ([1.0, 2.0, 3.0], [6.0, 5.0, 3.0]),
            ([2.0], [2.0]),
            ([-2.0], [-2.0]),
            ([-1.0, 2.0, -3.0], [-2.0, -1.0, -3.0]),
            ([], []),
        ]
        for values, expected in cases:
            with self.subTest(rewards=values):
                numpy_rewards = np.asarray(values, dtype=np.float32)
                numpy_before = numpy_rewards.copy()
                numpy_result = source.reward_to_go(numpy_rewards)
                np.testing.assert_array_equal(numpy_rewards, numpy_before)
                np.testing.assert_array_equal(numpy_result, np.asarray(expected, dtype=np.float32))
                self.assertEqual(numpy_result.dtype, np.dtype(np.float32))
                self.assertEqual(numpy_result.shape, (len(values),))

                tensor_rewards = torch.as_tensor(values, dtype=torch.float32)
                tensor_before = tensor_rewards.clone()
                tensor_result = core.reward_to_go(tensor_rewards)
                torch.testing.assert_close(tensor_rewards, tensor_before, rtol=0, atol=0)
                torch.testing.assert_close(tensor_result, torch.tensor(expected, dtype=torch.float32), rtol=0, atol=0)
                self.assertEqual(tensor_result.dtype, torch.float32)
                self.assertEqual(tuple(tensor_result.shape), (len(values),))

        list_rewards = [1, 2, 3]
        np.testing.assert_array_equal(source.reward_to_go(list_rewards), [6, 5, 3])
        self.assertEqual(list_rewards, [1, 2, 3])
        with self.assertRaisesRegex(ValueError, "shape"):
            source.reward_to_go([[1.0], [2.0]])
        with self.assertRaisesRegex(ValueError, "shape"):
            core.reward_to_go(torch.tensor([[1.0], [2.0]]))

    def test_paired_baseline_initialization_including_critic(self):
        torch.manual_seed(13579)
        state = torch.get_rng_state().clone()
        baseline = core.make_agent("baseline", 101)
        rtg = core.make_agent("rtg_baseline", 101)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        for name, value in parameters(baseline).items():
            with self.subTest(parameter=name):
                torch.testing.assert_close(value, parameters(rtg)[name], rtol=0, atol=0)

    def test_rtg_one_update_matches_actual_source_train(self):
        for terminal in (False, True):
            for single_step in (False, True):
                with self.subTest(terminal=terminal, single_step=single_step):
                    trajectory = fixed_trajectory(terminal=terminal, single_step=single_step)
                    reference = core.make_agent("rtg_baseline", 303)
                    actual = core.make_agent("rtg_baseline", 303)
                    source = core.load_source("rtg_baseline")
                    env = FixedTrajectoryEnv(trajectory)
                    actions = [torch.as_tensor(value.copy()) for value in trajectory["actions"]]
                    with mock.patch.object(source.gym, "make", return_value=env), \
                         mock.patch.object(source, "network", side_effect=[reference.policy, reference.critic]), \
                         mock.patch.object(source, "Parameter", return_value=reference.log_std), \
                         mock.patch.object(source, "get_action", side_effect=actions), \
                         contextlib.redirect_stdout(io.StringIO()):
                        source.train(epochs=1)
                    self.assertEqual(env.index, trajectory["length"])
                    self.assertTrue(env.closed)
                    core.update(actual, core.tensor_batch(trajectory))
                    for name, value in parameters(reference).items():
                        with self.subTest(parameter=name):
                            torch.testing.assert_close(value, parameters(actual)[name], rtol=1e-6, atol=1e-7)

    def test_actor_change_preserves_critic_update_on_same_trajectory(self):
        for terminal in (False, True):
            with self.subTest(terminal=terminal):
                batch = core.tensor_batch(fixed_trajectory(terminal=terminal))
                baseline = core.make_agent("baseline", 202)
                rtg = core.make_agent("rtg_baseline", 202)
                baseline_metrics = core.update(baseline, batch)
                rtg_metrics = core.update(rtg, batch)
                self.assertEqual(baseline_metrics["critic_loss"], rtg_metrics["critic_loss"])
                for name, value in baseline.critic.state_dict().items():
                    torch.testing.assert_close(value, rtg.critic.state_dict()[name], rtol=0, atol=0)
                for left, right in zip(baseline.critic.parameters(), rtg.critic.parameters()):
                    torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)
                # The fixed trajectory has distinct future returns, so this
                # case also guards against accidentally keeping full-return weights.
                self.assertNotEqual(baseline_metrics["actor_loss"], rtg_metrics["actor_loss"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
