from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("fitted extra is not installed")

import pandas as pd
import torch

from kdd2027_benchmark.cli import build_parser
from kdd2027_benchmark.fitted import workflow
from kdd2027_benchmark.fitted import run_kdd262_matched_fitted_simulator_ope as runner
from kdd2027_benchmark.fitted.kdd248_full_episode import (
    CalibratedRolloutProfile,
    LearnedSourceSimulator,
    ResponseRegime,
    TaskShape,
    build_unfitted_components,
    resolve_reward_emission_mode,
)


def reference_terminal_once(batch, components, action_dim, seed):
    valid = np.asarray(batch.valid_steps, dtype=bool)
    lengths = valid.sum(axis=1)
    rows = np.arange(len(lengths))
    last = lengths - 1
    actions = torch.as_tensor(np.asarray(batch.actions)[rows, last].astype(np.int64))
    context = torch.cat([
        torch.as_tensor(np.asarray(batch.observations)[rows, last], dtype=torch.float32),
        torch.as_tensor(np.asarray(batch.masks)[rows, last], dtype=torch.float32),
        torch.as_tensor(np.asarray(batch.recency)[rows, last], dtype=torch.float32),
        torch.nn.functional.one_hot(actions, num_classes=action_dim).to(torch.float32),
    ], dim=-1)
    with torch.inference_mode():
        mean, _ = components.reward(context)
    probability = np.clip((1.0 + mean.numpy().astype(np.float64)) / 2.0, 0.0, 1.0)
    uniform = np.random.default_rng([int(seed), 277003]).random(len(lengths))
    outcome = np.where(uniform < probability, 1.0, -1.0)
    rewards = np.where(valid, 0.0, np.nan).astype(np.float32)
    rewards[rows, last] = outcome
    corrected = dataclasses.replace(batch, rewards=rewards)
    corrected.validate()
    return corrected


class TerminalRewardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.task = TaskShape("fixture", 3, 4, 5)
        self.components = build_unfitted_components(
            self.task, "gaussian_recurrent", 3408, hidden_dim=8, latent_dim=8
        )
        self.profile = CalibratedRolloutProfile(
            state_mean=np.zeros(3), state_scale=np.ones(3),
            state_correlation_cholesky=np.eye(3),
            predicted_delta_center=np.zeros(3), predicted_delta_scale=np.ones(3),
            learned_signal_center=np.zeros(3),
            standardized_lower=np.full(3, -4.0), standardized_upper=np.full(3, 4.0),
            initial_action_probability=np.full(4, 0.25),
            action_transition_probability=np.full((4, 4), 0.25),
            initial_mask_probability=np.full(3, 0.8),
            mask_transition_probability=np.full((3, 2), 0.8),
            length_probability=np.full(5, 0.2),
        )

    def simulator(self, mode, calibrated=False, task=None):
        return LearnedSourceSimulator(
            task or self.task, self.components,
            rollout_profile=self.profile if calibrated else None,
            reward_emission_mode=mode,
        )

    def assert_terminal_rewards(self, batch) -> None:
        rows = np.arange(len(batch.lengths))
        last = batch.lengths - 1
        terminal = np.zeros_like(batch.valid_steps)
        terminal[rows, last] = True
        self.assertEqual(batch.rewards.dtype, np.float32)
        self.assertTrue(np.isin(batch.rewards[terminal], (-1.0, 1.0)).all())
        self.assertTrue((batch.rewards[batch.valid_steps & ~terminal] == 0.0).all())
        self.assertTrue(np.isnan(batch.rewards[~batch.valid_steps]).all())
        np.testing.assert_array_equal(
            np.sum((batch.rewards != 0.0) & batch.valid_steps, axis=1),
            np.ones(len(rows), dtype=int),
        )
        for discount in (0.01, 0.9, 0.99, 1.0):
            returns = batch.discounted_returns(discount)
            self.assertTrue(((returns >= -1.0) & (returns <= 1.0)).all())
            np.testing.assert_array_equal(returns, batch.rewards[rows, last] * discount**last)
        batch.validate()

    def test_terminal_once_timing_support_and_returns(self) -> None:
        for calibrated in (False, True):
            for regime in ResponseRegime:
                with self.subTest(calibrated=calibrated, regime=regime):
                    batch = self.simulator("terminal_once", calibrated).simulate(64, 277001, regime)
                    self.assert_terminal_rewards(batch)
                    self.assertTrue((batch.lengths == 1).any())
                    self.assertTrue((batch.lengths == self.task.horizon).any())

    def test_observed_regime_equals_reference_bit_for_bit(self) -> None:
        for calibrated in (False, True):
            with self.subTest(calibrated=calibrated):
                legacy = self.simulator("per_step_head", calibrated).simulate(64, 277001, ResponseRegime.OBSERVED)
                expected = reference_terminal_once(legacy, self.components, 4, 277001)
                actual = self.simulator("terminal_once", calibrated).simulate(64, 277001, ResponseRegime.OBSERVED)
                self.assertEqual(actual.rewards.tobytes(), expected.rewards.tobytes())
                self.assertEqual(actual.byte_digest(), expected.byte_digest())

    def test_other_regimes_use_rollout_action_conditioning(self) -> None:
        with torch.no_grad():
            self.components.reward.head.weight.zero_()
            self.components.reward.head.bias.zero_()
            self.components.reward.head.weight[0, -4:] = torch.tensor([-2.0, -0.5, 0.5, 2.0])
        forced = np.full((64, 5), 1, dtype=np.int64)
        uniform = np.random.default_rng([277001, 277003]).random(64)
        for regime, probability in (
            (ResponseRegime.NULL, 0.5),
            (ResponseRegime.PERTURBED, 0.75),
            (ResponseRegime.BOUNDED, 0.25),
        ):
            with self.subTest(regime=regime):
                batch = self.simulator("terminal_once", True).simulate(64, 277001, regime, forced_actions=forced)
                np.testing.assert_array_equal(
                    batch.rewards[np.arange(64), batch.lengths - 1],
                    np.where(uniform < probability, 1.0, -1.0),
                )

    def test_common_terminal_uniforms_across_policies_and_lengths(self) -> None:
        with torch.no_grad():
            self.components.reward.head.weight.zero_()
            self.components.reward.head.bias.zero_()
            self.components.reward.head.weight[0, -4:] = torch.tensor([-0.5, 0.5, 0.0, 0.0])
            self.components.termination.head.weight.zero_()
            self.components.termination.head.bias.zero_()
            self.components.termination.head.weight[0, -4:] = torch.tensor([-1.0, 1.0, 0.0, 0.0])
        simulator = self.simulator("terminal_once")
        uniform = np.random.default_rng([277001, 277003]).random(64)
        default_rng = np.random.default_rng
        batches = []
        with patch("kdd2027_benchmark.fitted.kdd248_full_episode.np.random.default_rng", wraps=default_rng) as generator:
            for action, probability in ((0, 0.25), (1, 0.75)):
                def policy(values, masks, recency, previous, step):
                    return np.broadcast_to(np.eye(4)[action], (len(values), 4))
                batch = simulator.simulate(64, 277001, ResponseRegime.OBSERVED, policy=policy, policy_seed=action + 19)
                batches.append(batch)
                np.testing.assert_array_equal(
                    batch.rewards[np.arange(64), batch.lengths - 1],
                    np.where(uniform < probability, 1.0, -1.0),
                )
            self.assertEqual(generator.call_count, 2)
            for call in generator.call_args_list:
                self.assertEqual(call.args, ([277001, 277003],))
        self.assertFalse(np.array_equal(batches[0].lengths, batches[1].lengths))

    def test_probability_clipping_and_strict_comparison(self) -> None:
        for mean, expected in ((-4.0, -1.0), (4.0, 1.0)):
            with self.subTest(mean=mean):
                with torch.no_grad():
                    self.components.reward.head.weight.zero_()
                    self.components.reward.head.bias[0] = mean
                batch = self.simulator("terminal_once").simulate(8, 277001, ResponseRegime.OBSERVED)
                self.assertTrue((batch.rewards[np.arange(8), batch.lengths - 1] == expected).all())
        with torch.no_grad():
            self.components.reward.head.bias[0] = 0.0
        with patch("kdd2027_benchmark.fitted.kdd248_full_episode.np.random.default_rng") as generator:
            generator.return_value.random.return_value = np.full(8, 0.5)
            batch = self.simulator("terminal_once").simulate(8, 277001, ResponseRegime.OBSERVED)
        self.assertTrue((batch.rewards[np.arange(8), batch.lengths - 1] == -1.0).all())

    def test_per_step_head_matches_v211_reward_bytes(self) -> None:
        # Captured before edits at 050405d, with the repository's pinned dependencies.
        # Digests are single-thread values; bitwise rollout output depends on the torch thread count.
        expected = {
            False: (
                "66549240223e3ed15eed909ccf3aa34630b7b669e01ae6aa2bfde17ecddcda2b",
                "5281a92104204298f9e9d293b9da11509beea8601be7cea2323935969d1e0c3b",
                "ece84457e0385594b3dda2a17692f3f4b2ce9bfeec5b40d8564d1d10b755e69b",
                "4ad8f7a7a0f351281e355fab099381c5d53f2db9fd4af60a75e3da7945b211e1",
            ),
            True: (
                "0c3d040b898e63dec9916c29bdae50b3593d165f7b2006803d3bd666af2c4170",
                "abb47c3affdbf8aed68c157095f018dc632e482c438696a312f5b3a311cc9aa0",
                "bca2ed96eb20288dd19117d554b0e63983eb54c43e5b3b509b182d1059414015",
                "502f7a24dc6339a6e2192707290f215201bb77f0f7aa57aec5625b5858d1022a",
            ),
        }
        previous_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous_threads)
        torch.set_num_threads(1)
        for calibrated, digests in expected.items():
            for regime, digest in zip(ResponseRegime, digests):
                with self.subTest(calibrated=calibrated, regime=regime):
                    batch = self.simulator("per_step_head", calibrated).simulate(64, 277001, regime)
                    self.assertEqual(hashlib.sha256(batch.rewards.tobytes()).hexdigest(), digest)

    def test_dense_defaults_preserve_legacy_batches(self) -> None:
        config = workflow._load(workflow.WORKFLOW_CONFIG)
        for name in ("respiratory_support", "shock"):
            with self.subTest(task=name):
                task = dataclasses.replace(self.task, name=name)
                mode = resolve_reward_emission_mode(config, name)
                self.assertEqual(mode, "per_step_head")
                default = self.simulator(mode, True, task).simulate(64, 277001, ResponseRegime.OBSERVED)
                legacy = self.simulator("per_step_head", True, task).simulate(64, 277001, ResponseRegime.OBSERVED)
                self.assertEqual(default.byte_digest(), legacy.byte_digest())
                self.assertTrue(np.any(default.rewards[default.valid_steps] != 0.0))

    def test_configuration_channels_and_explicit_overrides(self) -> None:
        for path in (workflow.WORKFLOW_CONFIG, workflow.MATCHED_CONFIG):
            config = workflow._load(path)
            for name in config["task_shapes"]:
                expected = "per_step_head" if name in ("respiratory_support", "shock") else "terminal_once"
                self.assertEqual(resolve_reward_emission_mode(config, name), expected)
                self.assertEqual(resolve_reward_emission_mode(config, name, "per_step_head"), "per_step_head")
            config["reward_channels"]["aki"] = "shock_next_mbp_component"
            self.assertEqual(resolve_reward_emission_mode(config, "aki"), "per_step_head")
            config["reward_emission_overrides"]["aki"] = "terminal_once"
            self.assertEqual(resolve_reward_emission_mode(config, "aki"), "terminal_once")
            self.assertEqual(resolve_reward_emission_mode(config, "aki", "per_step_head"), "per_step_head")

    def test_missing_or_unknown_mode_fails_loudly(self) -> None:
        with self.assertRaises(TypeError):
            LearnedSourceSimulator(self.task, self.components)
        with self.assertRaisesRegex(ValueError, "unsupported reward emission mode: unknown"):
            self.simulator("unknown")
        config = workflow._load(workflow.WORKFLOW_CONFIG)
        for source in ("channel", "task", "argument"):
            changed = copy.deepcopy(config)
            override = None
            if source == "channel":
                changed["reward_emission_modes"][changed["reward_channels"]["aki"]] = "unknown"
            elif source == "task":
                changed["reward_emission_overrides"]["aki"] = "unknown"
            else:
                override = "unknown"
            with self.subTest(source=source):
                with self.assertRaisesRegex(ValueError, "unsupported reward emission mode: unknown"):
                    resolve_reward_emission_mode(changed, "aki", override)
        del config["reward_emission_modes"]
        with self.assertRaises(KeyError):
            resolve_reward_emission_mode(config, "aki")

    def test_cli_exposes_explicit_legacy_mode(self) -> None:
        parser = build_parser()
        for command in (
            ["fitted-synthetic-smoke", "--output", "receipt.json"],
            ["fitted-simulator", "--mode", "evaluate", "--constructor-root", "synthetic", "--output", "results"],
        ):
            self.assertIsNone(parser.parse_args(command).reward_emission_mode)
            args = parser.parse_args([*command, "--reward-emission-mode", "per_step_head"])
            self.assertEqual(args.reward_emission_mode, "per_step_head")

    def test_credentialed_builder_and_policy_extension_inherit_mode(self) -> None:
        config = workflow._load(workflow.WORKFLOW_CONFIG)
        config["task_shapes"]["aki"] = [3, 4, 5]
        config["tasks"] = ["aki"]
        def load(path):
            return config if path == workflow.WORKFLOW_CONFIG else {"tasks": ["aki"]}
        for override, expected in ((None, "terminal_once"), ("per_step_head", "per_step_head")):
            with self.subTest(override=override), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "checkpoints").mkdir()
                (root / "profiles").mkdir()
                (root / "checkpoints/aki__gaussian_recurrent__seed3411.pt").write_bytes(b"synthetic fixture")
                (root / "profiles/aki__calibrated_profile.pt").write_bytes(b"synthetic fixture")
                def train(task, *args):
                    self.assertEqual(task.simulator.reward_emission_mode, expected)
                    return [], None, None, None, None
                def extend(task, groups, *args, **kwargs):
                    self.assertEqual(task.simulator.reward_emission_mode, expected)
                    batch = task.simulator.simulate(8, 277001, ResponseRegime.OBSERVED)
                    if expected == "terminal_once":
                        self.assert_terminal_rewards(batch)
                    return groups, [], {}
                with (
                    patch.object(workflow, "_load", side_effect=load),
                    patch.object(workflow, "_effective_config", side_effect=lambda value, **kwargs: value),
                    patch.object(workflow, "load_source_components", return_value=(self.components, {})),
                    patch.object(workflow, "_profile_from_path", return_value=self.profile),
                    patch.object(workflow, "train_policy_groups", side_effect=train),
                    patch.object(workflow, "extend_fitted_policy_groups", side_effect=extend) as extension,
                    patch.object(workflow, "direct_references", return_value=([], [], None, {})),
                    patch.object(workflow, "evaluate_ope_datasets", return_value=([], [], [])),
                ):
                    receipt = workflow.evaluate_credentialed(root, root, root / "output", "cpu", True, override)
                self.assertEqual(receipt["reward_emission_modes"], {"aki": expected})
                extension.assert_called_once()

    def test_kdd262_frozen_loader_selects_configuration_mode(self) -> None:
        config = workflow._load(workflow.MATCHED_CONFIG)
        config["tasks"] = ["aki"]
        config["task_shapes"]["aki"] = [3, 4, 5]
        config["e4r"] = {"path": "synthetic_fixture", "decision": "fixture", "result_tree_sha256": "fixture"}
        for expected in ("terminal_once", "per_step_head"):
            with self.subTest(mode=expected), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                contract_path = root / "contract.json"
                contract_path.write_text(json.dumps({"tasks": ["aki"], "hidden_dim": 8, "latent_dim": 8, "sensitivity_bound": 0.1}))
                config["e3c_contract_config"] = str(contract_path)
                config["reward_emission_overrides"] = {"aki": expected}
                checkpoint = root / "aki__gaussian_recurrent__seed3408.pt"
                checkpoint.write_bytes(b"synthetic fixture")
                profile = root / "aki__calibrated_profile.pt"
                profile.write_bytes(b"synthetic fixture")
                digest = hashlib.sha256(b"synthetic fixture").hexdigest()
                selected = pd.DataFrame([{"task": "aki", "source_family": "gaussian_recurrent", "initialization_seed": 3408, "checkpoint_sha256": digest}])
                profiles = pd.DataFrame([{"task": "aki", "profile_sha256": digest}])
                with (
                    patch.object(runner, "decision_token", return_value="fixture"),
                    patch.object(runner, "flat_tree_sha256", return_value="fixture"),
                    patch.object(runner.pd, "read_csv", side_effect=[selected, profiles]),
                    patch.object(runner, "load_source_components", return_value=(self.components, {})),
                    patch.object(runner.torch, "load", return_value=dataclasses.asdict(self.profile)),
                ):
                    tasks, receipts = runner._load_frozen_tasks(config, root, root)
                self.assertEqual(tasks[0].simulator.reward_emission_mode, expected)
                self.assertTrue(all(row["status"] == "pass" for row in receipts))


if __name__ == "__main__":
    unittest.main()
