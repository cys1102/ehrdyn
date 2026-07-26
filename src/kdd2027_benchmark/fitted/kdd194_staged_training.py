from __future__ import annotations

import hashlib
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from . import kdd161_model_free as frozen_mf
from . import kdd164_metric_selection as frozen_wm
from .kdd155v3_model_free import (
    BCQ,
    IQL,
    MLP,
    DecisionTransformer,
    PolicyData,
    _infer,
    _loader,
    _sequence_arrays,
    model_digest,
    normalize,
    one_hot,
    validate_probability,
)


PolicySnapshot = Callable[[nn.Module], np.ndarray]


def mean_policy_jsd(probabilities: list[np.ndarray]) -> float:
    if len(probabilities) < 2:
        return math.inf
    values = []
    for left, right in zip(probabilities[:-1], probabilities[1:], strict=True):
        if not validate_probability(left) or not validate_probability(right):
            raise RuntimeError("invalid validation-policy probability in JSD gate")
        midpoint = 0.5 * (left + right)
        floor = 1e-12
        kl_left = np.sum(left * (np.log(np.maximum(left, floor)) - np.log(np.maximum(midpoint, floor))), axis=1)
        kl_right = np.sum(right * (np.log(np.maximum(right, floor)) - np.log(np.maximum(midpoint, floor))), axis=1)
        values.append(float(np.mean(0.5 * (kl_left + kl_right))))
    return float(np.mean(values))


def late_relative_improvement(curve: list[float], window: int) -> float:
    if len(curve) < window:
        return math.inf
    local = np.asarray(curve[-window:], dtype=float)
    return float((local[0] - local[-1]) / max(abs(local[0]), 1e-12))


@dataclass(slots=True)
class StagedReceipt:
    selected_epoch: int
    stopped_epoch: int
    stage_reached: int
    validation_curve: list[float]
    late_relative_improvement: float
    mean_validation_policy_jsd: float
    stale_epochs: int
    convergence_status: str
    stop_reason: str
    gradient_updates: int
    runtime_seconds: float


def staged_fit(
    model: nn.Module,
    batches: Callable[[int], Any],
    train_loss: Callable[[nn.Module, tuple[torch.Tensor, ...]], torch.Tensor],
    validation_score: Callable[[nn.Module], float],
    validation_policy: PolicySnapshot,
    cfg: dict[str, Any],
    device: torch.device,
) -> StagedReceipt:
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
    best_state: dict[str, torch.Tensor] | None = None
    best_value = math.inf
    best_epoch = 0
    stale = 0
    curve: list[float] = []
    state_window: deque[dict[str, torch.Tensor]] = deque(maxlen=int(cfg["policy_checkpoint_window"]))
    updates = 0
    started = time.perf_counter()
    caps = [int(value) for value in cfg["staged_caps"]]
    stopped = caps[-1]
    stage_reached = caps[-1]
    stop_reason = "final_cap_reached"
    relative = math.inf
    jsd = math.inf
    for epoch in range(1, caps[-1] + 1):
        model.train()
        for batch in batches(epoch):
            batch = tuple(value.to(device, non_blocking=True) for value in batch)
            optimizer.zero_grad(set_to_none=True)
            loss = train_loss(model, batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(cfg["gradient_clip_norm"]))
            optimizer.step()
            updates += 1
        model.eval()
        value = float(validation_score(model))
        if not np.isfinite(value):
            raise RuntimeError("nonfinite validation objective")
        curve.append(value)
        state_window.append({name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()})
        if value < best_value - float(cfg["minimum_metric_improvement"]):
            best_value = value
            best_epoch = epoch
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if epoch < int(cfg["minimum_epochs"]):
            continue
        at_stage = epoch in caps
        patient_plateau = stale >= int(cfg["patience"])
        if not (at_stage or patient_plateau):
            continue
        relative = late_relative_improvement(curve, int(cfg["late_window"]))
        current_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        policy_values = []
        for checkpoint in state_window:
            model.load_state_dict(checkpoint); model.eval()
            policy_values.append(validation_policy(model))
        model.load_state_dict(current_state); model.eval()
        jsd = mean_policy_jsd(policy_values)
        gates_pass = (
            relative <= float(cfg["relative_validation_improvement_threshold"])
            and jsd <= float(cfg["mean_validation_policy_jsd_threshold"])
        )
        if gates_pass:
            stopped = epoch
            stage_reached = next(cap for cap in caps if epoch <= cap)
            stop_reason = "staged_plateau" if at_stage else "patience_plateau"
            break
    if best_state is None:
        raise RuntimeError("no finite validation checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    if stopped == caps[-1] and not (
        relative <= float(cfg["relative_validation_improvement_threshold"])
        and jsd <= float(cfg["mean_validation_policy_jsd_threshold"])
    ):
        status = "convergence_incomplete"
    else:
        status = "converged_or_plateaued"
    return StagedReceipt(
        best_epoch, stopped, stage_reached, curve, relative, jsd, stale, status,
        stop_reason, updates, time.perf_counter() - started,
    )


def _model_free_components(method: str, train: PolicyData, validation: PolicyData, actions: int,
                           cfg: dict[str, Any], seed: int, device: torch.device):
    execution_method = "soft_spibb" if method == "spibb_style_count_blend_adapter" else method
    keep = np.flatnonzero(train.reward_known)
    if execution_method not in {"behavior_cloning", "decision_transformer"} and len(keep) < 100:
        raise RuntimeError("insufficient_reward_known_transitions")
    extra: dict[str, Any] = {}
    if execution_method == "behavior_cloning":
        model = MLP(train.x.shape[1], actions, cfg["hidden_dim"]).to(device)
        batches = lambda epoch: _loader((train.x, train.action), cfg["batch_size"], seed + epoch)
        loss = lambda m, b: nn.functional.cross_entropy(m(b[0]), b[1])
        probability = lambda m: normalize(_infer(m, validation.x, device))
        objective, fidelity = "behavior_cloning_cross_entropy", "high_fidelity_discrete_reimplementation"
        updates_per_epoch = math.ceil(len(train.x) / cfg["batch_size"])
    elif execution_method in {"discrete_cql", "soft_spibb"}:
        model = MLP(train.x.shape[1], actions, cfg["hidden_dim"]).to(device)
        arrays = (train.x[keep], train.action[keep], train.reward[keep], train.next_x[keep], train.done[keep].astype(np.float32))
        batches = lambda epoch: _loader(arrays, cfg["batch_size"], seed + epoch)
        def loss(m, b):
            q = m(b[0]); chosen = q.gather(1, b[1][:, None]).squeeze(1)
            with torch.no_grad():
                target = b[2] + cfg["discount"] * (1 - b[4]) * m(b[3]).max(1).values
            return nn.functional.smooth_l1_loss(chosen, target) + cfg["cql_alpha"] * (torch.logsumexp(q, 1) - chosen).mean()
        probability = lambda m: frozen_mf._q_probability(execution_method, m, validation, train, actions, cfg, device)
        objective = "conservative_TD" if execution_method == "discrete_cql" else "count_blend_conservative_TD_adapter"
        fidelity = "high_fidelity_discrete_reimplementation" if execution_method == "discrete_cql" else "historical_spibb_style_count_blend_adapter_not_soft_spibb"
        updates_per_epoch = math.ceil(len(keep) / cfg["batch_size"])
    elif execution_method == "discrete_bcq":
        model = BCQ(train.x.shape[1], actions, cfg["hidden_dim"]).to(device)
        arrays = (train.x[keep], train.action[keep], train.reward[keep], train.next_x[keep], train.done[keep].astype(np.float32))
        batches = lambda epoch: _loader(arrays, cfg["batch_size"], seed + epoch)
        def loss(m, b):
            q, imitation = m(b[0]); chosen = q.gather(1, b[1][:, None]).squeeze(1)
            with torch.no_grad():
                next_q, next_imitation = m(b[3]); soft = torch.softmax(next_imitation, 1)
                eligible = soft >= cfg["bcq_behavior_threshold"] * soft.max(1, keepdim=True).values
                target = b[2] + cfg["discount"] * (1 - b[4]) * next_q.masked_fill(~eligible, -1e9).max(1).values
            return nn.functional.smooth_l1_loss(chosen, target) + nn.functional.cross_entropy(imitation, b[1])
        def probability(m):
            q = _infer(lambda x: m(x)[0], validation.x, device)
            imitation = normalize(_infer(lambda x: m(x)[1], validation.x, device))
            eligible = (imitation >= cfg["bcq_behavior_threshold"] * imitation.max(1, keepdims=True)) & validation.support
            return one_hot(np.argmax(np.where(eligible, q, -np.inf), 1), actions)
        objective, fidelity = "BCQ_TD_plus_imitation", "high_fidelity_discrete_reimplementation"
        updates_per_epoch = math.ceil(len(keep) / cfg["batch_size"])
    elif execution_method == "discrete_iql":
        model = IQL(train.x.shape[1], actions, cfg["hidden_dim"]).to(device)
        arrays = (train.x[keep], train.action[keep], train.reward[keep], train.next_x[keep], train.done[keep].astype(np.float32))
        batches = lambda epoch: _loader(arrays, cfg["batch_size"], seed + epoch)
        expectile, temperature, maximum = cfg["iql_expectile"], cfg["iql_temperature"], cfg["iql_max_weight"]
        def loss(m, b):
            q = m.q(b[0]); value = m.v(b[0]).squeeze(1); chosen = q.gather(1, b[1][:, None]).squeeze(1)
            with torch.no_grad():
                target = b[2] + cfg["discount"] * (1 - b[4]) * m.v(b[3]).squeeze(1)
            advantage = chosen.detach() - value
            q_loss = nn.functional.smooth_l1_loss(chosen, target)
            value_loss = (torch.where(advantage > 0, expectile, 1 - expectile) * advantage.square()).mean()
            weight = torch.exp(temperature * advantage.detach()).clamp(max=maximum)
            actor_loss = (weight * nn.functional.cross_entropy(m.policy(b[0]), b[1], reduction="none")).mean()
            return q_loss + value_loss + actor_loss
        probability = lambda m: normalize(_infer(m.policy, validation.x, device))
        objective, fidelity = "discrete_IQL_expectile_value_and_advantage_weighted_policy", "high_fidelity_discrete_reimplementation"
        updates_per_epoch = math.ceil(len(keep) / cfg["batch_size"])
    elif execution_method == "decision_transformer":
        context = int(cfg["decision_transformer_context"])
        tx, tr, tk = _sequence_arrays(train, context, cfg["discount"])
        vx, _, _ = _sequence_arrays(validation, context, cfg["discount"])
        initial = []
        for episode in np.unique(train.episode):
            local = np.flatnonzero(train.episode == episode); local = local[np.argsort(train.step[local])]
            running = 0.0
            for index in local[::-1]:
                running = float(train.reward[index]) + cfg["discount"] * running
            initial.append(running)
        target = float(np.quantile(initial, cfg["decision_transformer_target_return_quantile"]))
        vr = np.full((len(validation.x), context), target, dtype=np.float32)
        vk = np.ones((len(validation.x), context), dtype=np.float32)
        model = DecisionTransformer(train.x.shape[1], actions, cfg["hidden_dim"], context).to(device)
        batches = lambda epoch: _loader((tx, tr, tk, train.action), cfg["sequence_batch_size"], seed + epoch)
        loss = lambda m, b: nn.functional.cross_entropy(m(b[0], b[1], b[2])[:, -1], b[3])
        def probability(m):
            logits = []
            m.eval()
            with torch.no_grad():
                for start in range(0, len(vx), 2048):
                    logits.append(m(torch.from_numpy(vx[start:start + 2048]).to(device), torch.from_numpy(vr[start:start + 2048]).to(device), torch.from_numpy(vk[start:start + 2048]).to(device))[:, -1].cpu().numpy())
            return normalize(np.concatenate(logits))
        objective, fidelity = "causal_return_conditioned_action_cross_entropy", "high_fidelity_discrete_adapter"
        updates_per_epoch = math.ceil(len(train.x) / cfg["sequence_batch_size"])
        extra = {"target_return": target, "target_return_quantile": cfg["decision_transformer_target_return_quantile"]}
    else:
        raise RuntimeError(f"unknown frozen model-free method: {method}")
    return execution_method, model, batches, loss, probability, objective, fidelity, updates_per_epoch, extra


def train_model_free_staged(method: str, train: PolicyData, validation: PolicyData, actions: int,
                            cfg: dict[str, Any], seed: int, device: torch.device):
    torch.manual_seed(seed); np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.set_device(device.index or 0)
        torch.cuda.reset_peak_memory_stats()
    execution_method, model, batches, loss, probability, objective, fidelity, updates_per_epoch, extra = _model_free_components(
        method, train, validation, actions, cfg, seed, device,
    )
    score = lambda m: frozen_mf.observed_action_nll(probability(m), validation.action, cfg["probability_nll_floor_for_metric_only"])
    receipt = staged_fit(model, batches, loss, score, probability, cfg, device)
    output = probability(model)
    metadata = {
        "selected_epoch": receipt.selected_epoch, "epochs_run": receipt.stopped_epoch,
        "stage_reached": receipt.stage_reached, "validation_curve": ";".join(f"{v:.12g}" for v in receipt.validation_curve),
        "selected_validation_metric": float(min(receipt.validation_curve)),
        "validation_metric": "development_validation_observed_action_negative_log_likelihood",
        "late_window_relative_improvement": receipt.late_relative_improvement,
        "mean_validation_policy_jsd": receipt.mean_validation_policy_jsd,
        "stale_epochs": receipt.stale_epochs, "convergence_status": receipt.convergence_status,
        "stop_reason": receipt.stop_reason, "gradient_updates": receipt.gradient_updates,
        "optimizer_updates_per_epoch": updates_per_epoch, "runtime_seconds": receipt.runtime_seconds,
        "parameter_count": sum(v.numel() for v in model.parameters()),
        "peak_memory_mb": torch.cuda.max_memory_allocated(device.index or 0) / 2**20 if device.type == "cuda" else 0.0,
        "model_state_sha256": model_digest(model), "device": str(device),
        "published_objective_adapter": objective, "fidelity": fidelity,
        "execution_method": execution_method, **extra,
    }
    return output, metadata, model


def fit_world_model_staged(method: str, train: Any, validation: Any, seed: int,
                           cfg: dict[str, Any], device: torch.device,
                           validation_policy: Callable[[nn.Module], np.ndarray]):
    torch.manual_seed(seed); np.random.seed(seed)
    features = int(train.states.shape[-1]); actions_n = int(train.behavior_probability.shape[-1])
    model = frozen_wm.make_model(method, features, actions_n, cfg).to(device)
    train_actions = np.eye(actions_n, dtype=np.float32)[train.actions]
    dataset = TensorDataset(
        torch.from_numpy(train.observed), torch.from_numpy(train.masks.astype(np.float32)),
        torch.from_numpy(train.deltas), torch.from_numpy(train_actions), torch.from_numpy(train.next_states),
    )
    def batches(epoch: int):
        return DataLoader(dataset, batch_size=int(cfg["batch_size"]), shuffle=True,
                          generator=torch.Generator().manual_seed(seed + epoch), pin_memory=device.type == "cuda")
    def loss(m: nn.Module, batch: tuple[torch.Tensor, ...]):
        values, masks, deltas, action, target = batch
        mean, log_scale, auxiliary = frozen_wm._forward(m, method, values, masks, deltas, action)
        if log_scale is None:
            return torch.square(target - mean).mean()
        scale = torch.exp(log_scale)
        return (log_scale + 0.5 * torch.square((target - mean) / scale)).mean() + 0.01 * auxiliary
    def score(m: nn.Module) -> float:
        provisional = frozen_wm.Fit(method, seed, (m,), 0, 0, 0.0, "")
        prediction, _ = frozen_wm.native_predictions(provisional, validation, device)
        return float(np.sqrt(np.mean(np.square(prediction - validation.next_states))))
    receipt = staged_fit(model, batches, loss, score, validation_policy, cfg, device)
    fit = frozen_wm.Fit(method, seed, (model,), receipt.selected_epoch, receipt.stopped_epoch, receipt.runtime_seconds, "")
    fit.fingerprint = frozen_wm._fingerprint(fit.models)
    metadata = {
        "selected_epoch": receipt.selected_epoch, "epochs_run": receipt.stopped_epoch,
        "stage_reached": receipt.stage_reached, "validation_curve": ";".join(f"{v:.12g}" for v in receipt.validation_curve),
        "selected_validation_metric": float(min(receipt.validation_curve)),
        "validation_metric": "development_validation_native_mean_RMSE",
        "late_window_relative_improvement": receipt.late_relative_improvement,
        "mean_validation_policy_jsd": receipt.mean_validation_policy_jsd,
        "stale_epochs": receipt.stale_epochs, "convergence_status": receipt.convergence_status,
        "stop_reason": receipt.stop_reason, "gradient_updates": receipt.gradient_updates,
        "runtime_seconds": receipt.runtime_seconds, "parameter_count": sum(v.numel() for v in model.parameters()),
        "peak_memory_mb": torch.cuda.max_memory_allocated(device.index or 0) / 2**20 if device.type == "cuda" else 0.0,
        "model_state_sha256": fit.fingerprint, "device": str(device),
    }
    return fit, metadata


def six_epoch_model_free(method: str, train: PolicyData, validation: PolicyData, actions: int,
                         cfg: dict[str, Any], seed: int, device: torch.device):
    six = dict(cfg)
    six.update({"maximum_epochs": 6, "minimum_epochs": 3, "patience": 2,
                "convergence_window": 3, "convergence_relative_improvement_threshold": 0.005})
    execution_method = "soft_spibb" if method == "spibb_style_count_blend_adapter" else method
    return frozen_mf.train_method(execution_method, train, validation, actions, six, seed, device)


def state_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode()); digest.update(value.detach().cpu().contiguous().numpy().view(np.uint8))
    return digest.hexdigest()
