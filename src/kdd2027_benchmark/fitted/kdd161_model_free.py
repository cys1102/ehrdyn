from __future__ import annotations

import hashlib
import math
import time
from typing import Any, Callable

import numpy as np
import torch
from torch import nn

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


ProbabilityFunction = Callable[[nn.Module], np.ndarray]


def observed_action_nll(probability: np.ndarray, action: np.ndarray, floor: float) -> float:
    if not validate_probability(probability):
        raise RuntimeError("checkpoint policy probability is invalid")
    selected = probability[np.arange(len(action)), action.astype(int)]
    return float(-np.log(np.maximum(selected, floor)).mean())


def _fit(
    model: nn.Module,
    batches: Callable[[int], Any],
    train_loss: Callable[[nn.Module, tuple[torch.Tensor, ...]], torch.Tensor],
    validation_probability: ProbabilityFunction,
    validation_action: np.ndarray,
    cfg: dict[str, Any],
    device: torch.device,
) -> tuple[int, int, list[float], int, float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    best: dict[str, torch.Tensor] | None = None
    best_value = math.inf
    best_epoch = 0
    stale = 0
    curve: list[float] = []
    updates = 0
    started = time.perf_counter()
    for epoch in range(1, int(cfg["maximum_epochs"]) + 1):
        model.train()
        for batch in batches(epoch):
            batch = tuple(value.to(device, non_blocking=True) for value in batch)
            optimizer.zero_grad(set_to_none=True)
            loss = train_loss(model, batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip_norm"])
            optimizer.step()
            updates += 1
        model.eval()
        value = observed_action_nll(validation_probability(model), validation_action, cfg["probability_nll_floor_for_metric_only"])
        curve.append(value)
        if value < best_value - cfg["minimum_metric_improvement"]:
            best_value = value
            best_epoch = epoch
            best = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if epoch >= cfg["minimum_epochs"] and stale >= cfg["patience"]:
            break
    if best is None:
        raise RuntimeError("no finite checkpoint metric")
    model.load_state_dict(best)
    return epoch, best_epoch, curve, updates, time.perf_counter() - started


def _metadata(
    model: nn.Module,
    epoch: int,
    best_epoch: int,
    curve: list[float],
    updates: int,
    runtime: float,
    updates_per_epoch: int,
    cfg: dict[str, Any],
    device: torch.device,
    objective: str,
    fidelity: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    window = min(int(cfg["convergence_window"]), len(curve))
    late = np.asarray(curve[-window:], dtype=float)
    slope = float(np.polyfit(np.arange(window), late, 1)[0]) if window > 1 else math.nan
    relative = float((late[0] - late[-1]) / max(abs(late[0]), 1e-12)) if window > 1 else 0.0
    status = "convergence_incomplete" if epoch == cfg["maximum_epochs"] and relative > cfg["convergence_relative_improvement_threshold"] else "converged_or_plateaued"
    result = {
        "selected_epoch": best_epoch,
        "maximum_epoch": cfg["maximum_epochs"],
        "epochs_run": epoch,
        "validation_metric": "development_validation_observed_action_negative_log_likelihood",
        "selected_validation_metric": float(min(curve)),
        "validation_curve": ";".join(f"{value:.12g}" for value in curve),
        "late_epoch_validation_slope": slope,
        "late_window_relative_improvement": relative,
        "convergence_status": status,
        "optimizer_updates_per_epoch": updates_per_epoch,
        "gradient_updates": updates,
        "parameter_count": sum(value.numel() for value in model.parameters()),
        "runtime_seconds": runtime,
        "peak_memory_mb": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0,
        "model_state_sha256": model_digest(model),
        "device": str(device),
        "published_objective_adapter": objective,
        "fidelity": fidelity,
    }
    result.update(extra or {})
    return result


def _q_probability(method: str, model: nn.Module, validation: PolicyData, train: PolicyData, actions: int, cfg: dict[str, Any], device: torch.device) -> np.ndarray:
    q = _infer(model, validation.x, device)
    if method == "discrete_cql":
        return normalize(q, validation.support)
    candidate = normalize(q / cfg["soft_spibb_temperature"], validation.support)
    count = np.bincount(train.action, minlength=actions)
    beta = count / (count + cfg["soft_spibb_min_count"])
    output = validation.behavior * (1 - beta) + candidate * beta
    output = np.where(validation.support, output, 0)
    return output / output.sum(1, keepdims=True)


def train_method(method: str, train: PolicyData, validation: PolicyData, actions: int, cfg: dict[str, Any], seed: int, device: torch.device):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    keep = np.flatnonzero(train.reward_known)
    if method != "behavior_cloning" and method != "decision_transformer" and len(keep) < 100:
        raise RuntimeError("insufficient_reward_known_transitions")

    extra: dict[str, Any] = {}
    if method == "behavior_cloning":
        model = MLP(train.x.shape[1], actions, cfg["hidden_dim"]).to(device)
        batches = lambda epoch: _loader((train.x, train.action), cfg["batch_size"], seed + epoch)
        loss = lambda m, b: nn.functional.cross_entropy(m(b[0]), b[1])
        probability = lambda m: normalize(_infer(m, validation.x, device))
        objective, fidelity = "behavior_cloning_cross_entropy", "high_fidelity_discrete_reimplementation"
        updates_per_epoch = math.ceil(len(train.x) / cfg["batch_size"])
    elif method in {"discrete_cql", "soft_spibb"}:
        model = MLP(train.x.shape[1], actions, cfg["hidden_dim"]).to(device)
        arrays = (train.x[keep], train.action[keep], train.reward[keep], train.next_x[keep], train.done[keep].astype(np.float32))
        batches = lambda epoch: _loader(arrays, cfg["batch_size"], seed + epoch)
        def loss(m, b):
            q = m(b[0]); chosen = q.gather(1, b[1][:, None]).squeeze(1)
            with torch.no_grad():
                target = b[2] + cfg["discount"] * (1 - b[4]) * m(b[3]).max(1).values
            return nn.functional.smooth_l1_loss(chosen, target) + cfg["cql_alpha"] * (torch.logsumexp(q, 1) - chosen).mean()
        probability = lambda m: _q_probability(method, m, validation, train, actions, cfg, device)
        objective = "conservative_TD" if method == "discrete_cql" else "Soft_SPIBB_count_regularized_TD"
        fidelity = "high_fidelity_discrete_reimplementation" if method == "discrete_cql" else "conceptual_offline_adapter"
        updates_per_epoch = math.ceil(len(keep) / cfg["batch_size"])
    elif method == "discrete_bcq":
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
    elif method == "discrete_iql":
        model = IQL(train.x.shape[1], actions, cfg["hidden_dim"]).to(device)
        arrays = (train.x[keep], train.action[keep], train.reward[keep], train.next_x[keep], train.done[keep].astype(np.float32))
        batches = lambda epoch: _loader(arrays, cfg["batch_size"], seed + epoch)
        expectile, temperature, maximum = cfg["iql_expectile"], cfg["iql_temperature"], cfg["iql_max_weight"]
        def loss(m, b):
            q = m.q(b[0]); value = m.v(b[0]).squeeze(1); chosen = q.gather(1, b[1][:, None]).squeeze(1)
            with torch.no_grad():
                target = b[2] + cfg["discount"] * (1 - b[4]) * m.v(b[3]).squeeze(1)
            q_loss = nn.functional.smooth_l1_loss(chosen, target)
            advantage = chosen.detach() - value
            value_loss = (torch.where(advantage > 0, expectile, 1 - expectile) * advantage.square()).mean()
            weight = torch.exp(temperature * advantage.detach()).clamp(max=maximum)
            actor_loss = (weight * nn.functional.cross_entropy(m.policy(b[0]), b[1], reduction="none")).mean()
            return q_loss + value_loss + actor_loss
        probability = lambda m: normalize(_infer(m.policy, validation.x, device))
        objective, fidelity = "discrete_IQL_expectile_value_and_advantage_weighted_policy", "high_fidelity_discrete_reimplementation"
        updates_per_epoch = math.ceil(len(keep) / cfg["batch_size"])
    elif method == "decision_transformer":
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
        target_bytes = np.asarray([target, cfg["decision_transformer_target_return_quantile"]], dtype=np.float64).tobytes()
        extra = {"target_return": target, "target_return_quantile": cfg["decision_transformer_target_return_quantile"], "target_return_contract_sha256": hashlib.sha256(target_bytes).hexdigest(), "decoding": "native_softmax_no_epsilon_smoothing"}
    else:
        raise RuntimeError(f"unknown frozen method: {method}")

    epoch, best_epoch, curve, updates, runtime = _fit(model, batches, loss, probability, validation.action, cfg, device)
    output = probability(model)
    metadata = _metadata(model, epoch, best_epoch, curve, updates, runtime, updates_per_epoch, cfg, device, objective, fidelity, extra)
    return output, metadata, model
