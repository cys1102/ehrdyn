from __future__ import annotations

import copy
import hashlib
import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from scipy.special import ndtr
from torch.utils.data import DataLoader, TensorDataset

from .kdd069_rssm_models import DreamerV3CategoricalRSSM
from .kdd069_sequence_models import CausalTransformerTransition
from .paper_models import DeterministicGRUDPoint, SingleGaussianGRUD


BASE_MODELS = (
    "deterministic_grud_point",
    "single_gaussian_grud",
    "causal_transformer",
    "categorical_rssm",
)
PROBABILISTIC_MODELS = {
    "single_gaussian_grud", "causal_transformer", "categorical_rssm", "matched_gaussian_ensemble"
}


@dataclass(slots=True)
class Fit:
    method: str
    seed: int | str
    models: tuple[torch.nn.Module, ...]
    selected_epoch: int
    stopped_epoch: int
    training_seconds: float
    fingerprint: str


def make_model(method: str, features: int, actions: int, cfg: dict[str, Any]) -> torch.nn.Module:
    hidden = int(cfg["hidden_dim"])
    if method == "deterministic_grud_point":
        return DeterministicGRUDPoint(features, actions, hidden)
    if method == "single_gaussian_grud":
        return SingleGaussianGRUD(features, actions, hidden)
    if method == "causal_transformer":
        return CausalTransformerTransition(features, actions, hidden)
    if method == "categorical_rssm":
        latent = int(cfg["latent_dim"])
        return DreamerV3CategoricalRSSM(features, actions, hidden, max(2, latent // 4), 4)
    raise ValueError(method)


def _forward(model: torch.nn.Module, method: str, values: torch.Tensor, masks: torch.Tensor,
             deltas: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    output = model(values, masks, deltas, actions)
    if method == "deterministic_grud_point":
        return output, None, torch.zeros((), device=values.device)
    if method == "single_gaussian_grud":
        mean, log_scale = output
        return mean, log_scale, torch.zeros((), device=values.device)
    return output.mean, output.log_scale, output.auxiliary_loss


def _fingerprint(models: tuple[torch.nn.Module, ...]) -> str:
    digest = hashlib.sha256()
    for index, model in enumerate(models):
        digest.update(str(index).encode())
        for name, value in sorted(model.state_dict().items()):
            digest.update(name.encode()); digest.update(value.detach().cpu().contiguous().numpy().view(np.uint8))
    return digest.hexdigest()


def native_predictions(fit: Fit, data: Any, device: torch.device, batch: int = 256) -> tuple[np.ndarray, np.ndarray | None]:
    actions = np.eye(int(data.behavior_probability.shape[-1]), dtype=np.float32)[data.actions]
    member_means: list[np.ndarray] = []
    member_variances: list[np.ndarray] = []
    for model in fit.models:
        means: list[np.ndarray] = []; variances: list[np.ndarray] = []
        model.eval()
        with torch.inference_mode():
            for start in range(0, len(data.states), batch):
                stop = min(start + batch, len(data.states))
                mean, log_scale, _ = _forward(
                    model,
                    "single_gaussian_grud" if fit.method == "matched_gaussian_ensemble" else fit.method,
                    torch.from_numpy(data.observed[start:stop]).to(device),
                    torch.from_numpy(data.masks[start:stop].astype(np.float32)).to(device),
                    torch.from_numpy(data.deltas[start:stop]).to(device),
                    torch.from_numpy(actions[start:stop]).to(device),
                )
                means.append(mean.cpu().numpy())
                if log_scale is not None:
                    variances.append(torch.exp(2.0 * log_scale).cpu().numpy())
        member_means.append(np.concatenate(means))
        if variances:
            member_variances.append(np.concatenate(variances))
    mean = np.mean(member_means, axis=0)
    if not member_variances:
        return mean, None
    variance = np.mean(member_variances, axis=0) + np.var(member_means, axis=0)
    return mean, np.sqrt(np.maximum(variance, 1e-8))


def recursive_predictions(fit: Fit, data: Any, device: torch.device, batch: int = 128) -> tuple[np.ndarray, np.ndarray | None]:
    action_onehot = np.eye(int(data.behavior_probability.shape[-1]), dtype=np.float32)[data.actions]
    member_means: list[np.ndarray] = []; member_variances: list[np.ndarray] = []
    for model in fit.models:
        all_mean: list[np.ndarray] = []; all_var: list[np.ndarray] = []
        model.eval()
        with torch.inference_mode():
            for start in range(0, len(data.states), batch):
                stop = min(start + batch, len(data.states))
                values = torch.from_numpy(data.observed[start:stop, :1]).to(device)
                masks = torch.from_numpy(data.masks[start:stop, :1].astype(np.float32)).to(device)
                deltas = torch.from_numpy(data.deltas[start:stop, :1]).to(device)
                actions = torch.from_numpy(action_onehot[start:stop]).to(device)
                local_mean: list[torch.Tensor] = []; local_var: list[torch.Tensor] = []
                for step in range(actions.shape[1]):
                    mean, log_scale, _ = _forward(
                        model,
                        "single_gaussian_grud" if fit.method == "matched_gaussian_ensemble" else fit.method,
                        values, masks, deltas, actions[:, :step + 1],
                    )
                    current = mean[:, -1]
                    local_mean.append(current)
                    if log_scale is not None:
                        local_var.append(torch.exp(2.0 * log_scale[:, -1]))
                    values = torch.cat([values, current[:, None]], dim=1)
                    masks = torch.cat([masks, torch.zeros_like(current[:, None])], dim=1)
                    deltas = torch.cat([deltas, deltas[:, -1:] + 1.0], dim=1)
                all_mean.append(torch.stack(local_mean, dim=1).cpu().numpy())
                if local_var:
                    all_var.append(torch.stack(local_var, dim=1).cpu().numpy())
        member_means.append(np.concatenate(all_mean))
        if all_var:
            member_variances.append(np.concatenate(all_var))
    mean = np.mean(member_means, axis=0)
    if not member_variances:
        return mean, None
    variance = np.mean(member_variances, axis=0) + np.var(member_means, axis=0)
    return mean, np.sqrt(np.maximum(variance, 1e-8))


def fit_model(method: str, train: Any, validation: Any, seed: int, cfg: dict[str, Any],
              device: torch.device) -> Fit:
    torch.manual_seed(seed); np.random.seed(seed)
    features = int(train.states.shape[-1]); actions_n = int(train.behavior_probability.shape[-1])
    model = make_model(method, features, actions_n, cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
    train_actions = np.eye(actions_n, dtype=np.float32)[train.actions]
    dataset = TensorDataset(
        torch.from_numpy(train.observed), torch.from_numpy(train.masks.astype(np.float32)),
        torch.from_numpy(train.deltas), torch.from_numpy(train_actions), torch.from_numpy(train.next_states),
    )
    loader = DataLoader(dataset, batch_size=int(cfg["batch_size"]), shuffle=True,
                        generator=torch.Generator().manual_seed(seed), pin_memory=device.type == "cuda")
    best_state: dict[str, torch.Tensor] | None = None; best = math.inf; best_epoch = 0; stale = 0
    stopped = int(cfg["max_epochs"]); started = time.perf_counter()
    for epoch in range(1, int(cfg["max_epochs"]) + 1):
        model.train()
        for values, masks, deltas, action, target in loader:
            values, masks, deltas, action, target = (x.to(device, non_blocking=True) for x in (values, masks, deltas, action, target))
            mean, log_scale, auxiliary = _forward(model, method, values, masks, deltas, action)
            if log_scale is None:
                loss = torch.square(target - mean).mean()
            else:
                scale = torch.exp(log_scale)
                loss = (log_scale + 0.5 * torch.square((target - mean) / scale)).mean() + 0.01 * auxiliary
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["gradient_clip_norm"])); optimizer.step()
        provisional = Fit(method, seed, (model,), epoch, epoch, 0.0, "")
        prediction, _ = native_predictions(provisional, validation, device)
        score = float(np.sqrt(np.mean(np.square(prediction - validation.next_states))))
        if score < best - 1e-12:
            best = score; best_epoch = epoch; stale = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if epoch >= int(cfg["minimum_epochs"]) and stale >= int(cfg["early_stopping_patience"]):
            stopped = epoch; break
    if best_state is None:
        raise RuntimeError("no finite separate-validation checkpoint")
    model.load_state_dict(best_state); model.eval()
    fit = Fit(method, seed, (model,), best_epoch, stopped, time.perf_counter() - started, "")
    fit.fingerprint = _fingerprint(fit.models)
    return fit


def ensemble_fit(members: list[Fit]) -> Fit:
    if len(members) != 3 or any(member.method != "single_gaussian_grud" for member in members):
        raise ValueError("matched ensemble requires exactly three Single Gaussian members")
    fit = Fit("matched_gaussian_ensemble", "3408;3411;3414", tuple(member.models[0] for member in members),
              max(member.selected_epoch for member in members), max(member.stopped_epoch for member in members),
              sum(member.training_seconds for member in members), "")
    fit.fingerprint = _fingerprint(fit.models)
    return fit


def gaussian_crps(target: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> float:
    scale = np.maximum(scale, 1e-6); z = (target - mean) / scale
    phi = np.exp(-0.5 * np.square(z)) / math.sqrt(2.0 * math.pi)
    return float(np.mean(scale * (z * (2.0 * ndtr(z) - 1.0) + 2.0 * phi - 1.0 / math.sqrt(math.pi))))


def metrics(fit: Fit, validation: Any, device: torch.device) -> dict[str, Any]:
    one, one_scale = native_predictions(fit, validation, device)
    recursive, recursive_scale = recursive_predictions(fit, validation, device)
    target = validation.next_states; final_error = recursive[:, -1] - target[:, -1]
    row: dict[str, Any] = {
        "one_step_rmse": float(np.sqrt(np.mean(np.square(one - target)))),
        "final_horizon_recursive_rmse": float(np.sqrt(np.mean(np.square(final_error)))),
        "final_horizon_crps": np.nan, "mace": np.nan, "risk_coverage_area": np.nan,
        "probabilistic": recursive_scale is not None,
    }
    if recursive_scale is None:
        return row
    final_scale = recursive_scale[:, -1]
    row["final_horizon_crps"] = gaussian_crps(target[:, -1], recursive[:, -1], final_scale)
    levels = ((0.50, 0.67448975), (0.80, 1.28155157), (0.90, 1.64485363), (0.95, 1.95996398))
    coverages = [float(np.mean(np.abs(final_error) <= z * final_scale)) for _, z in levels]
    row["mace"] = float(np.mean([abs(observed - nominal) for observed, (nominal, _) in zip(coverages, levels)]))
    uncertainty = np.mean(final_scale, axis=1); error = np.mean(np.square(final_error), axis=1)
    order = np.argsort(uncertainty); fractions = np.linspace(0.1, 1.0, 10)
    risks = [float(np.sqrt(np.mean(error[order[:max(1, int(math.ceil(len(order) * q)))]]))) for q in fractions]
    row["risk_coverage_area"] = float(np.trapezoid(risks, fractions))
    return row


class _ProbeData:
    pass


def transition_tables(fit: Fit, env: Any, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate every finite state/action pair through the learned sequence interface."""
    states = np.repeat(np.arange(env.n_states), env.n_actions)
    actions = np.tile(np.arange(env.n_actions), env.n_states)
    probe = _ProbeData()
    eye = np.eye(env.n_states, dtype=np.float32)
    probe.states = np.repeat(eye[states, None, :], env.horizon, axis=1)
    probe.observed = probe.states.copy()
    probe.masks = np.ones_like(probe.states, dtype=bool)
    probe.deltas = np.ones_like(probe.states, dtype=np.float32)
    probe.actions = np.repeat(actions[:, None], env.horizon, axis=1).astype(np.int16)
    probe.behavior_probability = np.zeros((len(states), env.horizon, env.n_actions), dtype=np.float32)
    mean, scale = native_predictions(fit, probe, device)
    final_mean = mean[:, -1].reshape(env.n_states, env.n_actions, env.n_states)
    next_state = np.argmax(final_mean, axis=-1)
    if scale is None:
        uncertainty = np.zeros((env.n_states, env.n_actions), dtype=np.float64)
    else:
        uncertainty = np.mean(scale[:, -1], axis=-1).reshape(env.n_states, env.n_actions)
    return next_state, uncertainty


def select_models(metric_rows: list[dict[str, Any]], rules: list[str], tie_order: list[str]) -> list[dict[str, Any]]:
    order = {name: index for index, name in enumerate(tie_order)}
    selected: list[dict[str, Any]] = []
    keys = sorted({(row["task"], row["mechanism"], row["training_seed"]) for row in metric_rows})
    for task, mechanism, seed in keys:
        local = [row for row in metric_rows if row["task"] == task and row["mechanism"] == mechanism and row["training_seed"] == seed]
        for rule in rules:
            candidates = [row for row in local if np.isfinite(float(row[rule]))]
            if not candidates:
                selected.append({"task": task, "mechanism": mechanism, "training_seed": seed,
                                 "selection_rule": rule, "selected_model": "none", "status": "no_valid_metric"})
                continue
            candidates.sort(key=lambda row: (float(row[rule]), order[row["model"]]))
            winner = candidates[0]
            selected.append({"task": task, "mechanism": mechanism, "training_seed": seed,
                             "selection_rule": rule, "selected_model": winner["model"],
                             "selected_metric_value": winner[rule], "tie_break_order": ";".join(tie_order),
                             "policy_return_used_for_selection": False, "status": "selected"})
    return selected


def decision_from_gates(hard_failure: bool, associations: list[float]) -> str:
    if hard_failure:
        return "stop_adaptivity_exact_truth_or_selection_contract_failure"
    finite = [value for value in associations if np.isfinite(value)]
    if not finite or all(value <= 0.0 for value in finite):
        return "complete_negative_no_metric_predicts_policy_quality"
    return "complete_metric_to_policy_selection_benchmark"
