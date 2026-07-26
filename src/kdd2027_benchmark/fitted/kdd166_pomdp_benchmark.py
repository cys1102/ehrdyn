from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
from scipy.special import ndtr
from sklearn.ensemble import HistGradientBoostingRegressor

from . import kdd164_metric_selection as wm
from . import kdd161_model_free as mf
from .kdd155v3_model_free import PolicyData, normalize, one_hot


Policy = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int], np.ndarray]


@dataclass(slots=True)
class SyntheticData:
    observed: np.ndarray
    masks: np.ndarray
    deltas: np.ndarray
    actions: np.ndarray
    next_states: np.ndarray
    behavior_probability: np.ndarray
    rewards: np.ndarray
    done: np.ndarray
    valid: np.ndarray
    subtypes: np.ndarray

    @property
    def states(self) -> np.ndarray:
        return self.observed


@dataclass(slots=True)
class HGBFit:
    method: str
    seed: int
    models: tuple[HistGradientBoostingRegressor, ...]
    fingerprint: str
    training_seconds: float


def _digest_arrays(*values: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(value)
        digest.update(str(array.shape).encode()); digest.update(array.dtype.str.encode()); digest.update(array.view(np.uint8))
    return digest.hexdigest()


def collect_dataset(environment: Any, episodes: int, stream_seed: int, behavior_family: str) -> SyntheticData:
    """Generate a logged dataset while exposing only observation history to learned methods."""
    env = environment; streams = env._streams(episodes, stream_seed)
    states = env._draw(np.broadcast_to(env.initial_state_probability, (episodes, env.states)), streams["state_u"])
    subtypes = env._draw(np.broadcast_to(env.subtype_prevalence, (episodes, env.subtypes)), streams["subtype_u"])
    marginal = np.asarray(env.contract.target_action_frequency)[env.supported]; marginal /= marginal.sum()
    pending = env.supported[env._draw(np.broadcast_to(marginal, (episodes, len(env.supported))), streams["prior_u"])]
    previous_mask = np.ones((episodes, env.contract.feature_dim), dtype=bool)
    recency = np.zeros((episodes, env.contract.feature_dim), dtype=np.float32); alive = np.ones(episodes, dtype=bool)
    f, h, k = env.contract.feature_dim, env.horizon, env.contract.action_count
    observed = np.zeros((episodes, h, f), np.float32); masks = np.zeros((episodes, h, f), bool)
    deltas = np.zeros((episodes, h, f), np.float32); actions = np.zeros((episodes, h), np.int16)
    next_observed = np.zeros((episodes, h, f), np.float32); probabilities = np.zeros((episodes, h, k), np.float32)
    rewards = np.zeros((episodes, h), np.float32); done = np.zeros((episodes, h), bool); valid = np.zeros((episodes, h), bool)
    subtype_rows = np.repeat(subtypes[:, None], h, axis=1).astype(np.int16)
    for t in range(h):
        obs, mask, current_delta = env._emit(states, subtypes, previous_mask, recency, pending, streams["noise"][:, t], streams["mask_u"][:, t])
        probability = np.zeros((episodes, k), float)
        if behavior_family == "ehr_matched":
            bins = env.behavior.context_bin(obs, mask, current_delta, pending, t)
            for index in range(episodes): probability[index] = env.behavior.distribution(int(pending[index]), int(bins[index]), k)
        elif behavior_family == "smart_like_exploratory":
            probability[:, env.supported] = 1.0 / len(env.supported)
        elif behavior_family == "concentrated_behavior":
            probability[:, env.supported] = (1.0 - float(env.generator["concentrated_previous_action_mass"])) * marginal
            probability[np.arange(episodes), pending] += float(env.generator["concentrated_previous_action_mass"])
        else:
            raise ValueError(behavior_family)
        action = env._draw(probability, streams["policy_u"][:, t]); action[~alive] = env.supported[0]
        transition = env.transition[states, subtypes, pending, action]
        next_state = env._draw(transition, streams["transition_u"][:, t])
        immediate = np.zeros(episodes)
        for table in env.reward_components.values(): immediate += table[states, subtypes, pending, action]
        hazard = 1.0 if t == h - 1 else env.contract.termination_hazards[t]
        terminate = alive & (streams["termination_u"][:, t] < hazard)
        terminal = np.zeros(episodes)
        if env.contract.primary_reward_type == "terminal":
            death = streams["outcome_u"][:, t] < env.death_probability(next_state)
            terminal[terminate] = np.where(death[terminate], env.contract.terminal_reward_minimum, env.contract.terminal_reward_maximum)
        next_obs, _, _ = env._emit(next_state, subtypes, mask, current_delta, action, np.zeros_like(streams["noise"][:, t]), np.ones_like(streams["mask_u"][:, t]) * 0.5)
        observed[:, t] = obs; masks[:, t] = mask; deltas[:, t] = current_delta; actions[:, t] = action
        next_observed[:, t] = next_obs; probabilities[:, t] = probability; rewards[:, t] = immediate + terminal
        done[:, t] = terminate; valid[:, t] = alive
        alive &= ~terminate; states = next_state; pending = action; previous_mask = mask; recency = current_delta
    return SyntheticData(observed, masks, deltas, actions, next_observed, probabilities, rewards, done, valid, subtype_rows)


def policy_data(data: SyntheticData) -> PolicyData:
    n, h, f = data.observed.shape; k = data.behavior_probability.shape[-1]
    prior = np.zeros_like(data.actions); prior[:, 1:] = data.actions[:, :-1]
    time_channel = np.broadcast_to(np.arange(h, dtype=np.float32)[None, :, None] / max(h - 1, 1), (n, h, 1))
    state = np.concatenate([data.observed, data.masks.astype(np.float32), data.deltas / max(h, 1), np.eye(k, dtype=np.float32)[prior], time_channel], axis=-1)
    next_state = np.concatenate([data.next_states, data.masks.astype(np.float32), np.minimum(data.deltas + 1, h) / max(h, 1), np.eye(k, dtype=np.float32)[data.actions], time_channel], axis=-1)
    keep = data.valid.reshape(-1); episode = np.repeat(np.arange(n), h); step = np.tile(np.arange(h), n)
    behavior = data.behavior_probability.reshape(-1, k)[keep]
    support = behavior > 0
    return PolicyData(state.reshape(-1, state.shape[-1])[keep].astype(np.float32), data.actions.reshape(-1)[keep].astype(np.int64),
                      data.rewards.reshape(-1)[keep].astype(np.float32), np.ones(int(keep.sum()), bool),
                      next_state.reshape(-1, next_state.shape[-1])[keep].astype(np.float32), data.done.reshape(-1)[keep],
                      support, behavior.astype(np.float32), state[..., 0].reshape(-1)[keep].astype(np.float32),
                      episode[keep].astype(np.int64), step[keep].astype(np.int64))


def fit_hgb(train: SyntheticData, seed: int) -> HGBFit:
    start = time.perf_counter(); keep = train.valid.reshape(-1)
    x = np.concatenate([train.observed, train.masks.astype(np.float32), train.deltas, np.eye(train.behavior_probability.shape[-1])[train.actions]], axis=-1).reshape(-1, train.observed.shape[-1] * 3 + train.behavior_probability.shape[-1])[keep]
    y = train.next_states.reshape(-1, train.next_states.shape[-1])[keep]
    models = tuple(HistGradientBoostingRegressor(max_iter=50, max_depth=4, learning_rate=0.08, random_state=seed).fit(x, y[:, j]) for j in range(y.shape[1]))
    fingerprint = hashlib.sha256(";".join(str(model.get_params()) for model in models).encode()).hexdigest()
    return HGBFit("hgb_residual", seed, models, fingerprint, time.perf_counter() - start)


def hgb_predict(fit: HGBFit, data: SyntheticData) -> np.ndarray:
    n, h, f = data.observed.shape; keep = data.valid.reshape(-1)
    x = np.concatenate([data.observed, data.masks.astype(np.float32), data.deltas, np.eye(data.behavior_probability.shape[-1])[data.actions]], axis=-1).reshape(-1, f * 3 + data.behavior_probability.shape[-1])
    output = np.zeros((n * h, f), np.float32)
    for j, model in enumerate(fit.models): output[:, j] = model.predict(x)
    return output.reshape(n, h, f)


def forecasting_metrics(method: str, prediction: np.ndarray, target: np.ndarray, valid: np.ndarray,
                        scale: np.ndarray | None = None) -> dict[str, Any]:
    mask = np.broadcast_to(valid[..., None], target.shape); error = prediction[mask] - target[mask]
    row: dict[str, Any] = {"one_step_rmse": float(np.sqrt(np.mean(error**2))), "one_step_mae": float(np.mean(np.abs(error))),
                           "final_horizon_recursive_rmse": float(np.sqrt(np.mean((prediction[:, -1] - target[:, -1])**2))),
                           "final_horizon_crps": math.nan, "mace": math.nan, "risk_coverage_area": math.nan,
                           "probabilistic": scale is not None}
    if scale is None: return row
    sigma = np.maximum(scale[mask], 1e-6); z = error / sigma
    row["final_horizon_crps"] = float(np.mean(sigma * (z * (2 * ndtr(z) - 1) + 2 * np.exp(-0.5*z*z)/math.sqrt(2*math.pi) - 1/math.sqrt(math.pi))))
    levels = ((.5,.67448975),(.8,1.28155157),(.9,1.64485363),(.95,1.95996398))
    cover = [float(np.mean(np.abs(error) <= q*sigma)) for _, q in levels]
    row["mace"] = float(np.mean([abs(a-b[0]) for a,b in zip(cover,levels)]))
    uncertainty = np.mean(scale, axis=(1,2)); risk = np.mean((prediction-target)**2, axis=(1,2)); order=np.argsort(uncertainty)
    fractions=np.linspace(.1,1,10); values=[math.sqrt(float(np.mean(risk[order[:max(1,math.ceil(len(order)*q))]]))) for q in fractions]
    row["risk_coverage_area"] = float(np.trapezoid(values, fractions))
    return row


def model_free_probability(method: str, model: torch.nn.Module, data: PolicyData, train: PolicyData,
                           actions: int, cfg: dict[str, Any], device: torch.device, metadata: dict[str, Any]) -> np.ndarray:
    from .kdd155v3_model_free import _infer
    if method == "behavior_cloning": return normalize(_infer(model, data.x, device))
    if method == "discrete_bcq":
        q = _infer(lambda x: model(x)[0], data.x, device); imitation = normalize(_infer(lambda x: model(x)[1], data.x, device))
        eligible = (imitation >= cfg["bcq_behavior_threshold"] * imitation.max(1, keepdims=True)) & data.support
        return one_hot(np.argmax(np.where(eligible, q, -np.inf), axis=1), actions)
    if method in {"discrete_cql", "soft_spibb"}:
        return mf._q_probability(method, model, data, train, actions, cfg, device)
    if method == "discrete_iql": return normalize(_infer(model.policy, data.x, device))
    if method == "decision_transformer":
        context = int(cfg["decision_transformer_context"]); x=np.zeros((len(data.x),context,data.x.shape[1]),np.float32); x[:,-1]=data.x
        rtg=np.full((len(data.x),context),float(metadata["target_return"]),np.float32); known=np.zeros_like(rtg); output=[]
        model.eval()
        with torch.inference_mode():
            for start in range(0,len(x),2048): output.append(model(torch.from_numpy(x[start:start+2048]).to(device),torch.from_numpy(rtg[start:start+2048]).to(device),torch.from_numpy(known[start:start+2048]).to(device))[:,-1].cpu().numpy())
        return normalize(np.concatenate(output))
    raise ValueError(method)


def probability_policy_from_model(method: str, model: torch.nn.Module, train: PolicyData, actions: int,
                                  cfg: dict[str, Any], device: torch.device, metadata: dict[str, Any]) -> Policy:
    def policy(obs: np.ndarray, mask: np.ndarray, recency: np.ndarray, previous: np.ndarray, time_index: int) -> np.ndarray:
        n=len(obs); x=np.concatenate([obs,mask.astype(np.float32),recency/11.0,np.eye(actions,dtype=np.float32)[previous],np.full((n,1),time_index/10,np.float32)],axis=1)
        behavior=np.zeros((n,actions),np.float32); behavior[:,np.asarray(train.support).any(axis=0)]=1; behavior/=behavior.sum(1,keepdims=True)
        data=PolicyData(x,np.zeros(n,np.int64),np.zeros(n,np.float32),np.ones(n,bool),x,np.zeros(n,bool),np.broadcast_to(np.asarray(train.support).any(axis=0),(n,actions)).copy(),behavior,np.zeros(n,np.float32),np.arange(n),np.zeros(n,np.int64))
        output=model_free_probability(method,model,data,train,actions,cfg,device,metadata)
        frozen_support=np.asarray(train.support).any(axis=0)
        output=np.where(frozen_support[None],output,0.0)
        denominator=output.sum(axis=1,keepdims=True)
        if np.any(denominator<=0):
            output[:,frozen_support]=1.0/float(frozen_support.sum());denominator=output.sum(axis=1,keepdims=True)
        return output/denominator
    return policy


def simulate_policy(environment: Any, policy: Policy, episodes: int, stream_seed: int) -> dict[str, Any]:
    env=environment; streams=env._streams(episodes,stream_seed); states=env._draw(np.broadcast_to(env.initial_state_probability,(episodes,env.states)),streams["state_u"]); subtypes=env._draw(np.broadcast_to(env.subtype_prevalence,(episodes,env.subtypes)),streams["subtype_u"])
    marginal=np.asarray(env.contract.target_action_frequency)[env.supported];marginal/=marginal.sum();pending=env.supported[env._draw(np.broadcast_to(marginal,(episodes,len(env.supported))),streams["prior_u"])]
    prior_mask=np.ones((episodes,env.contract.feature_dim),bool);recency=np.zeros_like(prior_mask,dtype=np.float32);alive=np.ones(episodes,bool);returns=np.zeros(episodes);unsupported=0.;actions_all=[];terminal_count=np.zeros(episodes,np.int16)
    for t in range(env.horizon):
        obs,mask,recency=env._emit(states,subtypes,prior_mask,recency,pending,streams["noise"][:,t],streams["mask_u"][:,t]);prob=policy(obs,mask,recency,pending,t)
        if prob.shape!=(episodes,env.contract.action_count) or not np.isfinite(prob).all() or np.max(np.abs(prob.sum(1)-1))>1e-6: raise RuntimeError("invalid target probability")
        outside=np.ones(env.contract.action_count,bool);outside[env.supported]=False;unsupported+=float(prob[:,outside].sum());action=env._draw(prob,streams["policy_u"][:,t]);action[~alive]=env.supported[0]
        transition=env.transition[states,subtypes,pending,action];next_state=env._draw(transition,streams["transition_u"][:,t]);reward=np.zeros(episodes)
        for table in env.reward_components.values(): reward+=table[states,subtypes,pending,action]
        hazard=1. if t==env.horizon-1 else env.contract.termination_hazards[t];terminate=alive&(streams["termination_u"][:,t]<hazard)
        if env.contract.primary_reward_type=="terminal":
            death=streams["outcome_u"][:,t]<env.death_probability(next_state);terminal=np.zeros(episodes);terminal[terminate]=np.where(death[terminate],env.contract.terminal_reward_minimum,env.contract.terminal_reward_maximum);reward+=terminal;terminal_count[terminate]+=1
        returns+=(env.discount**t)*np.where(alive,reward,0);actions_all.append(action[alive].copy());alive&=~terminate;states=next_state;pending=action;prior_mask=mask
    actions=np.concatenate(actions_all);counts=np.bincount(actions,minlength=env.contract.action_count);p=counts/counts.sum();entropy=-float(np.sum(p[p>0]*np.log(p[p>0]))/math.log(env.contract.action_count))
    return {"mean_return":float(returns.mean()),"return_se":float(returns.std(ddof=1)/math.sqrt(episodes)),"returns":returns,"unsupported_mass":unsupported/max(episodes*env.horizon,1),"entropy":entropy,"distinct_actions":int(np.sum(counts>0)),"terminal_emission_max":int(terminal_count.max())}


def constant_policy(environment: Any, name: str) -> Policy:
    def fn(obs: np.ndarray, mask: np.ndarray, recency: np.ndarray, previous: np.ndarray, time_index: int) -> np.ndarray:
        n=len(obs); p=np.zeros((n,environment.contract.action_count))
        if name=="minimum": p[:,environment.supported[0]]=1
        elif name=="maximum": p[:,environment.supported[-1]]=1
        elif name=="random": p[:,environment.supported]=1/len(environment.supported)
        elif name=="ehr_matched":
            bins=environment.behavior.context_bin(obs,mask,recency,previous,time_index)
            for i in range(n): p[i]=environment.behavior.distribution(int(previous[i]),int(bins[i]),environment.contract.action_count)
        else: raise ValueError(name)
        return p
    return fn


def planner_policy(environment: Any, fit: Any, method: str, horizon: int, penalized: bool,
                   penalty: float, device: torch.device) -> Policy:
    """Frozen observation-only factorized planner; H1 is exhaustive and H4/H8 share its first-action interface."""
    f=environment.contract.feature_dim;k=environment.contract.action_count;signs=np.where(np.arange(f)%2==0,1.,-1.)
    contexts=np.asarray([(state,previous) for state in range(5) for previous in environment.supported],dtype=int)
    representative=np.asarray([signs*(state/2-1)*environment.observation_loading for state,_ in contexts],np.float32)
    representative_mask=np.ones_like(representative,bool);representative_delta=np.zeros_like(representative,np.float32)
    score_table=np.full((len(contexts),k),-np.inf)
    for action in environment.supported:
        action_array=np.full((len(contexts),1),action,np.int16); probability=np.zeros((len(contexts),1,k),np.float32);probability[:,:,action]=1
        dummy=SyntheticData(representative[:,None],representative_mask[:,None],representative_delta[:,None],action_array,representative[:,None],probability,np.zeros((len(contexts),1),np.float32),np.zeros((len(contexts),1),bool),np.ones((len(contexts),1),bool),np.zeros((len(contexts),1),np.int16))
        if method=="persistence_locf": mean=representative;scale=None
        elif method=="hgb_residual": mean=hgb_predict(fit,dummy)[:,0];scale=None
        else:
            mean,scale_full=wm.native_predictions(fit,dummy,device,batch=512);mean=mean[:,0];scale=scale_full[:,0] if scale_full is not None else None
        severity=np.sum(mean*signs[None],axis=1)/f;intensity=action/max(k-1,1)
        value=horizon*0.05*severity-0.03*intensity*intensity-0.02*np.abs(action-contexts[:,1])/max(k-1,1)
        if penalized and scale is not None:value-=penalty*np.mean(scale,axis=1)
        score_table[:,action]=value
    action_table=np.argmax(score_table,axis=1).reshape(5,len(environment.supported))
    previous_lookup={int(value):index for index,value in enumerate(environment.supported)}
    def fn(obs: np.ndarray, mask: np.ndarray, recency: np.ndarray, previous: np.ndarray, time_index: int) -> np.ndarray:
        signed=np.sum(obs*signs[None]*mask,axis=1)/np.maximum(mask.sum(1),1);state=np.clip(np.rint((signed/max(environment.observation_loading,1e-8)+1)*2),0,4).astype(int)
        prior=np.asarray([previous_lookup.get(int(value),0) for value in previous]);chosen=action_table[state,prior];return one_hot(chosen,k)
    return fn


def ope_estimates(data: SyntheticData, target_probability: np.ndarray, truth: float, estimators: list[str], clip: float) -> list[dict[str, Any]]:
    valid=data.valid; n,h=valid.shape; chosen=np.take_along_axis(target_probability,data.actions[...,None],axis=-1)[...,0]; beta=np.take_along_axis(data.behavior_probability,data.actions[...,None],axis=-1)[...,0]
    ratio=np.where(valid,chosen/np.maximum(beta,1e-12),1.0);ratio=np.minimum(ratio,clip);cum=np.cumprod(ratio,axis=1);discount=np.power(.99,np.arange(h));g=np.sum(data.rewards*valid*discount[None],axis=1);w=np.prod(ratio,axis=1)
    pdis=np.sum(cum*data.rewards*valid*discount[None],axis=1)
    normalized=cum/np.maximum(cum.sum(axis=0,keepdims=True),1e-12);wpdis_contribution=n*np.sum(normalized*data.rewards*valid*discount[None],axis=1)
    # Dataset-refit finite-horizon fitted-Q nuisance over coarse observable-history strata.
    signs=np.where(np.arange(data.observed.shape[-1])%2==0,1.,-1.);score=np.sum(data.observed*signs[None,None]*data.masks,axis=-1)/np.maximum(data.masks.sum(-1),1);bins=np.clip(np.rint((score+1.)*2.),0,4).astype(int)
    q=np.zeros((h,5,target_probability.shape[-1]),float);v=np.zeros((h+1,5),float)
    for t in range(h-1,-1,-1):
        next_bin=bins[:,min(t+1,h-1)];target=data.rewards[:,t]+.99*(~data.done[:,t])*v[t+1,next_bin]
        global_mean=float(np.mean(target[valid[:,t]])) if valid[:,t].any() else 0.
        q[t].fill(global_mean)
        for state in range(5):
            local_state=valid[:,t]&(bins[:,t]==state)
            for action in range(target_probability.shape[-1]):
                local=local_state&(data.actions[:,t]==action)
                if local.any():q[t,state,action]=float(np.mean(target[local]))
            if local_state.any():v[t,state]=float(np.mean(np.sum(target_probability[local_state,t]*q[t,state][None],axis=1)))
            else:v[t,state]=float(np.mean(q[t,state]))
    q_logged=np.asarray([q[t,bins[:,t],data.actions[:,t]] for t in range(h)]).T
    v_next=np.asarray([v[t+1,bins[:,min(t+1,h-1)]] for t in range(h)]).T
    delta=(data.rewards+.99*(~data.done)*v_next-q_logged)*valid
    previous=np.concatenate([np.ones((n,1)),cum[:,:-1]],axis=1);dr=v[0,bins[:,0]]+np.sum(previous*delta*discount[None],axis=1)
    normalized_previous=previous/np.maximum(previous.sum(axis=0,keepdims=True),1e-12);wdr_contribution=v[0,bins[:,0]]+n*np.sum(normalized_previous*delta*discount[None],axis=1)
    fqe=v[0,bins[:,0]];ess=float(w.sum()**2/max(np.sum(w*w),1e-12));rows=[]
    for estimator in estimators:
        if estimator=="IS": values=w*g;estimate=float(values.mean())
        elif estimator=="WIS": values=w*g;estimate=float(values.sum()/max(w.sum(),1e-12))
        elif estimator=="PDIS":values=pdis;estimate=float(values.mean())
        elif estimator in {"WPDIS","CWPDIS","support_restricted_WPDIS"}:values=wpdis_contribution;estimate=float(values.mean())
        elif estimator=="DR":values=dr;estimate=float(values.mean())
        elif estimator=="WDR":values=wdr_contribution;estimate=float(values.mean())
        elif estimator=="FQE":values=fqe;estimate=float(values.mean())
        else:raise ValueError(estimator)
        se=float(np.std(values,ddof=1)/math.sqrt(n)) if n>1 else math.nan;half=1.64485363*se
        rows.append({"estimator":estimator,"estimate":estimate,"interval_low":estimate-half,"interval_high":estimate+half,"truth":truth,"bias":estimate-truth,"squared_error":(estimate-truth)**2,"covered_90":estimate-half<=truth<=estimate+half,"ess":ess,"ess_fraction":ess/n,"unsupported_mass":float(np.mean((chosen<=0)&valid)),"weight_median":float(np.median(w)),"weight_p95":float(np.quantile(w,.95)),"weight_p99":float(np.quantile(w,.99)),"weight_max":float(np.max(w)),"finite":bool(np.isfinite(estimate))})
    return rows


def execution_decision(hard_failure: bool, numeric_methods: int, ope_rows: int, failures: int) -> str:
    if hard_failure: return "stop_environment_identity_or_evaluator_failure"
    if numeric_methods == 0 or ope_rows == 0 or failures: return "complete_negative_or_partial_pomdp_method_result"
    return "complete_pomdp_world_model_rl_ope_benchmark"
