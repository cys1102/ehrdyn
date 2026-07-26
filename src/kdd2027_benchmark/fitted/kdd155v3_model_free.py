from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def array_digest(value: np.ndarray) -> str:
    value=np.ascontiguousarray(value); digest=hashlib.sha256(f"{value.shape}|{value.dtype.str}".encode()); digest.update(value.view(np.uint8)); return digest.hexdigest()


def decision_from_execution(hard_stop: bool, rows: list[dict[str, Any]], reward_unavailable: bool) -> str:
    if hard_stop: return "stop_split_probability_or_leakage_failure"
    learned=[r for r in rows if r.get("method_family")=="learned"]; complete=[r for r in learned if r.get("execution_status")=="numeric_complete"]
    if not complete or reward_unavailable or len(complete)!=len(learned): return "complete_partial_with_reward_method_or_probability_nonexecutions"
    negative=any(bool(r.get("policy_collapse")) or float(r.get("mean_unsupported_probability_mass",0))>1e-8 for r in complete)
    return "complete_negative_model_free_collapse_or_support_failure" if negative else "complete_five_cohort_model_free_training"


class MLP(nn.Module):
    def __init__(self,inputs,outputs,hidden):
        super().__init__(); self.net=nn.Sequential(nn.Linear(inputs,hidden),nn.ReLU(),nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,outputs))
    def forward(self,x): return self.net(x)


class BCQ(nn.Module):
    def __init__(self,inputs,actions,hidden):
        super().__init__(); self.body=nn.Sequential(nn.Linear(inputs,hidden),nn.ReLU(),nn.Linear(hidden,hidden),nn.ReLU()); self.q=nn.Linear(hidden,actions); self.imitation=nn.Linear(hidden,actions)
    def forward(self,x):
        body=self.body(x); return self.q(body),self.imitation(body)


class IQL(nn.Module):
    def __init__(self,inputs,actions,hidden):
        super().__init__(); self.q=MLP(inputs,actions,hidden); self.v=MLP(inputs,1,hidden); self.policy=MLP(inputs,actions,hidden)


class DecisionTransformer(nn.Module):
    def __init__(self,inputs,actions,hidden,steps):
        super().__init__(); self.embed=nn.Linear(inputs+2,hidden); self.position=nn.Parameter(torch.zeros(1,steps,hidden)); layer=nn.TransformerEncoderLayer(hidden,4,hidden*2,.1,batch_first=True); self.encoder=nn.TransformerEncoder(layer,2); self.head=nn.Linear(hidden,actions)
    def forward(self,x,rtg,known):
        token=self.embed(torch.cat([x,rtg[...,None],known[...,None]],-1))+self.position[:,:x.shape[1]]; mask=torch.triu(torch.ones(x.shape[1],x.shape[1],device=x.device,dtype=torch.bool),diagonal=1); return self.head(self.encoder(token,mask=mask))


def model_digest(model):
    digest=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()): digest.update(name.encode()); digest.update(value.detach().cpu().contiguous().numpy().view(np.uint8))
    return digest.hexdigest()


def normalize(logits,support=None):
    value=np.asarray(logits,dtype=np.float64)
    if support is not None: value=np.where(support,value,-np.inf)
    value-=value.max(1,keepdims=True); output=np.exp(value); output[~np.isfinite(output)]=0
    if np.any(output.sum(1)<=0): raise RuntimeError("empty supported policy row")
    return output/output.sum(1,keepdims=True)


def one_hot(action,actions):
    output=np.zeros((len(action),actions)); output[np.arange(len(action)),action.astype(int)]=1; return output


@dataclass(slots=True)
class PolicyData:
    x: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    reward_known: np.ndarray
    next_x: np.ndarray
    done: np.ndarray
    support: np.ndarray
    behavior: np.ndarray
    severity: np.ndarray
    episode: np.ndarray
    step: np.ndarray


def _loader(arrays,batch,seed): return DataLoader(TensorDataset(*(torch.from_numpy(x) for x in arrays)),batch_size=batch,shuffle=True,generator=torch.Generator().manual_seed(seed),pin_memory=torch.cuda.is_available())


def _infer(function,x,device):
    output=[]
    with torch.no_grad():
        for start in range(0,len(x),8192): output.append(function(torch.from_numpy(x[start:start+8192]).to(device)).cpu().numpy())
    return np.concatenate(output)


def _finish(model,epoch,best_epoch,losses,started,cfg,device,objective,fidelity):
    slope=float(np.polyfit(np.arange(min(3,len(losses))),losses[-3:],1)[0]) if len(losses)>1 else math.nan
    return {"selected_epoch":best_epoch,"maximum_epoch":cfg["max_epochs"],"epochs_run":epoch,"validation_objective":min(losses),"late_epoch_validation_slope":slope,"convergence_status":"convergence_incomplete" if epoch==cfg["max_epochs"] and slope < -1e-4 else "converged_or_plateaued","parameter_count":sum(x.numel() for x in model.parameters()),"runtime_seconds":time.perf_counter()-started,"peak_memory_mb":torch.cuda.max_memory_allocated(device)/2**20 if device.type=="cuda" else 0,"checkpoint_sha256":model_digest(model),"device":str(device),"objective":objective,"fidelity":fidelity}


def _fit_loop(model,batches,validation_loss,cfg,device,train_loss):
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg["learning_rate"],weight_decay=cfg["weight_decay"]); best=None;best_epoch=0;stale=0;losses=[];started=time.perf_counter()
    for epoch in range(1,cfg["max_epochs"]+1):
        model.train()
        for batch in batches(epoch):
            batch=tuple(x.to(device,non_blocking=True) for x in batch); optimizer.zero_grad(set_to_none=True); loss=train_loss(model,batch); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),cfg["gradient_clip_norm"]); optimizer.step()
        value=float(validation_loss(model)); losses.append(value)
        if value<min(losses[:-1],default=math.inf)-1e-5: best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};best_epoch=epoch;stale=0
        else: stale+=1
        if epoch>=cfg["min_epochs"] and stale>=cfg["patience"]: break
    model.load_state_dict(best or model.state_dict()); return epoch,best_epoch,losses,started


def train_bc(train,validation,actions,cfg,seed,device):
    torch.manual_seed(seed); model=MLP(train.x.shape[1],actions,cfg["hidden_dim"]).to(device); batches=lambda epoch:_loader((train.x,train.action),cfg["batch_size"],seed+epoch); val=lambda m:nn.functional.cross_entropy(torch.from_numpy(_infer(m,validation.x,device)),torch.from_numpy(validation.action)).item(); loss=lambda m,b:nn.functional.cross_entropy(m(b[0]),b[1]); epoch,best,values,start=_fit_loop(model,batches,val,cfg,device,loss)
    return normalize(_infer(model,validation.x,device)),_finish(model,epoch,best,values,start,cfg,device,"behavior_cloning_cross_entropy","high_fidelity_discrete_reimplementation"),model


def train_q(method,train,validation,actions,cfg,seed,device):
    keep=np.flatnonzero(train.reward_known)
    if len(keep)<100: raise RuntimeError("insufficient_reward_known_transitions")
    torch.manual_seed(seed); model=MLP(train.x.shape[1],actions,cfg["hidden_dim"]).to(device); arrays=(train.x[keep],train.action[keep],train.reward[keep],train.next_x[keep],train.done[keep].astype(np.float32)); batches=lambda epoch:_loader(arrays,cfg["batch_size"],seed+epoch)
    def loss(m,b):
        q=m(b[0]); chosen=q.gather(1,b[1][:,None]).squeeze(1)
        with torch.no_grad(): target=b[2]+cfg["discount"]*(1-b[4])*m(b[3]).max(1).values
        base=nn.functional.smooth_l1_loss(chosen,target); return base+(cfg["cql_alpha"]*(torch.logsumexp(q,1)-chosen).mean() if method in {"discrete_cql","soft_spibb"} else 0)
    vk=np.flatnonzero(validation.reward_known)
    def val(m):
        with torch.no_grad():
            x=torch.from_numpy(validation.x[vk]).to(device);n=torch.from_numpy(validation.next_x[vk]).to(device);a=torch.from_numpy(validation.action[vk]).to(device);r=torch.from_numpy(validation.reward[vk]).to(device);d=torch.from_numpy(validation.done[vk].astype(np.float32)).to(device);q=m(x).gather(1,a[:,None]).squeeze(1);target=r+cfg["discount"]*(1-d)*m(n).max(1).values;return nn.functional.smooth_l1_loss(q,target).item()
    epoch,best,values,start=_fit_loop(model,batches,val,cfg,device,loss);q=_infer(model,validation.x,device)
    if method=="discrete_cql": output=normalize(q,validation.support)
    else:
        candidate=normalize(q/cfg["soft_spibb_temperature"],validation.support);count=np.bincount(train.action,minlength=actions);beta=count/(count+cfg["soft_spibb_min_count"]);output=validation.behavior*(1-beta)+candidate*beta;output=np.where(validation.support,output,0);output/=output.sum(1,keepdims=True)
    fidelity="high_fidelity_discrete_reimplementation" if method=="discrete_cql" else "conceptual_offline_adapter"
    return output,_finish(model,epoch,best,values,start,cfg,device,"conservative_TD" if method=="discrete_cql" else "Soft_SPIBB_count_regularized_TD",fidelity),model


def train_bcq(train,validation,actions,cfg,seed,device):
    keep=np.flatnonzero(train.reward_known)
    if len(keep)<100: raise RuntimeError("insufficient_reward_known_transitions")
    torch.manual_seed(seed); model=BCQ(train.x.shape[1],actions,cfg["hidden_dim"]).to(device); arrays=(train.x[keep],train.action[keep],train.reward[keep],train.next_x[keep],train.done[keep].astype(np.float32)); batches=lambda epoch:_loader(arrays,cfg["batch_size"],seed+epoch)
    def loss(m,b):
        q,imit=m(b[0]); chosen=q.gather(1,b[1][:,None]).squeeze(1)
        with torch.no_grad():
            nq,nimit=m(b[3]); eligible=torch.softmax(nimit,1)>=cfg["bcq_behavior_threshold"]*torch.softmax(nimit,1).max(1,keepdim=True).values; target=b[2]+cfg["discount"]*(1-b[4])*nq.masked_fill(~eligible,-1e9).max(1).values
        return nn.functional.smooth_l1_loss(chosen,target)+nn.functional.cross_entropy(imit,b[1])
    vk=np.flatnonzero(validation.reward_known)
    def val(m):
        with torch.no_grad():
            x=torch.from_numpy(validation.x[vk]).to(device);n=torch.from_numpy(validation.next_x[vk]).to(device);a=torch.from_numpy(validation.action[vk]).to(device);r=torch.from_numpy(validation.reward[vk]).to(device);d=torch.from_numpy(validation.done[vk].astype(np.float32)).to(device);q,imit=m(x);nq,nimit=m(n);eligible=torch.softmax(nimit,1)>=cfg["bcq_behavior_threshold"]*torch.softmax(nimit,1).max(1,keepdim=True).values;target=r+cfg["discount"]*(1-d)*nq.masked_fill(~eligible,-1e9).max(1).values;return (nn.functional.smooth_l1_loss(q.gather(1,a[:,None]).squeeze(1),target)+nn.functional.cross_entropy(imit,a)).item()
    epoch,best,values,start=_fit_loop(model,batches,val,cfg,device,loss);q=_infer(lambda x:model(x)[0],validation.x,device);imit=normalize(_infer(lambda x:model(x)[1],validation.x,device));eligible=(imit>=cfg["bcq_behavior_threshold"]*imit.max(1,keepdims=True))&validation.support;output=one_hot(np.argmax(np.where(eligible,q,-np.inf),1),actions)
    return output,_finish(model,epoch,best,values,start,cfg,device,"BCQ_TD_plus_imitation","high_fidelity_discrete_reimplementation"),model


def train_iql(train,validation,actions,cfg,seed,device):
    keep=np.flatnonzero(train.reward_known)
    if len(keep)<100: raise RuntimeError("insufficient_reward_known_transitions")
    torch.manual_seed(seed);model=IQL(train.x.shape[1],actions,cfg["hidden_dim"]).to(device)
    arrays=(train.x[keep],train.action[keep],train.reward[keep],train.next_x[keep],train.done[keep].astype(np.float32))
    batches=lambda epoch:_loader(arrays,cfg["batch_size"],seed+epoch)
    expectile=float(cfg["iql_expectile"]);temperature=float(cfg["iql_temperature"]);maximum=float(cfg["iql_max_weight"])
    def objective(m,b):
        q=m.q(b[0]);v=m.v(b[0]).squeeze(1);chosen=q.gather(1,b[1][:,None]).squeeze(1)
        with torch.no_grad(): target=b[2]+cfg["discount"]*(1-b[4])*m.v(b[3]).squeeze(1)
        q_loss=nn.functional.smooth_l1_loss(chosen,target);advantage=chosen.detach()-v;weight=torch.where(advantage>0,expectile,1-expectile)
        v_loss=(weight*advantage.square()).mean();actor_weight=torch.exp(temperature*advantage.detach()).clamp(max=maximum)
        actor_loss=(actor_weight*nn.functional.cross_entropy(m.policy(b[0]),b[1],reduction="none")).mean()
        return q_loss+v_loss+actor_loss
    vk=np.flatnonzero(validation.reward_known)
    def val(m):
        with torch.no_grad():
            x=torch.from_numpy(validation.x[vk]).to(device);a=torch.from_numpy(validation.action[vk]).to(device);r=torch.from_numpy(validation.reward[vk]).to(device);n=torch.from_numpy(validation.next_x[vk]).to(device);d=torch.from_numpy(validation.done[vk].astype(np.float32)).to(device)
            q=m.q(x).gather(1,a[:,None]).squeeze(1);target=r+cfg["discount"]*(1-d)*m.v(n).squeeze(1)
            return (nn.functional.smooth_l1_loss(q,target)+nn.functional.cross_entropy(m.policy(x),a)).item()
    epoch,best,values,start=_fit_loop(model,batches,val,cfg,device,objective);output=normalize(_infer(model.policy,validation.x,device))
    return output,_finish(model,epoch,best,values,start,cfg,device,"discrete_IQL_expectile_value_and_advantage_weighted_policy","high_fidelity_discrete_reimplementation"),model


def _sequence_arrays(data: PolicyData, context: int, discount: float):
    rows=len(data.x);features=data.x.shape[1];x=np.zeros((rows,context,features),np.float32);rtg=np.zeros((rows,context),np.float32);known=np.zeros((rows,context),np.float32)
    lookup={}
    for index,(episode,step) in enumerate(zip(data.episode,data.step,strict=True)):lookup[(int(episode),int(step))]=index
    returns=np.zeros(rows,np.float32)
    for episode in np.unique(data.episode):
        local=np.flatnonzero(data.episode==episode);local=local[np.argsort(data.step[local])];running=0.
        for index in local[::-1]:running=float(data.reward[index])+discount*running;returns[index]=running
    for row,(episode,step) in enumerate(zip(data.episode,data.step,strict=True)):
        start=max(0,int(step)-context+1);positions=[lookup.get((int(episode),value)) for value in range(start,int(step)+1)];positions=[value for value in positions if value is not None];offset=context-len(positions)
        if positions:x[row,offset:]=data.x[positions];rtg[row,offset:]=returns[positions];known[row,offset:]=data.reward_known[positions]
    return x,rtg,known


def train_decision_transformer(train,validation,actions,cfg,seed,device):
    context=int(cfg["decision_transformer_context"]);tx,tr,tk=_sequence_arrays(train,context,cfg["discount"]);vx,vr,vk=_sequence_arrays(validation,context,cfg["discount"])
    torch.manual_seed(seed);model=DecisionTransformer(train.x.shape[1],actions,cfg["hidden_dim"],context).to(device)
    batches=lambda epoch:_loader((tx,tr,tk,train.action),cfg["sequence_batch_size"],seed+epoch)
    def objective(m,b):return nn.functional.cross_entropy(m(b[0],b[1],b[2])[:,-1],b[3])
    def val(m):
        logits=[];m.eval()
        with torch.no_grad():
            for start in range(0,len(vx),2048):logits.append(m(torch.from_numpy(vx[start:start+2048]).to(device),torch.from_numpy(vr[start:start+2048]).to(device),torch.from_numpy(vk[start:start+2048]).to(device))[:,-1].cpu())
        return nn.functional.cross_entropy(torch.cat(logits),torch.from_numpy(validation.action)).item()
    epoch,best,values,start=_fit_loop(model,batches,val,cfg,device,objective);model.eval();logits=[]
    with torch.no_grad():
        for begin in range(0,len(vx),2048):logits.append(model(torch.from_numpy(vx[begin:begin+2048]).to(device),torch.from_numpy(vr[begin:begin+2048]).to(device),torch.from_numpy(vk[begin:begin+2048]).to(device))[:,-1].cpu().numpy())
    output=normalize(np.concatenate(logits))
    return output,_finish(model,epoch,best,values,start,cfg,device,"causal_return_conditioned_action_cross_entropy","high_fidelity_discrete_adapter"),model


def validate_probability(probability,tolerance=1e-6):
    value=np.asarray(probability)
    return bool(value.ndim==2 and np.isfinite(value).all() and np.all(value>=0) and np.max(np.abs(value.sum(1)-1))<=tolerance)
