"""A minimal unconstrained full-rate learner, not the validated heating surrogate."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from ashebo_counterflowbench import load_case,load_split,grade_operator

def raw(rows):
    x=[];target=[]
    for case in rows:
        y=case.state_y.copy();y[...,:2]-=np.log(case["reference_volume"])[None,:,None];y[...,6:8]/=case["reference_volume"][None,:,None]
        radius=np.broadcast_to((case["r"]/8.)[None,:,None],y.shape[:2]+(1,))
        kappa=np.full_like(radius,case.kappa/100.)
        x.append(np.concatenate([y,radius,kappa],-1).transpose(0,2,1))
        target.append(np.concatenate([case.rhs_y,case.heat_power[...,None]],-1).transpose(0,2,1))
    return np.concatenate(x),np.concatenate(target)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--steps",type=int,default=100);p.add_argument("--output",type=Path,required=True);args=p.parse_args()
    if args.steps<1:raise ValueError("steps must be positive")
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);torch.manual_seed(20261120)
    train_x,train_y=raw(load_split("train"));val_x,val_y=raw(load_split("validation"))
    mean=train_x.mean((0,2),keepdims=True);scale=np.maximum(train_x.std((0,2),keepdims=True),1e-8)
    sy=np.maximum(np.sqrt((train_y*train_y).mean((0,2),keepdims=True)),1e-12)
    tx,ty,vx,vy=[torch.tensor(a,dtype=torch.float64) for a in [(train_x-mean)/scale,train_y/sy,(val_x-mean)/scale,val_y/sy]]
    model=nn.Sequential(nn.Conv1d(10,32,1),nn.GELU(),nn.Conv1d(32,32,1),nn.GELU(),nn.Conv1d(32,9,1)).double()
    opt=torch.optim.Adam(model.parameters(),lr=.001);best=float("inf");weights=None;rng=np.random.default_rng(20261120);history=[]
    for step in range(1,args.steps+1):
        idx=rng.choice(len(tx),16,False);opt.zero_grad(set_to_none=True);loss=((model(tx[idx])-ty[idx])**2).mean();loss.backward();opt.step()
        if step%10==0 or step==args.steps:
            with torch.no_grad():validation=float(((model(vx)-vy)**2).mean())
            history.append(dict(step=step,validation_loss=validation))
            if validation<best:best=validation;weights=copy.deepcopy(model.state_dict())
    model.load_state_dict(weights);torch.save(model.state_dict(),args.output/"example.pt");np.savez(args.output/"normalizers.npz",mean=mean,scale=scale,target_scale=sy)
    freeze=dict(sha256=hashlib.sha256((args.output/"example.pt").read_bytes()).hexdigest(),selected_using="validation only",validation_loss=best,test_opened=False)
    (args.output/"freeze.json").write_text(json.dumps(freeze,indent=2)+"\n")
    # The example's test is first opened after its checkpoint and normalizers freeze.
    case=load_case("seed12_k100");x,_=raw([case])
    with torch.no_grad():pred=(model(torch.tensor((x-mean)/scale,dtype=torch.float64)).numpy()*sy).transpose(0,2,1)
    np.savez_compressed(args.output/"prediction.npz",rhs_y=pred[...,:8],heat_cells=pred[...,8],time=case.time)
    score=grade_operator(case,pred[...,:8],pred[...,8]);(args.output/"scores.json").write_text(json.dumps(score,indent=2)+"\n")
    (args.output/"training.json").write_text(json.dumps(history,indent=2)+"\n")
    print(json.dumps(dict(steps=args.steps,validation_loss=best,test_scores=score,scope="Example interface only; no claimed accuracy, conservation or free-rollout guarantee"),indent=2))

if __name__=="__main__":main()
