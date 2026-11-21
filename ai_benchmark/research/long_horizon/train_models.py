#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Yohannes Beyene Ashebo
"""Train three finite-grid DCT candidates; freeze before held-out evaluation."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'dependencies'))
from radial_generic import RadialFeatures,DCTBackbone,CanonicalPoisson,canonical_from_log,friction_apply

def save(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def relative(a,b):
    den=np.linalg.norm(b);return float(np.linalg.norm(a-b)/den) if den else None

def load_rows(split):
    manifest=json.loads((ROOT/'MANIFEST.json').read_text());rows=[]
    for e in manifest['cases']:
        if e['split']==split:
            assert hashlib.sha256((ROOT/e['file']).read_bytes()).hexdigest()==e['sha256']
            with np.load(ROOT/e['file']) as z:rows.append((e,{k:z[k] for k in z.files}))
    return rows

def arrays(rows):
    y=np.concatenate([z['state_y'].transpose(0,2,1) for _,z in rows])
    con=np.concatenate([z['rhs_conservative_y'].transpose(0,2,1) for _,z in rows])
    unit=np.concatenate([np.stack([z['rhs_dissipative_unit_y'][...,2],z['rhs_dissipative_unit_y'][...,3],z['heat_cells_unit']],axis=1) for _,z in rows])
    H=np.concatenate([z['H'] for _,z in rows]);c=np.concatenate([z['unit_mobility'] for _,z in rows])
    return dict(y=y,con=con,unit=unit,H=H,c=c)

def prepare_norm(train,volume,radius):
    norm=dict(volume=volume,radius=radius,con_mean=train['con'].mean(0),unit_mean=train['unit'].mean(0),
        con_scale=np.maximum(np.sqrt((train['con']**2).mean((0,2))),1e-10),
        unit_scale=np.maximum(np.sqrt((train['unit']**2).mean((0,2))),1e-20),
        H_mean=train['H'].mean(),H_scale=max(train['H'].std(),1.))
    x=canonical_from_log(torch.tensor(train['y']))
    for invariant,name in [(True,'inv'),(False,'raw')]:
        with torch.no_grad():fe=RadialFeatures(volume,radius,invariant)(x).numpy()
        norm[name+'_mean']=fe.mean((0,2),keepdims=True)
        norm[name+'_scale']=np.maximum(fe.std((0,2),keepdims=True),1e-8)
    mean=train['c'].mean(0);support=(train['c'].max(0)>0).astype(float)
    floor=.01*np.sqrt((train['c']**2).mean())
    norm['c_base']=np.maximum(mean,floor)*support;norm['support']=support
    return norm

class Candidate(nn.Module):
    """A learned vector field or learned Hamiltonian, with no native H call."""
    def __init__(self,kind,norm,cfg):
        super().__init__();self.kind=kind
        for key,value in norm.items():self.register_buffer(key,torch.as_tensor(value,dtype=torch.float64))
        self.features=RadialFeatures(norm['volume'],norm['radius'],kind!='generic_raw_phases')
        self.J=CanonicalPoisson(learn_filter=False)
        self.mechanical=DCTBackbone(11,9 if kind=='direct_DCT' else 1,width=cfg['width'],modes=cfg['modes'],layers=cfg['layers'])
        self.thermal=DCTBackbone(11,3 if kind=='direct_DCT' else 1,width=cfg['width'],modes=cfg['modes'],layers=cfg['layers'])
        # Initial direct rates use training means. Initial learned H is constant.
        for net in [self.mechanical,self.thermal]:nn.init.zeros_(net.head[-1].weight);nn.init.zeros_(net.head[-1].bias)
        if kind!='direct_DCT':nn.init.constant_(self.thermal.head[-1].bias,float(np.log(np.expm1(1.))))

    def split(self,y,create_graph=False):
        with torch.enable_grad():
            x=canonical_from_log(y)
            if self.kind!='direct_DCT' and not x.requires_grad:x=x.requires_grad_(True)
            feat=self.features(x);name='raw' if self.kind=='generic_raw_phases' else 'inv'
            feature=(feat-getattr(self,name+'_mean'))/getattr(self,name+'_scale')
            energy_output=self.mechanical(feature)
            H=self.H_mean+(energy_output[:,-1]*self.volume).sum(-1)
            thermal=self.thermal(feature)
            if self.kind=='direct_DCT':
                conservative=self.con_mean+energy_output[:,:8]*self.con_scale[None,:,None]
                unit=self.unit_mean+thermal*self.unit_scale[None,:,None]*self.support
                gradient=None
            else:
                gradient=torch.autograd.grad(H.sum(),x,create_graph=create_graph)[0]
                conx=self.J(gradient)
                conservative=torch.cat([conx[:,:2]/x[:,:2],conx[:,2:]],dim=1)
                gamma=F.softplus(thermal)
                # G = sqrt(c_base)*(+theta_C,-theta_R); rank one.
                z=torch.zeros_like(x[:,0]);a=torch.sqrt(self.c_base).expand_as(z)
                G=torch.stack([z,z,a,-a,z,z,z,z],dim=1)[:,None]
                diss,heat=friction_apply(torch.zeros_like(x),torch.ones_like(x[:,:1]),gradient,G,gamma)
                unit=torch.cat([diss[:,2:4],heat],dim=1)
        return conservative,unit,H,gradient

    def vector(self,y,k):
        con,unit,H,g=self.split(y,False)
        # No clipping/projection/true-H correction is applied here.
        zero=torch.zeros_like(con[:,0]);diss=torch.stack([zero,zero,unit[:,0],unit[:,1],zero,zero,zero,zero],dim=1)
        return (con+k[:,None,None]*diss).detach(),(k[:,None]*unit[:,2]).detach(),H.detach()

def loss_value(model,batch,graph=False):
    con,unit,H,_=model.split(batch['y'],graph)
    lc=(((con-batch['con'])/model.con_scale[None,:,None])**2).mean()
    ld=(((unit-batch['unit'])/model.unit_scale[None,:,None])**2).mean()
    le=(((H-batch['H'])/model.H_scale)**2).mean()
    return lc+ld+.01*le,dict(conservative=float(lc.detach()),dissipative_heat=float(ld.detach()),energy=float(le.detach()))

def evaluate_loss(model,data):
    totals=[];components=[];model.eval()
    for begin in range(0,len(data['H']),24):
        batch={key:value[begin:begin+24] for key,value in data.items() if key!='c'}
        value,parts=loss_value(model,batch,False);totals.append((len(batch['H']),float(value.detach())));components.append(parts)
    return sum(n*v for n,v in totals)/sum(n for n,_ in totals),{key:float(np.mean([d[key] for d in components])) for key in components[0]}

def train_one(kind,norm,train,val,protocol,out):
    cfg=protocol['training'];seed=protocol['networks']['training_seed'];torch.manual_seed(seed)
    model=Candidate(kind,norm,protocol['networks']).double()
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    rng=np.random.default_rng(seed);best=float('inf');best_state=None;best_update=None;history=[];began=time.perf_counter()
    for update in range(cfg['updates']+1):
        if update:
            idx=rng.choice(len(train['H']),cfg['batch_size'],replace=False)
            batch={key:value[idx] for key,value in train.items() if key!='c'}
            model.train();opt.zero_grad(set_to_none=True);value,_=loss_value(model,batch,True)
            if not torch.isfinite(value):raise FloatingPointError(f'Nonfinite loss {kind}, update{update}')
            value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip_norm']);opt.step()
        if update%cfg['validation_every']==0:
            tl,tp=evaluate_loss(model,train);vl,vp=evaluate_loss(model,val)
            row=dict(update=update,training_loss=tl,validation_loss=vl,training_components=tp,validation_components=vp,seconds=time.perf_counter()-began)
            history.append(row);print(json.dumps(dict(kind=kind,**row)),flush=True)
            if vl<best:best=vl;best_update=update;best_state=copy.deepcopy(model.state_dict())
            save(out/(kind+'_progress.json'),history)
    model.load_state_dict(best_state);file=out/(kind+'.pt')
    torch.save(dict(kind=kind,state_dict=model.state_dict(),best_update=best_update,validation_loss=best),file)
    save(out/(kind+'_training.json'),history)
    return dict(kind=kind,file=str(file.relative_to(ROOT)),sha256=hashlib.sha256(file.read_bytes()).hexdigest(),best_update=best_update,
        best_validation_loss=best,parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),seconds=time.perf_counter()-began)

def load_model(row,protocol):
    assert hashlib.sha256((ROOT/row['file']).read_bytes()).hexdigest()==row['sha256']
    with np.load(ROOT/'NORMALIZERS.npz') as z:norm={k:z[k] for k in z.files}
    model=Candidate(row['kind'],norm,protocol['networks']).double()
    model.load_state_dict(torch.load(ROOT/row['file'],map_location='cpu',weights_only=True)['state_dict']);model.eval();return model

def main():
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
    protocol=json.loads((ROOT/'PROTOCOL.json').read_text());rows=load_rows('train');valrows=load_rows('validation')
    train=arrays(rows);val=arrays(valrows);norm=prepare_norm(train,rows[0][1]['reference_volume'],rows[0][1]['r'])
    np.savez_compressed(ROOT/'NORMALIZERS.npz',**norm)
    train={key:torch.tensor(value) for key,value in train.items()};val={key:torch.tensor(value) for key,value in val.items()}
    out=ROOT/'models';out.mkdir(exist_ok=False);models=[]
    for kind in protocol['training']['models']:models.append(train_one(kind,norm,train,val,protocol,out))
    frozen=dict(protocol_sha256=hashlib.sha256((ROOT/'PROTOCOL.json').read_bytes()).hexdigest(),normalizers_sha256=hashlib.sha256((ROOT/'NORMALIZERS.npz').read_bytes()).hexdigest(),
        models=models,training_samples=len(train['H']),validation_samples=len(val['H']),test_arrays_opened_by_training=False,
        selection='Validation only, before held-out arrays are opened by model evaluation',training_seed=protocol['networks']['training_seed'])
    save(ROOT/'MODEL_FREEZE.json',frozen);print(json.dumps(dict(training_complete=True,models=models)),flush=True)

if __name__=='__main__':main()
