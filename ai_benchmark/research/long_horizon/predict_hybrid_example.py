#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Yohannes Beyene Ashebo
"""Run the frozen positive mobility model on the supplied held-out reference."""
import hashlib
import json
import numpy as np
import torch
from train_models import ROOT
from hybrid_validation import Hybrid,operator_score

def main():
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
    frozen=json.loads((ROOT/'HYBRID_FREEZE.json').read_text());cfg=json.loads((ROOT/'HYBRID_PROTOCOL.json').read_text())
    assert hashlib.sha256((ROOT/'models/native_H_hybrid.pt').read_bytes()).hexdigest()==frozen['model_sha256']
    with np.load(ROOT/'HYBRID_NORMALIZERS.npz') as z:norm={k:z[k] for k in z.files}
    model=Hybrid(norm,cfg['network']).double();model.load_state_dict(torch.load(ROOT/'models/native_H_hybrid.pt',map_location='cpu',weights_only=True));model.eval()
    with np.load(ROOT/'data/seed12_k100.npz') as z:z={k:z[k] for k in z.files}
    print(json.dumps(dict(seed_group=12,kappa=100,time_range=[0,10],**operator_score(model,z),
        note='Frozen operator inference uses the supplied exact native torque labels; the full coupled rollout is stored separately.'),indent=2))

if __name__=='__main__':main()
