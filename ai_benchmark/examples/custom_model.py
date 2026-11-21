"""The format for an external operator; replace training-mean mobility with your AI."""
import argparse
from pathlib import Path
import numpy as np
from ashebo_counterflowbench import load_case,load_split,grade_operator
import json

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,default=Path("prediction.npz"));args=p.parse_args()
    if args.output.suffix!=".npz":raise ValueError("Output must end with .npz")
    if args.output.exists():raise FileExistsError(args.output)
    # Normalization/fitting uses only the fixed training split.
    c=np.concatenate([c["unit_mobility"] for c in load_split("train")]).mean(0)
    case=load_case("seed12_k100");D=case.torque
    # This example is the known-H task: conservative rates and torque are supplied.
    rhs=case["rhs_conservative_y"].copy();rhs[...,2]-=case.kappa*c*D;rhs[...,3]+=case.kappa*c*D
    q=case.kappa*c*D*D
    np.savez_compressed(args.output,rhs_y=rhs,heat_cells=q,time=case.time)
    print(json.dumps(grade_operator(case,rhs,q),indent=2))

if __name__=="__main__":main()
