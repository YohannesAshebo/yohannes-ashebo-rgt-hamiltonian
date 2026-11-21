"""A small command-line workflow for researchers and external neural models."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from .data import RESOURCES,case_ids,dataset_index,load_case,load_split,verify_dataset,verify_solver
from .metrics import grade_operator,grade_rollout,reference_audit

def emit(value,path=None):
    content=json.dumps(value,indent=2,allow_nan=False)+"\n"
    if path:
        target=Path(path)
        if target.exists():raise FileExistsError(f"Output already exists: {target}; choose a new path")
        target.parent.mkdir(parents=True,exist_ok=True);target.write_text(content)
    print(content,end="")

def main(argv=None):
    parser=argparse.ArgumentParser(prog="counterflowbench",description="Ashebo CounterflowBench: reference data, frozen AI inference, and native audits")
    sub=parser.add_subparsers(dest="command",required=True)
    sub.add_parser("info",help="Show tasks, layouts and fixed train/test splits")
    v=sub.add_parser("verify",help="Check data/model hashes and optional solver integrity");v.add_argument("--solver",type=Path)
    d=sub.add_parser("demo",help="Score a positive mean-mobility baseline or frozen neural surrogate")
    d.add_argument("--case",default="seed12_k100");d.add_argument("--pretrained",action="store_true");d.add_argument("--output",type=Path)
    e=sub.add_parser("evaluate",help="Grade your prediction NPZ using native reference labels")
    e.add_argument("prediction",type=Path);e.add_argument("--case",required=True);e.add_argument("--output",type=Path);e.add_argument("--rollout",action="store_true");e.add_argument("--solver",type=Path)
    x=sub.add_parser("export",help="Copy one fixed dataset split for your own training pipeline")
    x.add_argument("--split",choices=["train","validation","test"],required=True);x.add_argument("--output",type=Path,required=True)
    g=sub.add_parser("generate",help="Generate one native reference; requires SciPy and a C compiler")
    g.add_argument("--solver",type=Path);g.add_argument("--seed",type=int,default=16);g.add_argument("--kappa",type=float,default=100.)
    g.add_argument("--stop",type=float,default=10.);g.add_argument("--frames",type=int,default=21);g.add_argument("--output",type=Path,required=True)
    r=sub.add_parser("rollout",help="Run the pretrained response coupled to the exact native Hamiltonian")
    r.add_argument("--case",default="seed12_k100");r.add_argument("--solver",type=Path);r.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(argv)
    try:
        if args.command=="info":
            index=dataset_index();emit(dict(name=index["name"],version=index["version"],cells=index["cells"],state_order=index["state_order"],canonical_order=index["canonical_order"],
                splits={s:dict(trajectories=len(case_ids(s)),snapshots=len(case_ids(s))*21) for s in ["train","validation","test"]},
                time_range=index["time_range"],units=index["units"],tasks=["known-H dissipative operator","full-flow operator prediction","free rollout and native H/heat/content audits"],
                known_physics="The frozen hybrid receives exact native conservative flow and phase torque. A full-flow learner is evaluated separately.",scope=index["scope"]))
        elif args.command=="verify":
            result=verify_dataset();f=json.loads((RESOURCES/"pretrained/HYBRID_FREEZE.json").read_text())
            for name,key in [("native_H_hybrid.pt","model_sha256"),("HYBRID_NORMALIZERS.npz","normalizers_sha256"),("HYBRID_PROTOCOL.json","protocol_sha256")]:
                if hashlib.sha256((RESOURCES/"pretrained"/name).read_bytes()).hexdigest()!=f[key]:raise ValueError(f"Model checksum failed: {name}")
            result["frozen_neural_model_verified"]=True
            if args.solver:result["solver"]=verify_solver(args.solver)
            audits=[reference_audit(load_case(cid)) for cid in case_ids()]
            result["saved_state_reference_audit_maxima"]={key:max(a[key] for a in audits) for key in ["energy_plus_heat_relative_drift","total_content_relative_drift","instantaneous_power_residual_max_abs"]}
            emit(result)
            if args.solver and not result["solver"]["verified"]:return 1
        elif args.command=="demo":
            case=load_case(args.case)
            if args.pretrained:
                import torch
                torch.set_num_threads(1)
                from .surrogate import load_pretrained
                prediction=load_pretrained().predict_operator(case);name="frozen neural positive-mobility hybrid"
            else:
                training=load_split("train");c=np.concatenate([z["unit_mobility"] for z in training]).mean(0);D=case.torque
                rhs=case["rhs_conservative_y"].copy();rhs[...,2]-=case.kappa*c*D;rhs[...,3]+=case.kappa*c*D
                prediction=dict(rhs_y=rhs,heat_cells=case.kappa*c*D*D);name="NumPy training-mean positive mobility baseline"
            emit(dict(model=name,task="known-H dissipative response",native_conservative_flow_and_torque_supplied=True,
                reference_audit=reference_audit(case),prediction=grade_operator(case,prediction["rhs_y"],prediction["heat_cells"])),args.output)
        elif args.command=="evaluate":
            case=load_case(args.case)
            with np.load(args.prediction,allow_pickle=False) as z:prediction={key:z[key] for key in z.files}
            if "time" in prediction and not np.array_equal(prediction["time"],case.time):raise ValueError("Prediction times do not match this case")
            if args.rollout:
                from .native import NativeHamiltonian
                with NativeHamiltonian(args.solver) as native:result=grade_rollout(case,prediction["state_y"],prediction["heat_cells"],native.energy)
            else:result=grade_operator(case,prediction["rhs_y"],prediction["heat_cells"])
            emit(result,args.output)
        elif args.command=="export":
            import shutil
            args.output.mkdir(parents=True,exist_ok=False)
            rows=[r for r in dataset_index()["cases"] if r["split"]==args.split]
            for row in rows:shutil.copy2(RESOURCES/"dataset"/row["file"],args.output/row["file"])
            (args.output/"index.json").write_text(json.dumps({**dataset_index(),"cases":rows},indent=2)+"\n")
            emit(dict(exported=len(rows),split=args.split,output=str(args.output)))
        elif args.command=="generate":
            from .native import NativeHamiltonian
            with NativeHamiltonian(args.solver) as native:case=native.generate(args.seed,args.kappa,args.stop,args.frames,args.output)
            emit(dict(generated=case.id,output=str(args.output),audit=reference_audit(case),scope="New reference; not an automatic extension of the shipped validation range"))
        elif args.command=="rollout":
            if args.output.suffix!=".npz":raise ValueError("Rollout output must end with .npz")
            if args.output.exists():raise FileExistsError(args.output)
            import torch
            torch.set_num_threads(1)
            from .surrogate import load_pretrained
            from .native import NativeHamiltonian
            case=load_case(args.case);model=load_pretrained()
            with NativeHamiltonian(args.solver) as native:
                result=native.hybrid_rollout(case.state_y[0],case.time,case.kappa,model)
                score=grade_rollout(case,result["state_y"],result["heat_cells"],native.energy)
            args.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(args.output,**result)
            emit(dict(output=str(args.output),model="frozen native-H hybrid",scores=score))
    except (ValueError,KeyError,FileNotFoundError,FileExistsError,ImportError,RuntimeError) as error:
        print(f"counterflowbench: {error}",file=sys.stderr);return 2
    return 0

if __name__=="__main__":raise SystemExit(main())
