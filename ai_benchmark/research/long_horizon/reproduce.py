#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Yohannes Beyene Ashebo
"""Reproduce the core long-reference and four-model study in a fresh folder."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--release',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    src=Path(__file__).resolve().parent;out=args.output.resolve();out.mkdir(exist_ok=False)
    for name in ['generate_long.py','train_models.py','evaluate_models.py','hybrid_validation.py','adaptive_diagnostic.py','PROTOCOL.json','HYBRID_PROTOCOL.json','requirements.txt','LICENSE']:
        shutil.copy2(src/name,out/name)
    shutil.copytree(src/'dependencies',out/'dependencies',ignore=shutil.ignore_patterns('__pycache__'))
    release=str(args.release.resolve())
    for script,extra in [('generate_long.py',['--release',release]),('train_models.py',[]),('evaluate_models.py',['--release',release]),('hybrid_validation.py',['--release',release]),('adaptive_diagnostic.py',['--release',release])]:
        subprocess.run([sys.executable,'-u',str(out/script),*extra],cwd=out,check=True)
    print('Core reproduction completed. Earlier-checkpoint transfer and source/plot report additionally require the previous saved adapter package.')

if __name__=='__main__':main()
