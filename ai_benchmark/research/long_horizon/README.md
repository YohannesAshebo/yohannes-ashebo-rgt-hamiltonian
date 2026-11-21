# Scientific reproduction recipes

This directory preserves the executed longer-study scripts and protocols. The historical narrative is in `ORIGINAL_RUN_REPORT.md`; it describes the local run before public packaging. Current data, trained models and recorded evidence are shipped once under `../../src/ashebo_counterflowbench/resources/`.

Run `reproduce.py --release /path/to/this/repository --output /path/to/a/new/directory` using a Python environment with NumPy, SciPy, PyTorch and Matplotlib. It generates a new dataset and retrains/evaluates all four candidates without overwriting the included records. The three existing citation/manifest metadata changes from the original release are exempted from the historical all-file gate; native, Hamiltonian, audit and preparation source checks remain enforced. The public `counterflowbench verify --solver ...` command separately protects all 94 files at the current publication baseline.

The core reproduction uses no upstream author notebook. Optional `check_frozen_short_models.py` and the historical report verification additionally require earlier separately saved experiment packages; they are not part of the core `reproduce.py` sequence.
