# Ashebo RGT Hamiltonian Reference Solver and Numerical Audit

**Yohannes Beyene Ashebo · Ashebophysics · Version 0.1.0 · Apache 2.0**

An open-source research reference for the radial Relational Gap Theory
Hamiltonian and its current-dependent counterflow extension. The release
includes the original Python/C implementation, saved field states, and an
independent audit of energy, gradients, continuity and heat transfer.

This is a research preview. It provides reproducible numerical checks;
physical validation and spatial convergence of evolved solutions are not
established. A separate static smooth-profile study finds approximately
first-order convergence of integrated heat on its tested input families.

## Run the audit

Use Python 3.12 and a C compiler named `cc` (tested on Linux).

```sh
python3 -m pip install -r requirements.txt
python3 audit_mechanics.py
```

The audit checks the two archived states at solver time s=90, writes its
results to `results/`, and records the absent 1024-cell case as not run.
It compiles the native C kernels in a temporary working directory and leaves
the source, input snapshots and original Hamiltonian equations unchanged.

For plots and a guarded analysis of the original grid-level audit:

```sh
python3 make_figures.py
python3 analyze_convergence.py inputs/full_hamiltonian_audit.json
```

For the five-grid, zero-integration operator study and saved-state radial
comparison:

```sh
python3 static_scaling.py
python3 make_static_figures.py
```

These are manufactured static profiles on 128 through 2048 cells, with fixed
active exchange strength. They are not new evolved s=90 solutions. The
fixed-continuum-cap family approaches a finite integrated heat value with
observed order approximately 1.00 over the finest grid triples. Around 99.8%
of the net historical s=90 heat jump is localized to reference radius r<1.75,
not the moving core boundary near r=2.32. See `docs/STATIC_SCALING.md` for
the complete study and its limits. No kappa/dx correction has been applied.

The default audit uses the visible source in `native/`. Researchers can inspect
and modify that source directly. To reproduce the untouched archived snapshot
instead, run:

```sh
python3 audit_mechanics.py --archive inputs/Ashebo_RGT_Autonomous_Heat_Test.zip --output snapshot_results
```

## What is included

| Path | Contents |
| --- | --- |
| `native/` | Original radial field model, current-dependent exchange, heat kernel, solver wrappers and saved states |
| `audit_mechanics.py` | Reconstructed independent review audit; not the original JSON generator |
| `analyze_convergence.py` | Richardson arithmetic with comparability checks and residual exclusions |
| `make_figures.py` | Static diagnostic figures |
| `static_scaling.py` | Five-grid manufactured-profile study and conservative radial binning |
| `make_static_figures.py` | Scaling and saved-state radial figures |
| `inputs/` | Unchanged source snapshots and the original four-row numerical audit |
| `results/` | Completed checks, per-cell heat data and figures |
| `docs/HAMILTONIAN.md` | Equations and the scope of the implemented variants |
| `docs/AUDIT_MECHANICS.md` | Detailed methods and reproducibility notes |
| `docs/KNOWN_LIMITATIONS.md` | Failed requirements, missing inputs and remaining scientific questions |
| `docs/STATIC_SCALING.md` | Static operator scaling and localization of the historical heat jump |

## Recorded numerical results

At the saved s=90 states:

| Check | 256 cells | 512 cells |
| --- | ---: | ---: |
| Maximum coordinate-gradient discrepancy, independent complex-step formula | 2.96e-16 | 7.28e-16 |
| Minimum sampled centered-difference discrepancy | 6.57e-12 | 7.66e-12 |
| Reproduced heat power, kappa=100 | 0.00150187190 | 0.00550508014 |

Both states pass the declared energy/gradient consistency review. The new
complex-step formula is a separately checked analytic continuation of the
real-coordinate energy; complex steps through the unmodified native evaluator
are not assumed valid. See the full JSON for all coordinates, finite-difference
steps, power checks and provenance.

These small discrepancies concern numerical identities at tested states.
They do not establish measured physical accuracy, a quantum spectrum,
physical mass generation, or correctness of a theory of gravity.

## Findings that remain unresolved

- The complete current-dependent flux can point outward for convergence and
  inward for restoration at some faces. The strict direction requirement fails
  in these saved states even though the base directional term has the assigned
  signs. The audit records those reversals without changing or clipping them.
- Heat power differs between the two grids. No matched N=1024, s=90 state or
  corresponding preparation/reference parameters are supplied. No empirical
  convergence exponent or continuum heat value is claimed.
- The exchange coefficient was calibrated separately on the original grids.
  A fixed-parameter spatial-convergence study must control this and the other
  continuum preparation rules.
- All quantities use native model units. A physical clock, mass/energy
  calibration and independent empirical validation remain open.

## License and citation

The project code, documentation and project-owned research assets in this
release are offered under the **Apache License 2.0**, including permission for
commercial use under that license. See `LICENSE`, `NOTICE` and
`THIRD_PARTY_NOTICES.md`. No noncommercial restriction or extra citation
condition is imposed.

Please cite:

> Ashebo, Yohannes Beyene. (2026). *Ashebo RGT Hamiltonian Reference Solver and
> Numerical Audit* (Version 0.1.0). Ashebophysics.

`CITATION.cff` supplies machine-readable attribution. Source repository: https://github.com/YohannesAshebo/yohannes-ashebo-rgt-hamiltonian. No DOI has been assigned.

## Development and contributions

The research is led by Yohannes Beyene Ashebo. Implementation and audits were
developed with assistance from OpenAI ChatGPT. This does not constitute
independent external replication or peer review.

Contributions should include a reproducible change and clearly state whether
they change equations, parameters, numerical methods or only documentation.
See `CONTRIBUTING.md`. Preserve the original baseline when comparing changes.
