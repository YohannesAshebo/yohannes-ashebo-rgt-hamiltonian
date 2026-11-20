# Known limitations and failed requirements

Version 0.1.0 deliberately retains these findings.

## Complete flux does not preserve assigned directions

The base directional term assigns convergence inward and restoration outward.
The exchange-weight derivative adds a transport contribution. The complete
Hamiltonian flux reverses the assigned signs at some saved faces.

At the N=512, s=90 state, representative complete fluxes are approximately
+0.0497977 for convergence and -0.0129083 for restoration, with outward
defined positive. These reversals are reproduced from the compiled continuity
equations. They fail the project's strict direction requirement. No clipping,
hidden sign correction or modified Hamiltonian has been applied.

## Spatial convergence is unestablished

The saved kappa=100 heat powers are 0.00150187190 and 0.00550508014 on 256 and
512 cells at s=90. Their ratio is about 3.665. The source snapshots lack the
matched 1024-cell evolved state, its strength-matching preparation state and
reference parameters. That case is recorded as not run.

No unknown empirical order or continuum heat power is inferred from two
grids. Even three values require comparable continuum inputs, controlled time
error and evidence for an asymptotic error law. The original grids also use
slightly different strength-matched exchange coefficients.

A separate zero-integration manufactured-profile check now uses five grids
and fixed active exchange strength. It finds approximately first-order
convergence of integrated heat on its declared input families. This does not
supply the missing evolved checkpoints. The original cap preparation also
shows growing pointwise phase-torque-density maxima in the far region; the
fixed-continuum-cap control does not. Integral scaling alone does not settle
that preparation issue. See `STATIC_SCALING.md`.

Gradient and compiled-RHS verification residuals are not treated as ordinary
spatial observables. Finite-difference perturbation sweeps are saved separately.

## Scope of the reproduced checks

The new audit independently reconstructs a real-coordinate scalar energy and
checks every coordinate gradient at the two saved s=90 states. Nearby real
probes verify the analytic rewrite. These are numerical implementation checks,
not independent external replication or experiments.

The original four-row audit generator was not located. Its JSON is preserved;
the review script is explicitly labeled a reconstruction, with its own steps
and probes. The later original rows at s=200/500 have not been reconstructed
as complete field-state gradient audits by this release.

## Physical interpretation remains open

The model has no independently established physical time, mass or energy
conversion in this release. Nonnegative energy and accurate balance identities
do not prove a stable localized core, physical mass generation, an observed
quantum spectrum, a Yang-Mills mass gap, or a successful gravity prediction.

This radial code is distinct from the molecular and commercial biophysics
research branches. This release makes no assertion that those branches use
identical equations or have been validated by the present checks.

## Platform and software limits

The checked environment is Linux x86_64, Python 3.12, with the package versions
in `TESTED_ENVIRONMENT.json`. NumPy long-double precision is platform dependent;
residuals may differ elsewhere. The native precision wrapper uses an internal
SciPy integrator interface, so an untested SciPy upgrade can require review.

The audit requires a working C compiler. Optional figures require Matplotlib.
This is a research reference rather than a supported production service.
