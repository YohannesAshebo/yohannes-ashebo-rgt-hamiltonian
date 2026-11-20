# Ashebo Hamiltonian: audit mechanics and convergence assessment

Yohannes Beyene Ashebo — November 20, 2026

This package contains a **new reconstructed review audit**, not the original
generator of `full_hamiltonian_audit.json`. The uploaded JSON and the recovered
native autonomous-heat source/checkpoint archive are included unchanged.
No Hamiltonian coefficient, archived state or source equation is edited.

## Findings

1. Both saved s=90 states reproduce the heat powers in the uploaded audit:
   0.001501871901855448 (256 cells) and 0.0055050801410761645 (512 cells).
2. A separately written scalar energy agrees with the native energy at the
   checkpoints and nearby perturbed states to at most 4.34e-19 native units.
3. Every real-coordinate partial derivative was checked: 2048 coordinates
   at N=256 and 4096 at N=512. A valid local complex-step extension of the
   rewritten scalar energy agrees with the native analytic gradients to
   2.96e-16 and 7.28e-16 maximum absolute error, respectively.
4. A centered-difference sweep across twelve perturbation sizes shows the
   expected truncation/roundoff tradeoff. Among the declared sampled sizes,
   epsilon=1e-7 is best in both grids, with maximum sampled discrepancies
   6.57e-12 and 7.66e-12. Large-step maxima occur at the second-to-last
   alpha coordinate; they decrease with epsilon and do not persist in the
   complex-step check. This verifies derivatives of the DISCRETE energy at
   these states; it does not validate the physical boundary conditions.
5. Independent heat-only directional finite differences reach residuals
   5.55e-14 and 1.66e-14 at epsilon=1e-6. The complex-step-gradient power
   check reaches 7.24e-20 and 2.37e-19. These verify energy transfer mechanics,
   not spatial convergence or physical heating units.
6. The source archive contains no matched N=1024 s=90 state, no corresponding
   strength-matching s=5 state, and no N=1024 reference parameters. The audit
   records that case as **not run**. No interpolated, manufactured or newly
   guessed state has been appended as an evolved third-grid result.

## Exact heat operator

Let F_C and F_R be the COMPLETE outward-positive face fluxes, including the
derivative of the current-dependent exchange weight. At each interior face,

    j_pair = max(0, min(-F_C/A, F_R/A)).

At cell i, define j_bar as the arithmetic mean of the two adjacent paired
face currents (zero at domain walls), and

    b_i = j_bar_i / (N_Ci + N_Ri),
    D_i = dH/dtheta_Ci - dH/dtheta_Ri.

Then

    theta_dot_Ci(heat) = -kappa * b_i * D_i,
    theta_dot_Ri(heat) = +kappa * b_i * D_i,
    Q_dot_i           =  kappa * b_i * D_i^2,
    H_dot(heat)       = -sum_i Q_dot_i.

The recorded coupling is kappa=100. This is a **phase-gradient energy sink**,
not an assumed integral of kappa times a chemical-potential gradient squared.
The phase-energy derivative D_i is a cell-integrated partial derivative.
For a smooth state, if D_i scales as proper cell volume V_i while N_C+N_R
also scales as V_i, Q_dot_i scales as V_i; the formula alone does not force
a 1/dx or 1/dx^2 divergence. Whether this smooth regime holds in the evolved
states must be tested.

The max/min paired-current gate is nonsmooth when branches change, which
can make instantaneous power sensitive to the state and grid. The total-flux
reversals found in the prior review remain present and are recorded again.
They are not repaired by excellent energy-balance residuals.

## Gradient coordinates and complex step

The real chart is

    y = (log N_C, log N_R, theta_C, theta_R, alpha, beta, P_alpha, P_beta).

The audit differentiates H with respect to these coordinates. No division by
cell volume is made. In particular dH/d(log N)=N*dH/dN. Dividing just one side
of a derivative check by V would compare different quantities. A continuum
functional derivative requires an explicit normalization on both sides.

The unmodified native evaluator uses real casting, abs/conjugation and stable
branches. Directly injecting a complex step into it is not a valid check.
The new scalar formula first rewrites |psi_C-psi_R|^2 using real inventories
and sine terms, rewrites the fixed-sign base-current magnitudes as positive
expressions, and only then analytically continues these real-chart formulas.
Stable softplus/sigmoid branches use locally equivalent analytic expressions.
The heat gate is not included in this analytic continuation. The continuation
is checked against the original real energy at neighboring states before its
derivatives are used. The default imaginary step is 1e-30 in complex long-double
arithmetic; this does not imply that the native simulation has that accuracy.

The original audit generator was not located in the recovered archives or
the attempted filename/content searches. Therefore the original epsilon,
probe set and meaning of its `full_gradient_max_abs_error` cannot be certified
from this JSON alone. The newly measured epsilon sweeps support the numerical
differentiation explanation without claiming to recover that missing script.

## Richardson analysis

Three dyadic grids can estimate an unknown p under the model

    Q(h) = Q_infinity + C*h^p + higher-order terms.

For shrinking, same-sign differences,

    p = log2((Q_512-Q_256)/(Q_1024-Q_512)),
    Q_infinity = Q_1024 + (Q_1024-Q_512)/(2^p-1).

Three values alone do not prove the asymptotic regime or identify a physical
mechanism. With a specified p or known exact solution, two-grid analysis can
answer narrower questions; it cannot determine both unknown p and limit.

The supplied analyzer:

- Rejects duplicate grid/time/domain records rather than overwriting them.
- Flags zero/roundoff-sized differences, oscillatory sequences and p<=0.
- Excludes gradient and compiled-RHS residuals from spatial extrapolation.
- Identifies exchange strength as a per-grid CALIBRATION diagnostic.
- Flags that the original 256/512 exchange coefficients differ by 9.35e-5
  relatively. An order from such a family would describe the combined
  preparation/calibration/discretization pipeline unless those effects are
  controlled. Fixed-Hamiltonian convergence requires fixed continuum inputs.
- Does not claim finite-difference epsilon optimality from grid-level data.
- Runs meaningful arithmetic checks against a known second-order sequence,
  oscillatory and divergent sequences, roundoff ties and duplicate records.

The original two-grid data yield no empirical p or continuum estimate here.
The proposed heat-power ranges for a future third value are hypotheses, not
unique diagnoses. A third point can still reflect transient phase shifts,
gate switching, preparation changes, or mixed error sources.

## Before a defensible N=1024, s=90 comparison

Recover or reproduce the complete matched preparation, including its s=5
strength-matching state and its pre-s=90 evolution history. Keep continuum
initial profiles, boundary rules, current scale, coefficients and speed-cap
rules controlled; explicitly distinguish a frozen-parameter study from a
per-grid recalibration study. Control temporal error separately. Save the
full state and parameter provenance before evaluating heat power. Prefer
additional nearby-time/interval observables to diagnose transient sensitivity.

The recovered pressure-diagnostic replay source itself depends on an earlier
exchange-volume model and its reference trajectories; the pressure supplement
also supplies only 256/512 checkpoints. It does not supply the missing third
state. An interpolated old state can check evaluator mechanics but cannot
serve as the evolved N=1024 member of this convergence test.

## Reproduce

Requires Python 3.10+, NumPy, SciPy, and `cc`; Matplotlib is optional for figures.
Extract this package and run:

```sh
OPENBLAS_NUM_THREADS=1 python3 audit_mechanics.py
python3 analyze_convergence.py inputs/full_hamiltonian_audit.json --output results/original_grid_analysis.json
python3 analyze_convergence.py results/mechanics_audit.json --output results/recheck_grid_analysis.json
python3 make_figures.py
```

All native extraction and compilation occur in a temporary directory. No long
trajectory is integrated. The default grid request includes 1024 so the absent
inputs are recorded explicitly. `results/mechanics_audit.json` preserves the
uploaded audit under `input_audit_unchanged` and keeps new checks separately.
CSV files expose per-cell volumes, inventories, D, b and heat power. The
archive includes the native source and both matched saved states, so the
current checks are self-contained.

## Method references

- NASA NPARC, Examining Spatial (Grid) Convergence:
  https://www.grc.nasa.gov/WWW/wind/valid/tutorial/spatconv.html
- SciPy least_squares documentation, requirements for the complex-step scheme:
  https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
- The original RGT full Hamiltonian reference, November 13, 2026, and the
  supplied `full_hamiltonian_audit.json`; original source archive included.

All results are in native model units. Physical mass, clocks, spectra and
empirical heating calibration are outside this numerical audit.
