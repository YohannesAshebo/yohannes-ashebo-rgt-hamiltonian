# Autonomous RGT counterflow heating

This test adds a counterflow-triggered heat transfer to the archived long autonomous RGT run. It starts at the original `s=90` checkpoints, when the current-managed Hamiltonian was activated in the reference experiment. The checkpoints preserve the earlier kick and evolution. The test applies no new kick, speed, boundary supply, or time-dependent forcing. The original directional transport law and metric evolution are active. That law assigns directions and caps. The evolving state determines each current and speed within the law.

The original `s=500` experiment uses first-order canonical content and phase variables with dynamical radial geometry. The finite-packet heat test used a different solver with independent complex momenta. The present heat term is a new adaptation to the first-order Hamiltonian. The coupling values in the two tests have no established numerical mapping.

## Transfer rule

Let H be the full current-managed field and geometry Hamiltonian, V_i the proper volume of a cell, and theta_C and theta_R its two phases. At each interior face, divide both Hamiltonian transport fluxes by the proper face area and take the smaller of the inward C and outward R currents when both have the required sign. Call this nonnegative paired current j_pair. A wrong sign or a vanishing current gives zero. The phase derivative of the state-dependent exchange coefficient is included in the flux.

At cell i, let w_i be the mean of the two adjacent paired currents divided by the total proper field density. The two domain walls have zero transport. Define g_i = dH/dtheta_C,i - dH/dtheta_R,i using the complete conservative RHS. The canonical equation dN_i/ds = (dH/dtheta_i)/eta provides this gradient directly.

```
dtheta_C,i/ds (transfer) = -kappa * w_i * g_i / V_i
dtheta_R,i/ds (transfer) = +kappa * w_i * g_i / V_i
dq_heat,i/ds              =  kappa * w_i * g_i^2 / V_i
```

The extra phase-rate terms leave the common phase unchanged and directly change neither content nor metric variables. Applying the chain rule to H gives dH/ds = minus the summed heat deposition rate. The same rule is integrated together with the fields, so the extracted energy affects their subsequent evolution. The bookkeeping conserves H plus the heat ledger. The transferred energy comes from phase-dependent Hamiltonian terms, which include exchange as well as transport. The heat ledger has no pressure, gravitational source term, transport, or radiative channel.

All terms in the chosen conservative Hamiltonian are bounded below. In particular the offset makes the quartic retention potential nonnegative, and the gradient, geometry, motion, and current-weighted phase-exchange terms are nonnegative. With no external energy supply the cumulative heat is bounded by the finite available energy. The time histories can test recurring or sustained heating over a measured interval, while a constant positive heating rate forever would need an energy supply or a different heat balance.

## Measurements

- Integrate from the archived `s=90` state through `s=500` at 256 cells and through the archived fine-grid comparison endpoint `s=200` at 512 cells. Use transfer coupling 0, 1, 10 and 100 with the same archived current scale. The off run repeats the original Python early leg and the compiled continuation.
- B is the integrated signed retention inside the first outward equality crossing connected to the center. R is its areal radius. A coordinate radius shorter than four grid cells is treated as unresolved.
- At the moving boundary and at the far probe `r=3.5`, calculate signed transport speeds by dividing each field's face flux by proper face area and its donor density and interpolating to the probe. Report the mean of |u_C| + u_R over samples with both directional speeds above 10^-5 and the fraction of samples satisfying that threshold. The threshold is a diagnostic floor in model units.
- Accumulate heat in four disjoint coordinate regions with boundaries at r = 1.75, 2.5 and 3. A separate overlapping readout includes cells within 0.25 coordinate units of the instantaneous equality crossing. It is a diagnostic band, so its total is not added to the four-region energy ledger.
- Measure late deposition rates by differencing the integrated heat ledger over the final 20 time units. Instantaneous samples use the original cadence of 0.1 in the long leg. Individual fast pulses are resolution-sensitive.
- An independent finite-difference derivative audit at the two `s=90` checkpoints checks that the heat gain equals the loss in the full current-managed H. Every ten time units the C path is also compared to the original conservative RHS and to the independent Python heat expression.

## Results through s = 500

| Heat coupling | Final B | Final R | Total heat | Late mean power as percent of early mean |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 4.350 | 2.178 | 0 | — |
| 1 | 4.298 | 2.201 | 0.000592 | 0.058% |
| 10 | 4.237 | 2.159 | 0.004938 | 0.198% |
| 100 | 3.854 | 2.159 | 0.022479 | 0.146% |

The early power window is s = 90–110 and the late window is s = 480–500. The matched no-heat replay reproduces the reference at s = 110 and ends within 0.6% in B and 1.1% in R of the archived s = 500 values. Individual samples at intermediate times are more sensitive. Each coupling retains a resolved core in every late sample. At coupling 100 the late mean areal radius is 2.179, compared with 2.172 in the no-heat control. Final signed retention is 11.4% lower than the control. The integrated heat is 0.45% of the starting Hamiltonian.

At coupling 100, 0.01966 heat units (87.5%) are deposited at r < 1.75, 0.000833 in the moving boundary band, and 0.000622 in the far region. The boundary readout overlaps the fixed bins. In the last 20 time units, the boundary band averages 5.95e-7 energy per model time and the far region averages 4.03e-9. Both directional speeds exceed the 10^-5 floor in 14.4% of late samples at the boundary, with conditional mean |u_C| + u_R = 1.93e-4. At r = 3.5 neither direction pair simultaneously exceeds that floor during the same window. These data describe weak late heating and a surviving core.

## Fine-grid comparison through s = 200

Every 512-cell run also retains a resolved core in every saved sample. At coupling 100, the final B is 2.948 versus 3.214 in the fine-grid control. The deposited heat is 0.03706, of which 78.7% is at r < 1.75. Mean power over s = 180–200 is 4.51% of the s = 90–110 mean. At the same coupling and time window, the fine-grid mean power is 5.91e-5 versus 2.89e-5 at 256 cells. The common window supports the qualitative findings of a persistent core, positive but declining heating, and an inner region with the most cumulative deposition. The heating magnitude and regional shares are resolution-sensitive.

The fine-grid strongest run's final-window mean heating is 1.89e-5 near the moving core edge and 1.15e-6 in the far region. The conditional mean sum speed is 3.50e-4 at the edge and 1.29e-4 at r = 3.5. Both directional speeds are above the diagnostic floor in 4.0% of samples at each probe. The largest relative error in H plus the integrated heat across all eight runs is 1.84e-9. The largest relative total-content error is 9.23e-15.

`summary.json` contains the regional totals and early and late statistics, including the common s = 180–200 comparison. `runs/` contains the full time histories. `autonomous_heat.png` shows how retention, radius, and heat deposition evolve.

## Reproduction

Python, NumPy, SciPy and a C compiler are required. The figure also uses Matplotlib.

```sh
cc -O3 -fPIC -shared radial_rhs.c -o radial_rhs.so -lm
cc -O3 -fPIC -shared current_rhs.c -o current_rhs.so -lm
cc -O3 -fPIC -shared autonomous_heat_rhs.c -o autonomous_heat_rhs.so -lm
export PYTHONPATH=.:study:speed_order:speed_sweep OPENBLAS_NUM_THREADS=1
mkdir -p runs
for strength in 0 1 10 100; do
  python3 autonomous_heat.py 256 "$strength" 500 "runs/n256_k${strength}.json"
  python3 autonomous_heat.py 512 "$strength" 200 "runs/n512_k${strength}.json"
done
python3 report.py
```
