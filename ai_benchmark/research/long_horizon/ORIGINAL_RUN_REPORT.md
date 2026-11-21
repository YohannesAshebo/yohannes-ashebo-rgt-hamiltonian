# Longer held-out AI validation of the Ashebo counterflow solver

Yohannes Beyene Ashebo — 20 November 2026

The original Hamiltonian was preserved. Fifty native trajectories through time 10 passed their declared numerical reference checks. A newly trained hybrid that keeps the exact Hamiltonian and learns its positive dissipative mobility completed all four planned longer prediction tests. Three newly trained replacements for the entire flow failed this experiment's accuracy and rollout checks. All results, including the unsuccessful replacements, are retained.

## What was actually run

The reference is the fixed `N=128` radial discretization: 1,024 mechanical variables and the original five heat ledgers. Only the first four native ledgers are summed for total accumulated heat. Native time is not calibrated to seconds. The horizon is 0–10, with 21 stored times, compared with the earlier pilot's horizon 0–0.005. These are synthetic solutions of the stated equations.

| Split | Initial-state seed groups | Damping κ | Trajectories | Saved states |
|---|---|---|---:|---:|
| Training | 0–7 | 10, 50, 100 | 24 | 504 |
| Validation | 8, 9 | 10, 50, 100 | 6 | 126 |
| Test | 12–15 | 0, 10, 50, 100, 500 | 20 | 420 |
| Total | 14 distinct initial states | | 50 | 1,050 |

All κ variants of one initial state stay in the same split. Test seeds 12–15 were not used in the preceding pilot or short-trained neural adapter. The initial states remain perturbations of one declared manufactured radial profile; the test is not a collection of unrelated physical systems. Normalizers use training states only. Checkpoints are selected by validation loss and hashed before their own test evaluation. This is an internal held-out test, not independent laboratory validation or a test whose answers were held by an outside party.

## Reference checks

| Check | Worst result |
|---|---:|
| Relative native H + accumulated heat error, across 50 runs | 3.47 × 10⁻¹⁵ |
| Relative global total-content error, across 50 runs | 1.68 × 10⁻¹⁵ |
| Independent complex-step gradient error, scaled, three selected full-state checks | 5.89 × 10⁻¹⁶ |
| Maximum scaled state difference under native time-step halving | 1.21 × 10⁻⁸ |

The independent gradient checks cover every one of the 1,024 mechanical coordinates at each of three selected states. They use the preceding audit's independent analytic continuation of the energy, not complex perturbations passed blindly through nonanalytic native branches. They do not independently check every gradient at every saved time.

The time control repeats seed 0/κ100 and seed 12/κ500 with maximum steps 0.005 and 0.0025 at tight DOP853 tolerances. Small balance residuals and a small fixed-grid time-refinement difference are different measurements. These results do not establish spatial continuum convergence.

## The working hybrid

This is a supervised approximation to the dissipative response. The exact native H and conservative flow are retained. A 9,233-parameter DCT network learns a positive unit-κ mobility `c_nn` from eleven gauge-invariant radial features and the exact native phase torque `D = H_thetaC - H_thetaR`. Exact current and mobility are not provided at inference. A mean radial mobility profile and observed support are estimated from the training data only.

The model produces phase rates `(-κ c_nn D, +κ c_nn D)` and per-cell heat `κ c_nn D²`. Their instantaneous contribution to `dH/dt + Qdot` cancels by construction; heat is nonnegative for κ ≥ 0. The content equations remain the exact native transport equations. Global combined content is conserved; neither each cell nor each sector separately has constant content. The κ=0 limit is enforced by the known factorization, not learned as an extrapolation trick.

The hybrid was specified after the full learned-energy models' *training and validation* logs showed poor gradient learning. Its checkpoint was then chosen from validation data only and frozen before hybrid test scoring. It uses more exact physics than the full-flow learners and is not claimed to be an information-matched winner over them.

| Held-out long rollout | Final accumulated-heat error | Feature error / persistence error | Max relative native H + heat drift |
|---|---:|---:|---:|
| Seed 12, κ100 | 0.0947% | 0.000251 | 6.82 × 10⁻¹⁵ |
| Seed 12, κ500 | 0.1223% | 0.001605 | 2.19 × 10⁻¹² |
| Seed 13, κ100 | 0.5089% | 0.000287 | 3.66 × 10⁻¹⁴ |
| Seed 13, κ500 | 0.2282% | 0.001546 | 3.92 × 10⁻¹³ |

All four completed time 10 with no clipping, projection, reference-trajectory correction, or state resets. Feature error uses training-normalized gauge-invariant intensive fields and is compared against simply keeping the initial state unchanged. The error/persistence ratio is not a percentage error in every raw field.

Across these four rollouts, the worst relative total-content drift was 1.78 × 10⁻¹⁴ and the largest instantaneous dissipative power residual was 6.94 × 10⁻¹⁸. Halving the hybrid maximum step for seed 12/κ500 changed the saved state by at most 2.34 × 10⁻⁹ on the stated scaled metric. Its refined energy-plus-heat error was 2.25 × 10⁻¹⁴.

**Instantaneous heat is less accurate than its accumulated total.** Across all 20 held-out operator cases, unit-κ phase-rate relative L2 errors were 6.58–8.83%, and instantaneous total-heat-rate relative L2 errors were 6.93–19.64%. The small final integral errors do not mean each instantaneous prediction is sub-percent accurate. The four long rollouts were selected in the protocol before their scores were seen; not every one of the 20 operator cases received a free long rollout.

The model still calls the native conservative solver and native torque. No speedup is claimed.

## Full learned-flow candidates: retained unsuccessful results

Three standalone models were trained with 1,200 minibatch updates and one neural random seed each. Each uses two same-width/depth DCT backbones; parameter counts are reported rather than asserted identical. All receive the same training and validation states, split operators and energy/heat labels. These are adapted finite-grid candidates, not reproductions of the upstream authors' stock paper experiments.

| Model | Trainable parameters | Held-out conservative-flow relative L2 error | Completed RK4 long rollouts |
|---|---:|---:|---:|
| Direct DCT vector-field predictor | 18,604 | 7.93–9.03% | 0 / 20 |
| Learned H, raw phases | 18,434 | 99.98% | 0 / 20 |
| Learned H, U(1) invariant features | 18,434 | 100.03% | 0 / 20 |

The learned-H candidates use the correct extensive canonical chart, native pair ordering, fixed canonical Poisson matrix, positive rank-one friction, corrected DCT/inverse, and connected energy gradients. The U(1) model's instantaneous global content rate on held-out reference states was below 5 × 10⁻¹⁶. **That structural success did not make its learned Hamiltonian gradient accurate.** A near-constant energy fit can have a very poor derivative. The direct predictor's aggregate flow error also concealed a large content-balance defect and unstable feedback.

Both RK4 step levels failed. A separate adaptive DOP853 check with unchanged frozen weights was run for seed 12/κ100. The direct model left the declared finite-state budget near t=0.00292; the raw-phase learned-H model near t=1.768; the U(1) learned-H model near t=1.392. These diagnostics rule out treating the original RK4 step choice as the sole observed problem. They do not prove mathematical nonexistence or that every possible learned architecture must fail. The reported results concern these trained approximations and budgets.

The six earlier short-trained hybrid models were also scored, without retraining, on all 20 longer test cases. All returned finite predictions, but their temporal transfer was poor. For the earlier validation-selected structured checkpoint, phase-rate errors were 237–309% and total-heat-rate errors 440–658%. We did not switch to a different old checkpoint after examining the new tests. Longer training, not a claim that short-time success automatically transfers, was needed for the new hybrid.

## What this establishes

There is now a concrete computational research use: a reproducible labeled radial benchmark for testing learned conservative/dissipative dynamics, and a working known-H constitutive surrogate within the tested initial-state family. The native audit can distinguish accurate predictions from networks that merely satisfy their own structural identities. The original Hamiltonian and all 94 frozen source files remain unchanged, as do all six previous checkpoints.

This is numerical validation against the native discrete equations. No experimental measurements were used. It does not establish a new empirical law of gravity, transfer to unrelated physical systems, general three-dimensional dynamics, or SU(3) integration. The auxiliary heat channel in the learned GENERIC-style construction is formal energy bookkeeping at unit reference temperature; no calibrated physical entropy or temperature is inferred.

## Files and reproduction

- `SUMMARY.json`: exact headline results and scope.
- `PROTOCOL.json`, `HYBRID_PROTOCOL.json`: experiment settings and selection rules.
- `MANIFEST.json`, `data/`, `reference_runs/`: all reference trajectories, labels, hashes and accepted-step histories.
- `models/`, normalizer NPZ files, freeze JSON files: all four new trained models, including failed full-flow candidates.
- `neural_rollouts/`, `adaptive_diagnostics/`, `hybrid_rollouts/`: successful and unsuccessful rollout evidence.
- `OPERATOR_TEST_RESULTS.json`, `HYBRID_OPERATOR_RESULTS.json`, `FROZEN_SHORT_MODEL_TRANSFER.json`: all test scores.
- `validation_results.png` and `.pdf`: standalone results figure.
- `PROVENANCE.json`: source protection, protocol history and exact runtime versions.
- `SHA256SUMS`: package integrity hashes.

Use Python 3.12 with the listed NumPy/SciPy/PyTorch/Matplotlib versions and a C compiler. The tested PyTorch build was 2.15.0+cpu, installed from the official CPU package index. The original source dependency is the public repository `YohannesAshebo/yohannes-ashebo-rgt-hamiltonian`, frozen release commit `9bb33e6162eedc20c00a67123212ebe466c618ac`; exact file identities are in `dependencies/BASELINE_FILE_BLOBS.json`. No native source or upstream author notebook was edited or duplicated into this package.

Quick frozen inference from the included sample labels:

```bash
python predict_hybrid_example.py
```

Full core reproduction in a **new output directory**, leaving this frozen result intact:

```bash
python reproduce.py --release /path/to/frozen-release --output /path/to/new-reproduction
```

`verify_and_report.py` additionally checks the earlier six checkpoints and therefore takes `--previous /path/to/Ashebo_Eight_Field_Neural_Adapter_2026-11-20`. The optional earlier-model transfer script also takes the pinned authors' checkout and compatibility-loader package; these are the same dependencies recorded in that earlier artifact. No full fresh-directory retraining replication is claimed here; the included executed evidence, checkpoint reload, hash checks and fresh operator/gauge recheck were completed.

Apache-2.0 for this code and dataset packaging. No new public GitHub upload or outreach was performed in this validation.
