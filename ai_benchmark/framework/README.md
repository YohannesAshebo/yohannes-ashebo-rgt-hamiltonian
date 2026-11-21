# Ashebo Hybrid

**`ashebo_hybrid` · Version 0.1.0 · Apache-2.0 · Yohannes Beyene Ashebo**

A small standalone Python framework for **known conservative mechanics plus learned nonnegative dissipation**. Bring your own energy, gradient, conservative vector field and conserved quantities. Attach a positive mobility model, integrate in explicit physical/log coordinates, and audit the predicted energy and content balances.

This package requires **no RGT checkout, native C source, CounterflowBench dataset or pretrained model**. NumPy is the only required dependency. PyTorch mobility networks and SciPy adaptive integration are optional. [CounterflowBench](../README.md) remains the separate, larger flagship dataset and recorded experiment.

## Install

From the public repository, using your own environment:

```bash
python -m pip install ./ai_benchmark/framework
python ai_benchmark/framework/examples/01_minimal_custom_system.py
```

For neural mobility and adaptive integration:

```bash
python -m pip install './ai_benchmark/framework[all]'
python ai_benchmark/framework/examples/02_ood_friction_rollout.py --output my_custom_run
```

The source package is installable; no PyPI publication is implied. Use `pip install -e ./ai_benchmark/framework` for development.

## A custom Hamiltonian in a few lines

```python
import numpy as np
from ashebo_hybrid import DeterministicBackbone, HybridSystem, ChartRK4, AuditGates

class MyHamiltonian(DeterministicBackbone):
    def energy(self, z): return z @ z / 2
    def gradient(self, z): return z
    def poisson(self, z): return np.array([[0., 1.], [-1., 0.]])

model = HybridSystem(
    backbone=MyHamiltonian(),
    mobility=lambda z: np.array([0.2]),
    directions=np.array([[0.], [1.]]),  # friction acts on momentum only
)
trajectory = ChartRK4(dt=0.005).rollout(model, np.array([1., 0.]), t_span=(0., 2.))
report = AuditGates(gradient_method="complex_step").verify(model, trajectory)
assert report["passed"], report
```

The constructor leaves your conservative equations unchanged. To learn the channel mobility, replace the callable with:

```python
from ashebo_hybrid.nn import DissipativeMobilityNet
mobility = DissipativeMobilityNet(state_dim=2, n_directions=1, width=16, layers=2)
model = HybridSystem(MyHamiltonian(), mobility, directions=np.array([[0.], [1.]]))
```

The neural head's parameters are trainable in PyTorch. A randomly initialized head is not a validated constitutive law. Example 02 trains unit-κ mobility on synthetic oscillator states, chooses weights by a separate validation set, freezes their hash, and runs an unseen κ=500 with adaptive audits. Linear dependence on κ is supplied by the architecture; it is not discovered as a zero-shot physical law.

## The contract

| Component | Caller supplies | Framework behavior |
|---|---|---|
| `DeterministicBackbone` | `energy(z)`, `gradient(z)`, and `poisson(z)` or `conservative_rhs(z)` | Preserves the supplied conservative flow; checks its energy power in audits |
| Invariants | `invariants(z)` and one row of `invariant_gradients(z)` per invariant | Projects dissipative directions onto their tangent nullspace, unless explicitly disabled |
| Dissipative directions | Finite matrix `G(z)` of shape `(dimension, rank)`, or a callable producing one | Constructs a low-rank PSD mobility; identity directions are the explicit default |
| Mobility | Nonnegative rank-vector, as a callable, or `DissipativeMobilityNet` | Rejects negative coefficients; supports a nonnegative exact off-gate |
| Coordinates | Physical/canonical initial states; optional `PositiveLogChart(positive_indices=...)` | Logs only explicitly selected positive entries and converts their rates by the chain rule |
| Integrator | Step size or tolerances and computation budgets | Float64 RK4 or adaptive DOP853, with optional state batches and recorded failures |
| `AuditGates` | Error tolerances appropriate to the problem's units | Reports predicted H+heat drift, invariant drift, power residuals and time-step-halving differences |

For energy gradient `g`, nonnegative coefficients `gamma`, directions `G` and κ≥0:

\[
\dot z=f_{\mathrm{cons}}(z)-\kappa G\,\mathrm{diag}(\gamma)G^Tg,
\qquad \dot Q=\kappa\sum_k\gamma_k(G_k^Tg)^2\ge0.
\]

When the supplied conservative mechanics obey `g · f_cons = 0`, the continuous equations obey `dH/dt + Qdot = 0`. Projected directions obey `grad C · G = 0`; the conservative flow must also preserve C. User-supplied invariant gradients must be correct. If H itself is declared an invariant of the mechanical state, projecting onto its tangent space suppresses that energy-lowering dissipation.

`extended_metric_action` exposes the auxiliary-storage PSD factorization with columns `(G_k, -G_k^T g)`. It annihilates `(g,1)` algebraically. Q is an extensive bookkeeping reservoir; this alone does not define physical temperature or entropy.

## Neural backbones and coordinates

The default neural head is an MLP. A DCT feature model is also available:

```python
net = DissipativeMobilityNet(
    state_dim=1024, n_directions=128, backend="dct",
    feature_shape=(8, 128), local_mobility=True,
    width=16, modes=16, layers=2,
)
```

Each direction column must correspond to its mobility coefficient. Provide a differentiable `feature_map` and `feature_dim` if your physical input needs invariant features. A DCT is a neural feature basis, not an automatic spherical Laplacian, wall condition, or Poisson bracket. Ordered field layouts and boundary semantics are the caller's responsibility.

For a common U(1) phase shift, `ashebo_hybrid.nn.PhaseDifferenceFeatures(state_dim, phase_indices)` retains non-phase fields and sin/cos of relative phases. Set `feature_dim` to its `output_dim` when passing it as `feature_map`. The NumPy counterpart is `ashebo_hybrid.core.common_phase_features`. These features impose the specified shift invariance; they do not identify a physical symmetry absent from the supplied model.

The integrators accept initial states `(dimension,)` or `(batch,dimension)`. A chart uses the same explicitly selected positive-coordinate indices in each batch member. It guarantees representable positive reconstructed values while within its finite chart budget; it does not establish that arbitrary dynamics stay away from zero forever. Failures are returned in the trajectory and fail the audit; values are never clipped to manufacture a pass.

The current time integrators run in NumPy/SciPy. Neural mobility training is differentiable, but these inference integrators are not claimed to provide end-to-end PyTorch adjoints through arbitrary NumPy conservative models.

## Audits and reproducibility

- **Finite steps are not exact continuous flow.** RK4/DOP853 can have nonzero H+Q and invariant drift. A coarse-step counterexample is retained in the tests. Set numerical tolerances for your physical scales and reject failing runs.
- Step halving repeats the same initial state and saved times with half the step cap. The report records whether caps were active. Two levels give a refinement indicator; they do not determine a convergence exponent or establish spatial convergence.
- Complex-step checking is optional and requires an analytic energy implementation that preserves complex inputs. Nonanalytic `abs`, clipping, branches or real casts can invalidate it. Autograd checking instead requires an explicit `backbone.torch_energy` implementation. Neither method promises a universal 10⁻¹⁶ discrepancy.
- `freeze_artifacts` records SHA-256 hashes of supplied checkpoint/normalizer files and a canonical JSON protocol. `verify_freeze` detects subsequent changes. Hashes do not establish independent blinding, external timestamps, or scientific truth.
- State-dependent skew matrices require their own Jacobi/Poisson validation. Energy and content audits do not prove that identity.

## Tested scope

The public engine tests include a damped canonical oscillator, a four-coordinate positive-content coupled-phase system with conserved total content, batching, tangent projection, PSD/degeneracy checks, exact κ=0 recovery, coarse-step rejection, budget failures, hash-change detection, MLP/DCT gradients, and a stiff adaptive case.

The separate CounterflowBench record contains 50 native reference trajectories, 4/4 planned successful known-H hybrid long rollouts, and 0/60 completed RK4 full-flow replacement tests. Those architectures use different amounts of exact physics; this is not a claim of universal neural superiority or empirical validation across domains.

When both packages are installed, `ashebo_counterflowbench.engine_bridge.create_system(native, surrogate)` adapts the native model and frozen positive-mobility network to this generic contract. Its physical state is the dataset's **canonical** `(128,8)` state flattened cell-major, with log-chart indices explicitly selected. The bridge reproduces the flagship operator and audits its global auxiliary heat; the benchmark's native rollout helper additionally records per-cell heat.

The API can be adapted to other **compatible closed systems after supplying and validating their mechanics and dissipation**. Driven robots, power networks with external work, and open reacting flows need explicit input/work/chemical bookkeeping before this closed-system H+Q audit applies. This first release does not automatically solve those domain-modeling questions.

Run tests:

```bash
python -m unittest discover -s ai_benchmark/framework/tests -v
```

## License and authorship

Apache-2.0; see [LICENSE](LICENSE), [NOTICE](NOTICE) and [CITATION.cff](CITATION.cff). Copyright 2026 Yohannes Beyene Ashebo. Implementation and audits developed with ChatGPT assistance. This framework is an additive component; none of the original 94 solver repository files are modified.
