# Radial Hamiltonian reference

Yohannes Beyene Ashebo · Ashebophysics · Version 0.1.0

This release covers the original radial v0.5 field model and the audited
current-dependent radial counterflow variant. The latter replaces selected
transport/exchange terms; it is not identical to every earlier Hamiltonian
variant. The molecular, gauge-lattice, 3D formation and calibrated orbital
research branches are outside this reference release.

All quantities below use native solver units. The equations document the
implemented model rather than a completed identification with physical mass,
seconds, quantum mechanics or measured gravity.

## Coordinates and energy density

For shells i=1,...,N with coordinate spacing h, centers x_i=(i-1/2)h and
faces f_i=ih, let

\[
 V_i^0=\frac{4\pi}{3}(f_i^3-f_{i-1}^3),\quad
 \ell_i=\alpha_i+2\beta_i,\quad V_i=V_i^0e^{\ell_i}.
\]

For channels nu=C,R,

\[
 \psi_{\nu i}=\sqrt{N_{\nu i}}e^{i\theta_{\nu i}},\quad
 n_{\nu i}=N_{\nu i}/V_i,\quad
 \phi_{\nu i}=\psi_{\nu i}/\sqrt{V_i},
\]
\[
 n_i=n_{Ci}+n_{Ri},\quad
 \chi_i=(n_{Ci}-n_{Ri})/n_\star,\quad
 a=(J/\lambda)^{1/3},\quad Y_i=|\psi_{Ci}-\psi_{Ri}|^2.
\]

The local potential density is

\[
 U_i=\frac g2(n_i-n_0)^2+\frac\lambda4\chi_i^4-J\chi_i
 +\frac34Ja+\frac{k_V}{2}(\ell_i-\zeta\chi_i)^2
 +\frac{k_S}{2}(\alpha_i-\beta_i)^2.
\]

The audited convention is zeta=-0.2, giving ell+0.2 chi inside the volume
square. The shifted quartic has the exact factorization

\[
 \frac\lambda4\chi^4-J\chi+\frac34Ja
 =\frac\lambda4(\chi-a)^2(\chi^2+2a\chi+3a^2),\quad J=\lambda a^3.
\]

It is nonnegative for lambda>0. Along with the stated positive coefficients
and admissible positive inventories, the audited radial energy has a lower
bound. This does not establish global dynamical stability or a continuum
existence theorem.

## Common radial energy

Let Delta_i z=z_(i+1)-z_i and W_i^0=4 pi f_i^2/h. Then

\[
 T_Q=\sum_i\left(\frac{P_{\alpha i}^2}{2I V_i^0}
                    +\frac{P_{\beta i}^2}{4I V_i^0}\right),
\]
\[
 E_{\nabla Q}=\frac{k_Q}{2}\sum_{i=1}^{N-1}W_i^0
 [(\Delta_i\alpha)^2+2(\Delta_i\beta)^2]
 +2k_Q\sum_i V_i^0\frac{(\alpha_i-\beta_i)^2}{x_i^2},
\]
\[
 E_d=\Omega_d n_\star\sum_i V_i^0\chi_i^2,\quad
 E_\Omega=\Omega\sum_iY_i,
\]
\[
 H_{\rm com}=T_Q+\sum_i V_iU_i+E_{\nabla Q}+E_d+E_\Omega.
\]

The original v0.5 model adds

\[
 E_{\nabla\phi}=\frac K2\sum_{i=1}^{N-1}
 W_i\sum_\nu|\Delta_i\phi_\nu|^2,\quad
 W_i=W_i^0e^{-\bar\alpha_i+2\bar\beta_i},
\]
\[
 E_p=\Omega_p\sum_i(\chi_i/a)^2Y_i,\quad
 H_{v0.5}=H_{\rm com}+E_{\nabla\phi}+E_p.
\]

The radial v0.5 source is `native/radial_original/hamiltonian_v05.py` and its
base `candidate_v04.py`. These files retain their original bytes.

## Current-dependent counterflow variant

Define sigmoid S(z)=1/(1+exp(-z)), face area and proper spacing

\[
 A_i=4\pi f_i^2 e^{\beta_i+\beta_{i+1}},\quad
 L_i=h e^{(\alpha_i+\alpha_{i+1})/2},\quad
 M_i=S[-(\chi_i+\chi_{i+1})/(2a)].
\]

The donor densities are n_C,i^d=n_C,i+1 and n_R,i^d=n_R,i. Channel signs
sigma_C=-1 and sigma_R=+1 are assigned by the law. Let

\[
 u_{\nu i}=\sigma_\nu\eta\Delta_i\theta_\nu/(p_0L_i),\quad
 \epsilon_{\nu i}=v_{\nu i}^{\max}p_0\log(1+e^{u_{\nu i}}).
\]

The energy contributions are

\[
 E_{\rm amp}=\frac K2\sum_i\frac{A_i}{L_i}
   \sum_\nu(\sqrt{n_{\nu,i+1}}-\sqrt{n_{\nu i}})^2,
\]
\[
 E_{\rm dir}=\sum_i M_iA_iL_i\sum_\nu n_{\nu i}^d\epsilon_{\nu i},
\]
\[
 F_{\nu i}^{(0)}=M_iA_i n_{\nu i}^d\sigma_\nu v_{\nu i}^{\max}S(u_{\nu i}),
\]
\[
 j_i^f=(|F_{Ci}^{(0)}|+|F_{Ri}^{(0)}|)/A_i,\quad
 j_i=(j_{i-1}^f+j_i^f)/2,\quad w_i=j_0/(j_i+j_0),
\]
\[
 E_x=\widehat\Omega_p\sum_i w_i(\chi_i/a)^2Y_i,\quad
 H_{\rm cf}=H_{\rm com}+E_{\rm amp}+E_{\rm dir}+E_x.
\]

Wall currents are zero. The audited run has Omega=0 and disables the old
Omega_p contribution when E_x replaces E_p. All derivatives include the
state dependence of w_i. In particular the complete face flux is
F=F^(0)+F^(x), where F^(x) is supplied by differentiating the current-weighted
exchange energy. Dropping that contribution changes the equations and the
energy identity. The complete flux can reverse sign; see the limitations.

The implementation is `native/engine/directional_radial.py`,
`native/autonomous.py` and their C kernels. Initial direction signs and
speed-cap preparation are inputs, not independently predicted physical laws.

## Evolution and the heat ledger

The conservative equations are

\[
 \dot N_{\nu i}=\eta^{-1}\frac{\partial H}{\partial\theta_{\nu i}},\quad
 \dot\theta_{\nu i}=-\eta^{-1}\frac{\partial H}{\partial N_{\nu i}},\quad
 \dot q_i=\frac{\partial H}{\partial P_{qi}},\quad
 \dot P_{qi}=-\frac{\partial H}{\partial q_i},\quad q=\alpha,\beta.
\]

The implemented numerical chart uses log inventories and unwrapped phases.
For positive inventories the local complex-coordinate identity is

\[
 i\eta\dot\psi_\nu=\partial H/\partial\psi_\nu^*.
\]

This is a local Schrödinger-type variational representation. It does not
establish a physical identification eta=hbar, a linear quantum operator or
a measured quantum spectrum. The heat adaptation adds dissipation to the
conservative core.

Using complete outward-positive fluxes, define

\[
 j_i^{\rm pair}=\max(0,\min[-F_{Ci}/A_i,F_{Ri}/A_i]),\quad
 b_i=\frac{j_{i-1}^{\rm pair}+j_i^{\rm pair}}{2(N_{Ci}+N_{Ri})},\quad
 D_i=H_{\theta_{Ci}}-H_{\theta_{Ri}}.
\]
\[
 \dot\theta_{Ci}^{\rm heat}=-\kappa b_iD_i,\quad
 \dot\theta_{Ri}^{\rm heat}=+\kappa b_iD_i,\quad
 \dot Q_i=\kappa b_iD_i^2\ge0,
\]
\[
 \dot H_{\rm heat}=-\sum_i\dot Q_i,\qquad
 \frac{d}{ds}\left(H+\sum_iQ_i\right)=0
\]

for the closed-boundary heat adaptation. The paired-current gate is
nonsmooth at branch changes. The audit uses kappa=100. Heat, energy and
solver time remain in native units.
