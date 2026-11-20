/* Dissipate the relative phase only where the two real transport currents oppose. */
#include "radial_rhs.c"
#include "current_rhs.c"

void heat_rhs(int n, const double *state, const double *parameters,
              const double *v0, const double *area0, const double *edge_weight,
              const double *radius, const double *caps, double *work,
              double omega, double j0, double kappa, double *out) {
    quadratic_rhs(n, state, parameters, v0, area0, edge_weight, radius, caps, work, out);
    add_current_rhs(n, state, parameters, caps, omega, j0, work, out);
    double *heat = out + 8*n;
    for (int i=0; i<5; ++i) heat[i] = 0.;
    if (kappa == 0.) return;

    const double *nc=work, *nr=work+n, *volume=work+2*n, *chi=work+3*n;
    const double *mob=work+4*n, *sc=work+7*n, *sr=work+8*n, *pc=work+9*n, *pr=work+10*n;
    const double *scale=work+11*n, *weight_grad=work+13*n;
    const double *beta=state+5*n;
    const double eta=parameters[0], dr=parameters[15];
    double *paired=work+14*n;
    double core=-1.;
    if (chi[0] > 0.) {
        for (int i=0; i<n-1; ++i) {
            if (chi[i] > 0. && chi[i+1] <= 0.) {
                core=radius[i]+dr*chi[i]/(chi[i]-chi[i+1]);
                break;
            }
        }
    }
    for (int i=0; i<n-1; ++i) {
        double area=area0[i]*exp(beta[i]+beta[i+1]);
        double correction=.5*(weight_grad[i]+weight_grad[i+1])*scale[i]/(eta*area);
        double inward=mob[i]*pc[i]*(1.+correction*(1.-sc[i]));
        double outward=mob[i]*pr[i]*(1.+correction*(1.-sr[i]));
        paired[i]=fmax(0.,fmin(inward,outward));
    }
    for (int i=0; i<n; ++i) {
        double overlap=.5*((i ? paired[i-1] : 0.)+(i<n-1 ? paired[i] : 0.));
        double gate=overlap*volume[i]/(nc[i]+nr[i]);
        double gradient=eta*(nc[i]*out[i]-nr[i]*out[n+i]);
        double phase_rate=kappa*gate*gradient/volume[i];
        out[2*n+i]-=phase_rate;
        out[3*n+i]+=phase_rate;
        double q=phase_rate*gradient;
        int region=radius[i]<1.75 ? 0 : radius[i]<2.5 ? 1 : radius[i]<3. ? 2 : 3;
        heat[region]+=q;
        if (core >= 4.*dr && fabs(radius[i]-core)<=.25) heat[4]+=q;
    }
}
