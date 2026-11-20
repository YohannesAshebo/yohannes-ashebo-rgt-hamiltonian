/* Equivalent float64 evaluation of the decreasing law in autonomous.py. */
#include <math.h>

static double logistic(double value) {
    double e = exp(-fabs(value));
    return (value >= 0.0 ? 1.0 : e) / (1.0 + e);
}

void add_current_rhs(int n, const double *y, const double *p,
                     const double *caps, double omega, double j0,
                     double *work, double *out) {
    /* Reuse base RHS scratch for contents, volume and chi. */
    const double *nc = work, *nr = work+n, *volume = work+2*n, *chi = work+3*n;
    const double *tc = y+2*n, *tr = y+3*n, *a = y+4*n;
    double *dn_c = out, *dn_r = out+n, *dt_c = out+2*n, *dt_r = out+3*n;
    double *dpa = out+6*n, *dpb = out+7*n;
    double *mob = work+4*n, *zc = work+5*n, *zr = work+6*n;
    double *sc = work+7*n, *sr = work+8*n, *pc = work+9*n, *pr = work+10*n;
    double *inv_scale = work+11*n, *j = work+12*n, *weight_grad = work+13*n;
    double eta = p[0], ns = p[12], dr = p[15], cs = p[16], p0 = p[17];

    for (int i=0; i<n-1; ++i) {
        double L = dr*exp(.5*(a[i]+a[i+1]));
        inv_scale[i] = eta/(p0*L);
        zc[i] = -(tc[i+1]-tc[i])*inv_scale[i];
        zr[i] = (tr[i+1]-tr[i])*inv_scale[i];
        sc[i] = logistic(zc[i]);
        sr[i] = logistic(zr[i]);
        mob[i] = logistic(-.5*(chi[i]+chi[i+1])/cs);
        pc[i] = caps[i]*sc[i]*nc[i+1]/volume[i+1];
        pr[i] = caps[n-1+i]*sr[i]*nr[i]/volume[i];
        j[i] = mob[i]*(pc[i]+pr[i]);
    }
    for (int i=0; i<n; ++i) {
        double current = .5*((i > 0 ? j[i-1] : 0.) + (i < n-1 ? j[i] : 0.));
        double denom = current+j0, factor = j0/denom;
        double phase = tr[i]-tc[i], root = sqrt(nc[i]*nr[i]);
        double amp = sqrt(nc[i])-sqrt(nr[i]), sh = sin(.5*phase);
        double overlap = amp*amp+4.*root*sh*sh;
        double ratio = chi[i]/cs, g = omega*ratio*ratio, energy = g*overlap;
        double density_term = 2.*omega*chi[i]*overlap/(cs*cs*volume[i]*ns);
        double cosine = cos(phase), gamma = 2.*g*root*sin(phase)*factor/eta;
        dn_c[i] -= gamma/nc[i];
        dn_r[i] += gamma/nr[i];
        dt_c[i] -= factor*(density_term+g*(1.-root*cosine/nc[i]))/eta;
        dt_r[i] -= factor*(-density_term+g*(1.-root*cosine/nr[i]))/eta;
        dpa[i] += 2.*factor*energy;
        dpb[i] += 4.*factor*energy;
        weight_grad[i] = -energy*j0/(denom*denom);
    }
    for (int i=0; i<n-1; ++i) {
        int k = i+1;
        double w = .5*(weight_grad[i]+weight_grad[k]);
        double chi_grad = -mob[i]*(1.-mob[i])*(pc[i]+pr[i])/(2.*cs);
        double c_grad = -w*mob[i]*pc[i]*(1.-sc[i])*inv_scale[i];
        double r_grad = w*mob[i]*pr[i]*(1.-sr[i])*inv_scale[i];
        double geom = .5*w*mob[i]*(pc[i]*(1.-sc[i])*zc[i]+pr[i]*(1.-sr[i])*zr[i]);
        double c_donor = w*mob[i]*pc[i], r_donor = w*mob[i]*pr[i];
        dn_c[i] -= c_grad/(eta*nc[i]);
        dn_c[k] += c_grad/(eta*nc[k]);
        dn_r[i] -= r_grad/(eta*nr[i]);
        dn_r[k] += r_grad/(eta*nr[k]);
        dt_c[i] -= w*chi_grad/(ns*volume[i]*eta);
        dt_r[k] += w*chi_grad/(ns*volume[k]*eta);
        dt_c[k] -= (w*chi_grad/(ns*volume[k])+c_donor/nc[k])/eta;
        dt_r[i] -= (-w*chi_grad/(ns*volume[i])+r_donor/nr[i])/eta;
        double left = w*chi_grad*chi[i]+r_donor, right = w*chi_grad*chi[k]+c_donor;
        dpa[i] += left+geom;
        dpa[k] += right+geom;
        dpb[i] += 2.*left;
        dpb[k] += 2.*right;
    }
}
