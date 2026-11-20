/* Float64 evaluation of the archived candidate radial quadratic RHS.
 * Energy and core readouts remain with the original Python Hamiltonian.
 */
#include <math.h>
#include <stddef.h>

static double sigmoid(double value) {
    double e = exp(-fabs(value));
    return (value >= 0.0 ? 1.0 : e) / (1.0 + e);
}

static void motion_law(double z, double cap, double p0, double sign,
                       double *energy, double *velocity) {
    double e = exp(-fabs(z));
    *energy = cap * p0 * (fmax(z, 0.0) + log1p(e));
    *velocity = sign * cap * (z >= 0.0 ? 1.0 : e) / (1.0 + e);
}

void quadratic_rhs(int n, const double *y, const double *p,
                   const double *v0, const double *area0, const double *edge_weight,
                   const double *r, const double *caps, double *work, double *out) {
    const double eta=p[0], K=p[1], legacy=p[2], g=p[3], lambda=p[4], J=p[5];
    const double kv=p[6], ks=p[7], zeta=p[8], I=p[9], kg=p[10];
    const double nb=p[11], ns=p[12], od=p[13], op=p[14];
    const double dr=p[15], cs=p[16], p0=p[17];
    const double *tc=y+2*n, *tr=y+3*n, *a=y+4*n, *b=y+5*n;
    const double *pa=y+6*n, *pb=y+7*n;
    double *nc=work, *nr=work+n, *vol=work+2*n, *chi=work+3*n;
    double *ac=work+4*n, *ar=work+5*n;
    double *dnc=out, *dnr=out+n, *muc=out+2*n, *mur=out+3*n;
    double *da=out+4*n, *db=out+5*n, *ga=out+6*n, *gb=out+7*n;
    const double offset=.75*J*cs;
    for (int i=0; i<n; ++i) {
        double ell=a[i]+2*b[i], volume_ratio=exp(ell);
        vol[i]=v0[i]*volume_ratio;
        nc[i]=exp(y[i]);
        nr[i]=exp(y[n+i]);
        double rc=nc[i]/vol[i], rr=nr[i]/vol[i];
        ac[i]=sqrt(rc);
        ar[i]=sqrt(rr);
        chi[i]=(rc-rr)/ns;
        double x=chi[i], x2=x*x, density_error=rc+rr-nb;
        double strain=ell-zeta*x, shear=a[i]-b[i];
        double u=.5*g*density_error*density_error + .25*lambda*x2*x2 - J*x + offset
                 + .5*kv*strain*strain + .5*ks*shear*shear;
        double common=g*density_error, split=(lambda*x2*x-J-kv*zeta*strain)/ns;
        double pressure=rc*(common+split)+rr*(common-split)-u;
        double delta=tr[i]-tc[i], root=sqrt(nc[i]*nr[i]);
        double coupling=legacy+op*x2/(cs*cs);
        double amp_diff=sqrt(nc[i])-sqrt(nr[i]);
        double half_sine=sin(.5*delta);
        double overlap=amp_diff*amp_diff + 4*root*half_sine*half_sine;
        double ed=v0[i]*od*ns*x2, ep=(coupling-legacy)*overlap;
        double weighted=2*od*x/volume_ratio + 2*op*x*overlap/(cs*cs*vol[i]*ns);
        double cosine=cos(delta);
        muc[i]=common+split+weighted+coupling*(1-root*cosine/nc[i]);
        mur[i]=common-split-weighted+coupling*(1-root*cosine/nr[i]);
        double gamma=2*coupling*root*sin(delta)/eta;
        dnc[i]=-gamma;
        dnr[i]=gamma;
        double angular=4*kg*v0[i]*shear/(r[i]*r[i]);
        ga[i]=vol[i]*(-pressure+kv*strain+ks*shear)-2*(ed+ep)+angular;
        gb[i]=vol[i]*(-2*pressure+2*kv*strain-ks*shear)-4*(ed+ep)-angular;
        da[i]=pa[i]/(I*v0[i]);
        db[i]=pb[i]/(2*I*v0[i]);
    }
    for (int i=0; i<n-1; ++i) {
        int j=i+1;
        double A=area0[i]*exp(b[i]+b[j]), L=dr*exp(.5*(a[i]+a[j]));
        double c=ac[i]-ac[j], R=ar[i]-ar[j], square=c*c+R*R;
        double factor=.5*K*A/L;
        muc[i]+=factor*c/(vol[i]*ac[i]);
        muc[j]-=factor*c/(vol[j]*ac[j]);
        mur[i]+=factor*R/(vol[i]*ar[i]);
        mur[j]-=factor*R/(vol[j]*ar[j]);
        double left=c*ac[i]+R*ar[i], right=-c*ac[j]-R*ar[j];
        ga[i]+=factor*(-.5*square-left);
        ga[j]+=factor*(-.5*square-right);
        gb[i]+=factor*(square-2*left);
        gb[j]+=factor*(square-2*right);
        double geom_a=kg*edge_weight[i]*(a[i]-a[j]);
        double geom_b=2*kg*edge_weight[i]*(b[i]-b[j]);
        ga[i]+=geom_a;
        ga[j]-=geom_a;
        gb[i]+=geom_b;
        gb[j]-=geom_b;
        double cap_c=caps[i], cap_r=caps[n-1+i];
        if (cap_c==0.0 && cap_r==0.0) continue;
        double pc=eta*(tc[j]-tc[i])/L, pr=eta*(tr[j]-tr[i])/L;
        double ec, er, vc, vr;
        motion_law(-pc/p0, cap_c, p0, -1, &ec, &vc);
        motion_law(pr/p0, cap_r, p0, 1, &er, &vr);
        double mobility=sigmoid(-.5*(chi[i]+chi[j])/cs);
        double rc=nc[j]/vol[j], rr=nr[i]/vol[i];
        double Bc=A*L*rc, Br=A*L*rr;
        double C=Bc*ec, restorative=Br*er;
        double fc=mobility*A*rc*vc, fr=mobility*A*rr*vr;
        dnc[i]-=fc;
        dnc[j]+=fc;
        dnr[i]-=fr;
        dnr[j]+=fr;
        double chain=-(C+restorative)*mobility*(1-mobility)/cs;
        muc[i]+=chain/(2*vol[i]*ns);
        muc[j]+=(mobility*A*L*ec+chain/(2*ns))/vol[j];
        mur[i]+=(mobility*A*L*er-chain/(2*ns))/vol[i];
        mur[j]-=chain/(2*vol[j]*ns);
        double moment=.5*(Bc*pc*vc+Br*pr*vr);
        ga[i]+=mobility*(.5*(C-restorative)-moment)-.5*chain*chi[i];
        ga[j]+=mobility*(.5*(restorative-C)-moment)-.5*chain*chi[j];
        gb[i]+=mobility*(C-restorative)-chain*chi[i];
        gb[j]+=mobility*(restorative-C)-chain*chi[j];
    }
    for (int i=0; i<n; ++i) {
        dnc[i]/=nc[i];
        dnr[i]/=nr[i];
        muc[i]/=-eta;
        mur[i]/=-eta;
        ga[i]=-ga[i];
        gb[i]=-gb[i];
    }
}
