"""Change domain extent only; preserve bulk law, calibration and inner data."""
from dataclasses import replace
import numpy as np
from common import DirectionalRadial,LD
from directional_radial import BASE,sigmoid
from formation import initial

def model(radius,cells):
    m=DirectionalRadial(cells,parameters=replace(BASE,domain_radius=radius));y=initial(m,0)
    q=m.unpack(y);native=m.original.terms_and_gradient(m.to_original(y))
    pC=m.p.eta*np.diff(q[2])/m.dr;pR=m.p.eta*np.diff(q[3])/m.dr
    j=int(np.argmin(abs(m.faces[1:-1]-1)));m.p0=abs(pC[j])
    rho=np.exp(q[:2])/m.V0;chi=(rho[0]-rho[1])/m.p.n_star
    mob=sigmoid(-(chi[:-1]+chi[1:])/(2*m.cs))
    m.vmax=np.stack([abs(native['flux'][0])/(m.A0*rho[0,1:]*mob*sigmoid(-pC/m.p0)),abs(native['flux'][1])/(m.A0*rho[1,:-1]*mob*sigmoid(pR/m.p0))])
    return m,y
