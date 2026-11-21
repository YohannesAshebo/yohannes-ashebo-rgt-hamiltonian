"""Torch backbones; DCT is a feature basis, not a physical boundary operator."""
import math
import torch
from torch import nn
from torch.nn import functional as F

def dct_ortho(x):
    n=x.shape[-1];k=torch.arange(n,device=x.device,dtype=x.dtype);phase=torch.exp(-.5j*math.pi*k/n)
    X=torch.fft.rfft(torch.cat([x,x.flip(-1)],-1),dim=-1)[...,:n];scale=torch.full((n,),1/math.sqrt(2*n),device=x.device,dtype=x.dtype);scale[0]=1/(2*math.sqrt(n))
    return (X*phase).real*scale

def idct_ortho(X):
    n=X.shape[-1];k=torch.arange(n,device=X.device,dtype=X.dtype);a=torch.full((n,),math.sqrt(2*n),device=X.device,dtype=X.dtype);a[0]=2*math.sqrt(n)
    half=X*a*torch.exp(.5j*math.pi*k/n);return torch.fft.irfft(torch.cat([half,torch.zeros_like(half[...,:1])],-1),n=2*n,dim=-1)[...,:n]

class DCTLayer(nn.Module):
    def __init__(self,width,modes):
        super().__init__();self.modes=modes;self.weight=nn.Parameter(torch.randn(width,width,modes)/(width*width));self.local=nn.Conv1d(width,width,1)
    def forward(self,x):
        n=x.shape[-1];m=min(n,self.modes);X=dct_ortho(x);active=torch.einsum("bim,iom->bom",X[...,:m],self.weight[...,:m])
        return F.gelu(idct_ortho(F.pad(active,(0,n-m)))+self.local(x))

class DCTBackbone(nn.Module):
    def __init__(self,channels,outputs,width=16,modes=16,layers=2):
        super().__init__();self.lift=nn.Conv1d(channels,width,1);self.layers=nn.Sequential(*[DCTLayer(width,modes) for _ in range(layers)]);self.head=nn.Conv1d(width,outputs,1)
    def forward(self,x):return self.head(self.layers(self.lift(x)))
