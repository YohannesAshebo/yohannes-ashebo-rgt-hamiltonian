"""Standalone known-mechanics/learned-dissipation engine; no RGT dependency."""
from .core import DeterministicBackbone,HybridSystem,PositiveLogChart,IdentityChart
from .integrators import ChartRK4,AdaptiveDOP853,Trajectory
from .audit import AuditGates,freeze_artifacts,verify_freeze

__version__="0.1.0"
__all__=["DeterministicBackbone","HybridSystem","PositiveLogChart","IdentityChart","ChartRK4","AdaptiveDOP853","Trajectory","AuditGates","freeze_artifacts","verify_freeze"]
