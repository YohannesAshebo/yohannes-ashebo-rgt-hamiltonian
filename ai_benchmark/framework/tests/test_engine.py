import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from ashebo_hybrid import DeterministicBackbone,HybridSystem,PositiveLogChart,ChartRK4,AdaptiveDOP853,AuditGates,freeze_artifacts,verify_freeze
from ashebo_hybrid.core import factor_rates,extended_metric_action

class Oscillator(DeterministicBackbone):
    def energy(self,z):return z@z/2
    def gradient(self,z):return z
    def poisson(self,z):return np.array([[0.,1.],[-1.,0.]])

class CoupledContents(DeterministicBackbone):
    def energy(self,z):return (z[0]**2+z[1]**2)/2+.3*(1-np.cos(z[2]-z[3]))
    def gradient(self,z):
        phase=.3*np.sin(z[2]-z[3]);return np.array([z[0],z[1],phase,-phase])
    def poisson(self,z):return np.array([[0.,0.,1.,0.],[0.,0.,0.,1.],[-1.,0.,0.,0.],[0.,-1.,0.,0.]])
    def invariants(self,z):return np.array([z[0]+z[1]])
    def invariant_gradients(self,z):return np.array([[1.,1.,0.,0.]])

class EngineTests(unittest.TestCase):
    def oscillator(self):return HybridSystem(Oscillator(),lambda z:np.array([.2]),directions=np.array([[0.],[1.]]))
    def test_rank_K_psd_and_energy_degeneracy(self):
        rng=np.random.default_rng(0);G=rng.normal(size=(8,3));gamma=np.array([.1,.3,.8]);g=rng.normal(size=8)
        columns=[]
        for i in range(9):
            vz,vQ=extended_metric_action(np.eye(9)[:8,i],np.eye(9)[8,i],g,G,gamma);columns.append(np.concatenate([vz,[vQ]]))
        M=np.column_stack(columns)
        np.testing.assert_allclose(M,M.T,atol=1e-13);self.assertGreaterEqual(np.linalg.eigvalsh(M).min(),-1e-12)
        np.testing.assert_allclose(M@np.r_[g,1],0,atol=1e-12)
        f,Q,_=factor_rates(g,G,gamma);self.assertAlmostEqual(float(g@f+Q),0,places=12);self.assertGreaterEqual(Q,0)
    def test_negative_mobility_rejected(self):
        with self.assertRaises(ValueError):factor_rates(np.ones(2),np.eye(2),np.array([1.,-1.]))
    def test_rk4_batch_and_gradient_audit(self):
        model=self.oscillator();tr=ChartRK4(.005).rollout(model,np.array([[1.,0.],[.5,.2]]),t_span=(0.,1.))
        self.assertEqual(tr.states.shape[1:],(2,2));self.assertEqual(tr.states.dtype,np.float64)
        report=AuditGates(gradient_method="complex_step").verify(model,tr);self.assertTrue(report["passed"],report)
    def test_positive_chart_and_content_tangency(self):
        directions=np.array([[1.,0.],[0.,0.],[0.,1.],[0.,-1.]])
        model=HybridSystem(CoupledContents(),lambda z:np.array([.05,.1]),directions=directions,chart=PositiveLogChart((0,1)))
        first=model.evaluate(np.array([.8,1.2,.2,-.1]));np.testing.assert_allclose(first["invariant_gradients"]@first["directions"],0,atol=1e-15)
        tr=ChartRK4(.005).rollout(model,np.array([.8,1.2,.2,-.1]),t_span=(0.,2.));self.assertTrue(np.all(tr.states[:,:2]>0))
        report=AuditGates(gradient_method="complex_step").verify(model,tr);self.assertTrue(report["passed"],report)
    def test_disabled_limit_does_not_call_network(self):
        def disabled(z):raise AssertionError("Network should not be called at kappa0")
        model=HybridSystem(Oscillator(),disabled)
        np.testing.assert_array_equal(model.evaluate(np.array([1.,2.]),0)["rhs"],np.array([2.,-1.]))
    def test_coarse_step_cannot_claim_exact_energy(self):
        model=self.oscillator();tr=ChartRK4(1.).rollout(model,np.array([1.,0.]),t_span=(0.,2.))
        report=AuditGates().verify(model,tr);self.assertFalse(report["passed"]);self.assertGreater(report["energy_heat_relative_drift"],1e-5)
    def test_budget_failure_is_reported(self):
        model=self.oscillator();tr=ChartRK4(.01,max_steps=1).rollout(model,np.array([1.,0.]),t_span=(0.,1.))
        self.assertFalse(tr.completed);self.assertEqual(tr.failure["type"],"TimeoutError");self.assertFalse(AuditGates().verify(model,tr)["passed"])
    def test_wrong_conservative_flow_is_not_repaired(self):
        class Bad(Oscillator):
            def conservative_rhs(self,z):return super().conservative_rhs(z)+np.array([.1,0.])
        model=HybridSystem(Bad(),lambda z:np.zeros(2));tr=ChartRK4(.01).rollout(model,np.array([1.,0.]),t_span=(0.,.1))
        self.assertFalse(AuditGates().verify(model,tr)["passed"])
    def test_freeze_detects_weight_and_protocol_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);w=p/"weights.bin";w.write_bytes(b"before");freeze_artifacts({"weights":w},{"kappa":500},p/"freeze.json")
            self.assertTrue(verify_freeze(p/"freeze.json")["verified"]);w.write_bytes(b"after");self.assertFalse(verify_freeze(p/"freeze.json")["verified"])
    @unittest.skipUnless(importlib.util.find_spec("scipy"),"Optional SciPy")
    def test_adaptive_stiff_case_with_step_control(self):
        model=self.oscillator();tr=AdaptiveDOP853(.01,rtol=1e-10,atol=1e-12).rollout(model,np.array([1.,0.]),kappa=500,times=np.linspace(0,1,11))
        report=AuditGates().verify(model,tr);self.assertTrue(report["passed"],report)

@unittest.skipUnless(importlib.util.find_spec("torch"),"Optional PyTorch")
class NeuralTests(unittest.TestCase):
    def test_independent_autograd_gradient_check(self):
        import torch
        from ashebo_hybrid.audit import gradient_check
        class TorchOscillator(Oscillator):
            def torch_energy(self,z):return (z*z).sum()/2
        result=gradient_check(TorchOscillator(),np.array([1.,.2]),method="autograd");self.assertLess(result["max_scaled_error"],1e-14)
    def test_positive_trainable_mlp(self):
        import torch
        from ashebo_hybrid.nn import DissipativeMobilityNet
        torch.set_num_threads(1);model=DissipativeMobilityNet(4,n_directions=2);z=torch.randn(3,4,dtype=torch.float64,requires_grad=True);gamma=model(z)
        self.assertEqual(tuple(gamma.shape),(3,2));self.assertTrue(torch.all(gamma>=0));gamma.sum().backward();self.assertTrue(torch.isfinite(z.grad).all())
    def test_dct_roundtrip_and_local_shape(self):
        import torch
        from ashebo_hybrid.nn import DissipativeMobilityNet,dct_ortho,idct_ortho
        z=torch.randn(2,16,dtype=torch.float64);torch.testing.assert_close(idct_ortho(dct_ortho(z)),z,rtol=1e-12,atol=1e-12)
        model=DissipativeMobilityNet(16,n_directions=8,backend="dct",feature_shape=(2,8),local_mobility=True,width=4,modes=4)
        self.assertEqual(tuple(model(z).shape),(2,8));self.assertTrue(torch.all(model(z)>=0))
    def test_common_phase_features_are_shift_invariant(self):
        import torch
        from ashebo_hybrid.core import common_phase_features
        from ashebo_hybrid.nn import PhaseDifferenceFeatures,DissipativeMobilityNet
        x=np.array([[.8,1.2,.2,-.1]]);shifted=x.copy();shifted[:,2:]+=1.3
        np.testing.assert_allclose(common_phase_features(x,(2,3)),common_phase_features(shifted,(2,3)),atol=1e-15)
        mapping=PhaseDifferenceFeatures(4,(2,3));model=DissipativeMobilityNet(4,n_directions=1,feature_map=mapping,feature_dim=mapping.output_dim)
        torch.testing.assert_close(model(torch.tensor(x)),model(torch.tensor(shifted)),rtol=1e-12,atol=1e-12)

if __name__=="__main__":unittest.main()
