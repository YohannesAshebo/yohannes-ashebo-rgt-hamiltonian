"""Read-only native access, repeated import isolation, and an evolved reference."""
from pathlib import Path
import importlib.util
import shutil
import tempfile
import sys
import types
import unittest
import numpy as np
from ashebo_counterflowbench import load_case,verify_solver,reference_audit
from ashebo_counterflowbench.native import NativeHamiltonian

SOLVER=Path(__file__).resolve().parents[2]
AVAILABLE=bool(shutil.which("cc")) and importlib.util.find_spec("scipy") is not None and (SOLVER/"native").is_dir()

@unittest.skipUnless(AVAILABLE,"Native tests require the repository, SciPy and cc")
class NativeContractTests(unittest.TestCase):
    def test_generation_cannot_implicitly_overwrite_npz(self):
        with tempfile.TemporaryDirectory() as tmp:
            existing=Path(tmp)/"reference.npz";existing.write_bytes(b"preserve")
            with NativeHamiltonian(SOLVER) as native:
                with self.assertRaises(ValueError):native.generate(output=Path(tmp)/"reference")
                with self.assertRaises(FileExistsError):native.generate(output=existing)
            self.assertEqual(existing.read_bytes(),b"preserve")
    def test_foreign_run_module_is_preserved(self):
        original=sys.modules.get("run");dummy=types.ModuleType("run");sys.modules["run"]=dummy
        try:
            with NativeHamiltonian(SOLVER) as native:self.assertTrue(np.isfinite(native.energy(load_case().state_y[0])))
            self.assertIs(sys.modules["run"],dummy)
        finally:
            if original is None:sys.modules.pop("run",None)
            else:sys.modules["run"]=original
    def test_native_labels_and_two_clean_contexts(self):
        case=load_case("seed12_k100")
        self.assertTrue(verify_solver(SOLVER)["verified"])
        for index in [0,10]:
            with NativeHamiltonian(SOLVER) as native:
                got=native.evaluate(case.state_y[index],100)
                self.assertAlmostEqual(native.energy(case.state_y[index]),float(case["H"][index]),places=10)
                for key in ["grad_H_x","rhs_conservative_y","rhs_dissipative_unit_y","heat_cells_unit","complete_flux","gamma"]:
                    np.testing.assert_allclose(got[key],case[key][index],rtol=1e-11,atol=1e-12,err_msg=key)
                np.testing.assert_allclose(native.conservative_rhs(case.state_y[index]),case["rhs_conservative_y"][index],rtol=1e-11,atol=1e-12)
            with self.assertRaises(RuntimeError):native.energy(case.state_y[index])
        self.assertTrue(verify_solver(SOLVER)["verified"])
    def test_generated_reference_matches_included_trajectory(self):
        reference=load_case("seed12_k100")
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/"reference.npz"
            with NativeHamiltonian(SOLVER) as native:generated=native.generate(12,100,.5,2,output)
            self.assertTrue(output.is_file());np.testing.assert_allclose(generated.state_y,reference.state_y[:2],rtol=1e-8,atol=1e-9)
            audit=reference_audit(generated);self.assertLess(audit["energy_plus_heat_relative_drift"],1e-10)
            self.assertLess(audit["total_content_relative_drift"],1e-10)
        self.assertTrue(verify_solver(SOLVER)["verified"])
    @unittest.skipUnless(importlib.util.find_spec("torch") is not None,"Optional PyTorch required")
    def test_hybrid_native_coupling_preserves_original_energy(self):
        import torch
        from ashebo_counterflowbench.surrogate import load_pretrained
        torch.set_num_threads(1);c=load_case("seed12_k100");model=load_pretrained()
        with NativeHamiltonian(SOLVER) as native:
            result=native.hybrid_rollout(c.state_y[0],np.array([0.,.5]),100,model)
            H=np.array([native.energy(y) for y in result["state_y"]]);Q=result["heat_cells"].sum(1)
            self.assertLess(float(np.max(abs(H+Q-H[0]))/abs(H[0])),1e-10)
            self.assertGreater(float(Q[-1]),0)
            np.testing.assert_allclose(result["state_y"][1],c.state_y[1],rtol=0,atol=.002)
        self.assertTrue(verify_solver(SOLVER)["verified"])
    @unittest.skipUnless(importlib.util.find_spec("ashebo_hybrid") and importlib.util.find_spec("torch"),"Optional standalone framework and PyTorch")
    def test_standalone_engine_bridge_recovers_flagship_operator(self):
        import torch
        from ashebo_counterflowbench.engine_bridge import create_system
        from ashebo_counterflowbench.surrogate import load_pretrained
        torch.set_num_threads(1);c=load_case("seed12_k100");surrogate=load_pretrained();prediction=surrogate.predict_operator(c)
        with NativeHamiltonian(SOLVER) as native:
            system=create_system(native,surrogate);value=system.evaluate(c.state_x[0].ravel(),100)
        expected=prediction["rhs_y"][0].copy();expected[:,:2]*=c.state_x[0,:,:2]
        np.testing.assert_allclose(value["rhs"].reshape(128,8),expected,rtol=1e-10,atol=1e-11)
        self.assertAlmostEqual(value["heat_rate"],float(prediction["heat_cells"][0].sum()),places=12)
        self.assertLess(float(abs(value["invariant_gradients"]@value["rhs"]).max()),1e-10)

if __name__=="__main__":unittest.main()
