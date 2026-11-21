"""Scientific contract checks for charts, splits, physical audits and model loading."""
import json
import pickle
from pathlib import Path
import tempfile
import unittest
import numpy as np
from ashebo_counterflowbench import load_case,load_split,dataset_index,verify_dataset,verify_solver,grade_operator,grade_rollout,reference_audit
from ashebo_counterflowbench.data import RESOURCES

class DataAndAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.case=load_case("seed12_k100")
    def test_all_hashes_and_grouped_splits(self):
        result=verify_dataset();self.assertEqual(result["counts"],dict(train=24,validation=6,test=20));self.assertEqual(result["saved_states"],dict(train=504,validation=126,test=420))
        self.assertEqual(result["groups"]["test"],[12,13,14,15])
    def test_readonly_and_explicit_chart(self):
        c=self.case;self.assertFalse(c.state_y.flags.writeable);self.assertTrue(np.all(c.state_x[...,:2]>0));np.testing.assert_allclose(np.log(c.state_x[...,:2]),c.state_y[...,:2],rtol=1e-14,atol=5e-16)
    def test_canonical_pairs_recover_reversible_rhs(self):
        c=self.case;g=c["grad_H_x"];flow=np.stack([g[...,2],g[...,3],-g[...,0],-g[...,1],g[...,6],g[...,7],-g[...,4],-g[...,5]],-1)
        flow[...,:2]/=c.state_x[...,:2]
        np.testing.assert_allclose(flow,c["rhs_conservative_y"],rtol=1e-12,atol=1e-12)
    def test_reference_power_and_local_content(self):
        c=self.case;a=reference_audit(c);self.assertLess(a["energy_plus_heat_relative_drift"],1e-11);self.assertLess(a["instantaneous_power_residual_max_abs"],1e-10)
        g=grade_operator(c,c.rhs_y,c.heat_power);self.assertEqual(g["rhs_relative_L2"],0);self.assertLess(g["local_reference_content_balance_max_abs"],1e-10)
    def test_fifth_ledger_is_not_total_heat(self):
        c=self.case;np.testing.assert_array_equal(c.accumulated_heat,c["heat_ledgers"][:,:4].sum(1))
        self.assertGreater(float(np.max(abs(c["heat_ledgers"][:,4]))),0)
    def test_bad_content_and_negative_heat_are_visible(self):
        c=self.case;rhs=c.rhs_y.copy();rhs[:,20,0]+=.01;q=-np.ones_like(c.heat_power)
        score=grade_operator(c,rhs,q);self.assertEqual(score["negative_heat_fraction"],1);self.assertGreater(score["global_content_rate_max_abs"],.001)
        self.assertGreater(score["local_reference_content_balance_max_abs"],.001)
    def test_invalid_shape_and_nonfinite_are_rejected(self):
        with self.assertRaises(ValueError):grade_operator(self.case,self.case.rhs_y[...,0],self.case.heat_power)
        bad=self.case.rhs_y.copy();bad[0,0,0]=np.nan
        with self.assertRaises(ValueError):grade_operator(self.case,bad,self.case.heat_power)
    def test_rollout_recomputes_H_on_predicted_states(self):
        c=self.case;calls=[];y=c.state_y.copy();y[-1,:,4]+=.01;Q=np.zeros_like(c.heat_power)
        def evaluator(state):calls.append(state.copy());return float(c["H"][0])+float(state[:,4].sum())
        grade_rollout(c,y,Q,evaluator);self.assertEqual(len(calls),21);np.testing.assert_array_equal(calls[-1],y[-1])
        with self.assertRaises(TypeError):grade_rollout(c,y,Q,c["H"])
    def test_arbitrary_case_names_do_not_escape_dataset(self):
        with self.assertRaises(KeyError):load_case("../../README")
    def test_trajectory_can_cross_spawn_worker_boundary(self):
        copied=pickle.loads(pickle.dumps(self.case));np.testing.assert_array_equal(copied.rhs_y,self.case.rhs_y);self.assertFalse(copied.state_y.flags.writeable)

try:import torch
except ImportError:torch=None

@unittest.skipIf(torch is None,"Optional PyTorch dependency not installed")
class TorchInterfaceTests(unittest.TestCase):
    def test_pretrained_outputs_match_frozen_validation(self):
        from ashebo_counterflowbench.surrogate import load_pretrained
        torch.set_num_threads(1);rng=torch.random.get_rng_state().clone();dtype=torch.get_default_dtype();model=load_pretrained()
        self.assertTrue(torch.equal(rng,torch.random.get_rng_state()));self.assertEqual(dtype,torch.get_default_dtype())
        c=load_case("seed12_k100");prediction=model.predict_operator(c);score=grade_operator(c,prediction["rhs_y"],prediction["heat_cells"])
        expected=next(row for row in json.loads((RESOURCES/"evidence/HYBRID_OPERATOR_RESULTS.json").read_text()) if row["seed_group"]==12 and row["kappa"]==100)
        self.assertAlmostEqual(score["phase_relaxation_residual_relative_L2"],expected["unit_phase_rate_relative_L2"],places=11)
        self.assertAlmostEqual(score["total_heat_rate_relative_L2"],expected["unit_total_heat_relative_L2"],places=12)
        self.assertEqual(score["negative_heat_fraction"],0);self.assertLess(score["native_H_power_residual_max_abs"],1e-10)
    def test_pretrained_gauge_and_disabled_limit(self):
        from ashebo_counterflowbench.surrogate import load_pretrained
        model=load_pretrained();c=load_case("seed12_k100");y=c.state_y.copy();y[...,2:4]+=.71
        np.testing.assert_allclose(model.predict_unit(y,c.torque),model.predict_unit(c.state_y,c.torque),rtol=1e-10,atol=1e-14)
        zero=load_case("seed12_k0");pred=model.predict_operator(zero);np.testing.assert_array_equal(pred["rhs_y"],zero["rhs_conservative_y"]);self.assertTrue(np.all(pred["heat_cells"]==0))
    def test_tensor_and_window_layouts(self):
        from ashebo_counterflowbench.torch_data import OperatorDataset,WindowDataset
        train=OperatorDataset("train");self.assertEqual(len(train),504);self.assertEqual(tuple(train[0]["state_y"].shape),(8,128));self.assertEqual(train[0]["state_y"].dtype,torch.float64)
        window=WindowDataset("test",11);self.assertEqual(len(window),220);self.assertEqual(tuple(window[-1]["state_y"].shape),(11,8,128));self.assertIn(window[-1]["seed_group"],[12,13,14,15])
        with self.assertRaises(ValueError):WindowDataset("test",22)

if __name__=="__main__":unittest.main()
