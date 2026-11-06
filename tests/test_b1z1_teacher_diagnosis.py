"""Check the diagnostic reproduces eligibility and resolves gain sensitivity."""
import torch
from test_b1z1_local_feedforward import local_fixture
from diagnose_b1z1_local_feedforward import describe
from rsl_rl.algorithms import b1z1_local_feedforward as local


def test_report_matches_teacher_and_lower_omega_reduces_reference_acceleration():
    a,actions,state,fixed,acc,ids,dofs,data=local_fixture()
    acc.zero_();data['local_ref_position_start'][:,0]=.1
    a.cfg['actor_phys_compliance_omega']=6.
    high=describe(a,actions,state,fixed,acc,ids,dofs,data)
    valid=local.target(a,actions,state,fixed,acc,ids,dofs,data)[4]
    assert high['valid_fraction']==float(valid.float().mean())
    a.cfg['actor_phys_compliance_omega']=1.
    low=describe(a,actions,state,fixed,acc,ids,dofs,data)
    assert high['desired_task_accel']>low['desired_task_accel']*35
    assert high['arm_acceleration_norm']>low['arm_acceleration_norm']*35


def test_damping_tradeoff_is_visible_without_changing_inputs():
    a,actions,state,fixed,acc,ids,dofs,data=local_fixture()
    acc.zero_();data['local_ref_position_start'][:,0]=.1
    before=actions.clone()
    a.cfg['actor_phys_compliance_dls_damping']=.001
    low=describe(a,actions,state,fixed,acc,ids,dofs,data)
    a.cfg['actor_phys_compliance_dls_damping']=.5
    high=describe(a,actions,state,fixed,acc,ids,dofs,data)
    assert high['task_acceleration_residual']>low['task_acceleration_residual']
    assert high['arm_acceleration_norm']<low['arm_acceleration_norm']
    torch.testing.assert_close(actions,before,atol=0,rtol=0)
