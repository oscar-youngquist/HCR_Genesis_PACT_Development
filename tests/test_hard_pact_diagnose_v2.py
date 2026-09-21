from types import SimpleNamespace
from dataclasses import replace
import torch
from test_hard_pact_qp_diagnose import owner
from test_hard_pact_reduced_qp import inputs
from rsl_rl.algorithms.hard_pact_qp_diagnose import QPCapture, candidate_assessment, select_replay_rows
from rsl_rl.algorithms.hard_pact_qp_backends import QPBackendResult
from scripts.diagnose_hard_pact_qp import configure_diagnostic_warmup


def test_resumed_warmup_absolute_offset():
    qp=owner()
    runner=SimpleNamespace(current_learning_iteration=1100,alg=SimpleNamespace(qp_config=qp.cfg,hard_pact_qp=qp),
                           _set_hard_pact_qp_iteration=lambda i:None)
    assert configure_diagnostic_warmup(runner,2)==1102
    assert runner.alg.qp_config.warmup_iterations==1102
    assert runner.alg.hard_pact_qp.cfg.warmup_iterations==1102
    assert runner.current_learning_iteration==1100
    from rsl_rl.algorithms.ppo_hard_pact import PPO_HardPACT
    runner.alg.hard_pact_features=SimpleNamespace(execution_qp=True)
    assert [PPO_HardPACT.qp_enabled_at_iteration(runner.alg,i) for i in (1100,1101,1102)]==[False,False,True]


def packet(tmp_path,recovery=False):
    qp=owner();d=inputs(2);m=qp._build(d)
    if recovery:m=qp._soft_joint_problem(m)
    p=QPCapture(tmp_path).before(qp,m,d,"recovery" if recovery else "primary",torch.tensor([7,9]))
    return qp,d,m,p


def test_projection_gap_acceptance_and_padding(tmp_path):
    qp,d,m,p=packet(tmp_path)
    p["solver"]="cupiqp"
    z=torch.zeros(3,24,dtype=torch.float64);z[:,0]=1e10
    gap=torch.tensor([0.,1.,float("nan")]);relative=gap.clone()
    report=candidate_assessment(p,z,gap,relative)
    assert report["real_rows"]==2
    assert report["raw_primal_feasible"].tolist()==[False,False]
    assert report["post_primal_feasible"].tolist()==[True,True]
    assert report["production_accepted"].tolist()==[True,False]
    post=z[:2]*m.variable_scale
    post[:,:12]=post[:,:12].clamp(m.tau_lower,m.tau_upper)
    ok,_,_=qp._certificate(m,post/m.variable_scale,qp.cfg.ppo_feasibility_tolerance)
    assert torch.equal(report["production_accepted"],ok & (gap[:2]<=qp.cfg.ppo_duality_gap_abs))


def test_original_joint_constraints_and_recovery_slack(tmp_path):
    qp,d,m,p=packet(tmp_path,True)
    p["data"]["joint_position"][:,0]=3.
    z=torch.zeros(2,36,dtype=torch.float64);z[:,24]=2.
    r=candidate_assessment(p,z)
    joint=r["joint"]
    assert joint["empty"][:,0].all()
    assert joint["upper_family"][:,0].eq(2).all()
    assert joint["violations"]["position_rad"][:,0].eq(1).all()
    assert joint["slack_rad_s2"][:,0].eq(2*m.variable_scale[24]).all()
    assert not r["original_hard_joint_satisfied"].any()
    assert "post_projection/physical/joint_soft" in r["groups"]
    torch.testing.assert_close(joint["lower"],joint["lower_by_family"].max(-1).values)


def test_coverage_reservations_budget_counts_and_row_selection(tmp_path):
    qp,d=owner(),inputs(2);m=qp._build(d)
    capture=QPCapture(tmp_path,limit=8,healthy_limit=1)
    z=torch.zeros_like(m.p);z[0,0]=1e15
    for phase in ("rollout","ppo"):
        qp._diagnostics_phase=phase
        for stage in ("primary","recovery"):
            for _ in range(5):
                p=capture.before(qp,m,d,stage,torch.arange(2))
                capture.after(p,QPBackendResult(z),torch.tensor([False,True]))
    assert capture.count==4
    assert sum(v for k,v in capture.counts.items() if k.endswith("/attempts"))==20
    assert all(capture.coverage[phase+"/"+stage] for phase in ("rollout","ppo") for stage in ("primary","recovery"))
    saved=torch.load(next(tmp_path.glob("*.pt")),weights_only=True)
    assert select_replay_rows(saved,2)==[0,1]
    assert select_replay_rows(saved,0)==[]


def test_budget_exhaustion_still_counts_all_attempts(tmp_path):
    qp,d=owner(),inputs(2);m=qp._build(d)
    capture=QPCapture(tmp_path,byte_limit=1)
    for _ in range(3):
        p=capture.before(qp,m,d,"primary",torch.arange(2))
        capture.after(p,QPBackendResult(torch.zeros_like(m.p)),torch.tensor([False,True]))
    assert capture.counts["ppo/primary/attempts"]==3
    assert capture.counts["ppo/primary/rows"]==6
    assert capture.counts["ppo/primary/rejected_rows"]==3
    assert capture.summary()["dropped_for_budget"]==3


def test_replay_keeps_original_acceptance_and_bounded_kkt(tmp_path):
    from scripts.diagnose_hard_pact_qp import replay_one
    qp,d,m,p=packet(tmp_path)
    p["accepted"]=torch.tensor([True,False])
    p["raw_primal"]=torch.zeros_like(m.p)
    report=replay_one(p,"cpu","baseline",row=1,kkt_rows=1)
    assert report["captured_acceptance"].tolist()==[False]
    assert report["production_accepted_rows"]==1
    assert len(report["conditioning"])==1
    assert report["conditioning"][0]["row_identity"]==9


def test_runtime_observers_do_not_change_gradients(tmp_path):
    import json
    from rsl_rl.algorithms.hard_pact_qp_diagnose_runtime import diagnostic_runtime
    from rsl_rl.algorithms import ppo_hard_pact as ppo
    actor=torch.nn.Linear(1,12).double()
    qp=owner()
    runner=SimpleNamespace(current_learning_iteration=1102,
        alg=SimpleNamespace(hard_pact_qp=qp,actor_critic=actor,decoder=torch.nn.Linear(1,1)),
        _set_hard_pact_qp_iteration=lambda i:None)
    def learn(n,init_at_random_ep_len=False):
        d=inputs(2);d["tau_nom"]=actor(torch.ones(2,1,dtype=torch.float64))
        out=qp.solve(differentiable=True,**d)
        loss=ppo.projection_loss(out.tau_safe,d["tau_nom"],qp.torque_limits,
                                torch.ones(2,dtype=torch.bool),out.differentiated_mask)
        (loss+d["tau_nom"].square().mean()).backward()
    runner.learn=learn
    learn(1);expected=actor.weight.grad.clone();actor.zero_grad()
    with diagnostic_runtime(runner,tmp_path,gradients=True) as runtime:
        runtime.run_iteration("capture_overhead_included")
    torch.testing.assert_close(actor.weight.grad,expected,rtol=0,atol=0)
    report=json.loads((tmp_path/"runtime.json").read_text())
    assert report["gradient_totals"]["qp_loss/primary"]["count"]>0
    assert report["gradient_totals"]["qp_input/tau_nom"]["count"]>0
