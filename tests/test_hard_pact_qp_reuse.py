"""Current 24-D assembly reuse; backend ownership tests live in cupiqp_pool."""
import torch
from test_go2_hard_pact_qp import make_qp, qp_data

# Discrete stance intentionally has no VJP.
LEARNED = ("tau_nom", "force_pred_world", "wrench_pred_world")

def coupled_data(batch, dtype, device="cpu"):
    data={key:value.to(device) for key,value in qp_data(batch,dtype).items()}
    data['contact_probability'].fill_(.8)
    data['foot_acceleration_bias'].fill_(.15)
    data['wrench_pred_world'][:,2]=.7
    for foot in range(4):
        data['foot_jacobians'][:,foot,:,:3]=torch.eye(3,device=device,dtype=dtype)
        data['foot_jacobians'][:,foot,:,6+3*foot:9+3*foot]=.1*torch.eye(3,device=device,dtype=dtype)
    return data

def test_changed_mechanics_refreshes_hessian_and_preserves_old_problem():
    qp=make_qp();data=coupled_data(3,torch.float64)
    first=qp._build(data);snapshot=first.Q.clone()
    data['mass_matrix']*=1.2
    second=qp._build(data)
    torch.testing.assert_close(first.Q,snapshot,rtol=0,atol=0)
    assert not torch.equal(first.Q,second.Q)
    assert second.Q.shape==(3,24,24)

def test_inference_then_replay_matches_fresh_gradients():
    qp=make_qp();data=coupled_data(2,torch.float64)
    with torch.inference_mode():qp.solve(differentiable=False,**data)
    for key in LEARNED:data[key].requires_grad_()
    outputs=[]
    for instance in (qp,make_qp()):
        result=instance.solve(differentiable=True,**data)
        assert result.differentiated_mask.all()
        loss=result.tau_safe.square().sum()+result.force_world.square().sum()*.01
        outputs.append(torch.autograd.grad(loss,[data[k] for k in LEARNED]))
    for a,b in zip(*outputs):
        torch.testing.assert_close(a,b,rtol=1e-7,atol=1e-8)
