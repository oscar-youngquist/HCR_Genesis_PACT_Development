"""Standalone CPU smoke: only PyTorch, no repo, simulator or solver imports."""
from pathlib import Path
import torch

root=Path(__file__).resolve().parent
torch.set_num_threads(2)
model=torch.jit.load(str(root/'hardpact_oct07_20000.pt'),map_location='cpu').eval()
gold=torch.load(root/'golden_vectors.pt',map_location='cpu',weights_only=True)
with torch.inference_mode():
    action,z,explicit=model(gold['obs'],gold['history'])
    for output,key in zip((action,z,explicit),('action','latent','explicit')):
        torch.testing.assert_close(output,gold[key],rtol=0,atol=0)
    grf,wrench,clipped=model.physics(z,explicit,gold['physics_tau'])
    for output,key in zip((grf,wrench,clipped),('grf_yaw','wrench_raw','wrench_clipped')):
        torch.testing.assert_close(output,gold[key],rtol=0,atol=0)
    data={k[6:]:v for k,v in gold.items() if k.startswith('input_')}
    primary=model.primary(data);recovery=model.recovery(primary)
    for prefix,problem in [('primary',primary),('recovery',recovery)]:
        for key,value in problem.items():
            torch.testing.assert_close(value,gold[prefix+'_'+key],rtol=0,atol=0)
    zero=torch.zeros_like(data['tau_nom'])
    q=data['joint_position'];v=data['joint_velocity'];one=torch.ones_like(q)
    selected=model.fallback(torch.full_like(zero,float('nan')),data['torque_limits'])
    assert torch.isfinite(selected).all() and (selected==0).all()
    _,available=model.feedforward_command(zero,zero,zero,one)
    assert not available.any()
    obs,entry=model.observation(torch.zeros(8,3),torch.zeros(8,3),torch.zeros(8,3),q,v,torch.full((8,24),200.))
    assert obs.abs().max()<=100 and entry.max()==200
    history=model.append_history(torch.zeros(8,570),entry)
    torch.testing.assert_close(history[:,-57:],entry)
    # All four execution slots independently rebuild nominal and QP inputs.
    # This tests dispatch/API use, NOT a numerical solver or physical rollout.
    for k in range(4):
        data['joint_position']=q+k*.005*v
        _,_,pd,_,nominal=model.nominal(action,data['joint_position'],v,30*one,.75*one,one,one,one,data['torque_limits'])
        data['tau_nom']=nominal
        current=model.primary(data)
        assert torch.isfinite(current['Q']).all()
        ff,available=model.feedforward_command(nominal,pd,one,one)
        torch.testing.assert_close(pd+ff,nominal,atol=2e-5,rtol=2e-5)
print('PASS: standalone load; exact golden neural/primary/recovery outputs; finite fallback; zero-gain mask; history ordering/clipping; four fresh assembly calls. No numerical QP solve.')
