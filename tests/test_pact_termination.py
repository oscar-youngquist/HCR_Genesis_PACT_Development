"""Exercise actual Go1/Go2 termination methods without building a simulator."""
from types import SimpleNamespace as NS
import pytest
import torch


@pytest.fixture(params=['go1','go2'])
def robot(request):
    import legged_gym.envs
    if request.param=='go1':
        from legged_gym.envs.go1.go1_pact.go1_pact import Go1PACT
        cls=Go1PACT
    else:
        from legged_gym.envs.go2.go2_pact.go2_pact import Go2PACT
        cls=Go2PACT
    env=NS(cfg=NS(termination=NS(termination_terms=['roll','pitch','height_min','height_max'],
        roll_threshold=.7,pitch_threshold=1.,height_min=.2,height_max=1.5),
        env=NS(fail_to_terminal_time_s=.02),terrain=NS(reset_out_of_bounds=True)),
        dt=.01,fail_buf=torch.zeros(3,dtype=torch.long),episode_length_buf=torch.zeros(3,dtype=torch.long),
        max_episode_length=1000,non_failure_reset_buf=torch.zeros(3,dtype=torch.bool),
        simulator=NS(link_contact_forces=torch.zeros(3,2,3),termination_contact_indices=[0],
            _base_euler=torch.zeros(3,3),base_pos=torch.tensor([[0.,0.,.4]]*3),
            measured_heights=torch.zeros(3,4),_base_pos_out_of_bounds_buf=torch.zeros(3,dtype=torch.bool)))
    env.check=lambda:cls.check_termination(env)
    return env


@pytest.mark.parametrize('condition',['roll','pitch','height_min','height_max','contact'])
def test_each_failure_alone_terminates_after_consecutive_delay(robot,condition):
    e=robot;s=e.simulator
    if condition=='roll':s._base_euler[0,0]=-.8
    elif condition=='pitch':s._base_euler[0,1]=1.1
    elif condition=='height_min':s.base_pos[0,2]=.1
    elif condition=='height_max':s.base_pos[0,2]=1.6
    else:s.link_contact_forces[0,0,2]=11.
    e.check();assert e.fail_buf.tolist()==[1,0,0] and not e.reset_buf.any()
    e.check();assert e.fail_buf.tolist()==[2,0,0] and e.reset_buf.tolist()==[True,False,False]


def test_recovery_and_overlapping_conditions_count_once(robot):
    e=robot;s=e.simulator
    s._base_euler[0]=torch.tensor([.8,1.1,0.]);s.link_contact_forces[0,0,2]=11.
    e.check();assert e.fail_buf[0]==1
    s._base_euler.zero_();s.link_contact_forces.zero_()
    e.check();assert e.fail_buf[0]==0
    s.link_contact_forces[0,0,2]=11.
    e.check();assert not e.reset_buf[0] and e.fail_buf[0]==1
    # Switching failure type without a healthy step still represents failure.
    s.link_contact_forces.zero_();s._base_euler[0,0]=.8
    e.check();assert e.reset_buf[0]


@pytest.mark.parametrize('delay,steps',[(0.,1),(.025,3),(.03,3)])
def test_delay_rounding_and_zero_delay_require_failure(robot,delay,steps):
    e=robot;e.cfg.env.fail_to_terminal_time_s=delay
    e.check();assert not e.reset_buf.any()
    e.simulator._base_euler[0,0]=.8
    for _ in range(steps-1):e.check();assert not e.reset_buf.any()
    e.check();assert e.reset_buf.tolist()==[True,False,False]


def test_timeout_bounds_and_contact_link_selection(robot):
    e=robot;e.episode_length_buf[0]=1001;e.simulator._base_pos_out_of_bounds_buf[1]=True
    e.simulator.link_contact_forces[2,1,2]=100. # unselected hip link
    e.check()
    assert e.reset_buf.tolist()==[True,True,False] and e.fail_buf.tolist()==[0,0,0]
    assert e.time_out_buf.tolist()==[True,False,False]
    assert e.non_failure_reset_buf.tolist()==[False,True,False]


def test_height_is_terrain_relative_and_terms_can_be_disabled(robot):
    e=robot;e.simulator.base_pos[:,2]=2.4;e.simulator.measured_heights[:]=2.
    e.check();e.check();assert not e.reset_buf.any()
    e.cfg.termination.termination_terms=[];e.simulator._base_euler[:,0]=1.
    e.check();e.check();assert not e.reset_buf.any()
