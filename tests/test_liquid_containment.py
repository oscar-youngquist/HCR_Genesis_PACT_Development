"""Guard the experimental contact isolation and acceptance criteria."""
from types import SimpleNamespace as NS
import pytest
from liquid_collection.containment import PROFILES, verify_cavity_contacts
from liquid_collection.payload import validate_payload
from liquid_collection.validation.run import payloads


def test_profiles_preserve_payload_spawn_clearance():
    for profile in PROFILES.values():
        for payload in payloads():
            validate_payload(payload, .1, profile['particle_size'], .001,
                             property_expansion=.10, geometry_expansion=.02)


def test_plane_contacts_are_massless_and_isolated_from_rigid_pairs():
    geoms = [NS(idx=i, needs_coup=i>=6, contype=(1<<30 if i>=6 else 1),
                conaffinity=(0 if i>=6 else 65535)) for i in range(12)]
    entities = [NS(geoms=[g], get_mass=lambda i=i: 1. if i<6 else 0.)
                for i,g in enumerate(geoms)]
    sim=NS(liquid_attachment_targets=[(e,[0,0,0]) for e in entities],
           liquid_contact_parts=[dict(size=[1,1,1],rho=1.) for _ in range(6)],
           _cfg=NS(liquid=NS(cavity_coupling='planes')),
           _scene=NS(sim=NS(rigid_solver=NS(geoms=geoms))))
    assert verify_cavity_contacts(sim)['actual_dry_mass_kg']==6.
    geoms[0].conaffinity=1<<30
    with pytest.raises(ValueError,match='filter bit'):
        verify_cavity_contacts(sim)
    geoms[0].conaffinity=65535
    geoms[0].needs_coup=True
    with pytest.raises(ValueError,match='double counted'):
        verify_cavity_contacts(sim)
    geoms[0].needs_coup=False
    entities[-1].get_mass=lambda: .1
    with pytest.raises(ValueError,match='shell mass'):
        verify_cavity_contacts(sim)


def test_full_duration_requires_every_environment(tmp_path, monkeypatch):
    import numpy as np
    from liquid_collection import containment
    from liquid_collection.storage import atomic_json
    atomic_json(tmp_path/'metrics.json',dict(validity_counts=dict(
        finite=0, contained=0, all_active=0, attached=0, domain_contact=0)))
    row=dict(time_s=11.,record_sequence=1,
             interval_max_wall_penetration_m=np.zeros(50),
             terminal=np.ones(50,dtype=bool),episode_selected=np.ones(50,dtype=bool),
             episode_native_step=np.full(50,22000),interval_contained=np.ones(50,dtype=bool))
    monkeypatch.setattr(containment,'stream_rows',lambda path: [row] if path==tmp_path else [])
    job=dict(profile='planes_8mm_4', selected=dict(policy=dict(robot='go2')),
             payload_label='high_fill',episode_seconds=11.,num_envs=50,
             particle_size=.008,substeps=4,baseline=False)
    assert containment.summarize(tmp_path,job,1.)['passed']
    row['terminal'][-1]=False
    assert not containment.summarize(tmp_path,job,1.)['passed']
    row['terminal'][-1]=True
    job['episode_seconds']=1.
    row['episode_native_step'][:]=2000
    assert not containment.summarize(tmp_path,job,1.)['passed']


def test_timing_probe_does_not_claim_full_duration(tmp_path, monkeypatch):
    import numpy as np
    from liquid_collection import containment
    from liquid_collection.storage import atomic_json
    atomic_json(tmp_path/'metrics.json',dict(steps=20,validity_counts=dict(
        finite=0, contained=0, all_active=0, attached=0, domain_contact=0)))
    row=dict(time_s=.2,record_sequence=1,interval_max_wall_penetration_m=np.zeros(2),
             terminal=np.zeros(2,dtype=bool),episode_selected=np.ones(2,dtype=bool),
             episode_native_step=np.full(2,400),interval_contained=np.ones(2,dtype=bool))
    monkeypatch.setattr(containment,'stream_rows',lambda path: [row] if path==tmp_path else [])
    job=dict(profile='planes_10mm_4',selected=dict(policy=dict(robot='go1')),
             payload_label='high_fill',episode_seconds=11.,num_envs=2,
             particle_size=.010,substeps=4,baseline=False,scaling_probe=True,max_steps=20)
    result=containment.summarize(tmp_path,job,1.)
    assert result['timing_probe_completed']
    assert not result['passed']
