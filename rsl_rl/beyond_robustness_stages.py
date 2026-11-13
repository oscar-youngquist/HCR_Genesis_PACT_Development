"""Physical curriculum stages, independent of PPO's optimization phase.

Stage 1 is the light box at disturbance progress zero. Stage 2 begins when
the EXISTING disturbance gate advances progress, without another clock.
Persistent external wrenches are disabled; velocity pushes remain inherited.
No liquid stage. Geometry remains creation-time. Requested changes become
physical only on resetting environments; there is no extra progression clock.
"""
from dataclasses import dataclass, asdict
import math


@dataclass(frozen=True)
class BREnvironmentStages:
    light_disturbance_scale: float = .01

    def __post_init__(self):
        if not math.isfinite(self.light_disturbance_scale) or not 0<=self.light_disturbance_scale<=1:
            raise ValueError('light_disturbance_scale must be in [0,1]')


class BRStageState:
    def __init__(self, config):
        self.config=config
        self.stage=1
        self.transitions=[]

    def update(self, iteration, disturbance_progress):
        progress=self.payload_progress(disturbance_progress)
        stage=2 if progress>0 else 1
        if stage!=self.stage:
            self.transitions.append(dict(from_stage=self.stage,to_stage=stage,iteration=int(iteration),reason='disturbance_curriculum'))
            self.stage=stage

    def payload_progress(self, disturbance_progress):
        progress=float(disturbance_progress)
        if not math.isfinite(progress) or not 0<=progress<=1:
            raise ValueError('BR disturbance progress must be in [0,1]')
        return progress

    @property
    def disturbance_scale(self):
        return self.config.light_disturbance_scale if self.stage==1 else 1.

    def state_dict(self):
        return dict(version=2,config=asdict(self.config),stage=self.stage,transitions=list(self.transitions))

    def validate(self, state):
        if state.get('version')!=2 or state['config']!=asdict(self.config) or state['stage'] not in (1,2):
            raise ValueError('Incompatible BR environment-stage contract')

    def load_state_dict(self, state):
        self.validate(state)
        self.stage=state['stage']; self.transitions=list(state['transitions'])
