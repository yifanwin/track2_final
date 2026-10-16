"""Judge reset health gate, not a score penalty or contestant action policy."""
import math
import numpy as np
from track2_judge.scenario.prototype_backend import physical_snapshot
from track2_judge.util.common import plain

class SceneInitializationRejected(RuntimeError):pass

def validate_initial_scene(env):
    original=env._after_simulation_step;frames=0;depth=0.;speed=0.;qvel=0.;nonfinite=False
    def sample():
        nonlocal frames,depth,speed,qvel,nonfinite
        original();frames+=1;state=plain(physical_snapshot(env));q=np.asarray(state['qvel']).reshape(-1)
        nonfinite|=not np.isfinite(q).all();qvel=max(qvel,float(np.max(np.abs(q[:6]))))
        for obj in state['objects'].values():
            v=np.asarray(obj['linear_velocity_m_s']);nonfinite|=not np.isfinite(v).all();speed=max(speed,float(np.linalg.norm(v)))
        for contact in state['contacts']:
            for point in contact['points']:depth=max(depth,max(0,-point['separation_m']))
    env._after_simulation_step=sample
    try:
        q=np.asarray(env.agent.robot.get_qpos()).reshape(-1);env.step(q.astype(np.float32))
    finally:env._after_simulation_step=original
    result={'scope':'judge-only reset validation','frames':frames,'max_penetration_m':depth,'max_actor_speed_m_s':speed,'max_arm_velocity_rad_s':qvel,'nonfinite':nonfinite,
            'thresholds':'candidate engineering gate: 10mm penetration, 5m/s actor, URDF5rad/s arm; approval required for production','score_change':False}
    env.initialization_health_evidence=result
    if frames!=env._sim_steps_per_control or nonfinite or depth>.01 or speed>5 or qvel>5:raise SceneInitializationRejected('candidate scene reset rejected; private evidence retained')
    return result
