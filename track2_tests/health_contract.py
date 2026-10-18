"""Synthetic health contracts + actual server hook/privacy/noninterference checks."""
import copy,json,hashlib,sys
from pathlib import Path
SOURCE=Path(sys.argv[1]).resolve(strict=True);OUT=Path(sys.argv[2]).resolve()
sys.path.insert(0,str(SOURCE));OUT.mkdir(parents=True,exist_ok=False)
from track2_judge.diagnostics.physics_health import TrustedPhysicsHealthRecorder
def state(v=(0,0,0),contact=False,name='cube'):
    return {'trusted_physics_substep':True,'objects':{name:{'actor_p_m':[0,0,1],'linear_velocity_m_s':list(v)}},'qpos':[0]*8,'qvel':[0]*8,'contacts':[{'bodies':[name,'table'],'points':[{'separation_m':-.002,'impulse_Ns':[0,0,.1]}]}] if contact else []}
results=[]
def test(name,fn):
    try:detail=fn();results.append({'name':name,'status':'PASS','detail':detail})
    except Exception as e:results.append({'name':name,'status':'FAIL','error':type(e).__name__,'message':str(e)[:150]})
def ballistic():
    r=TrustedPhysicsHealthRecorder(dt=1/240);r.sample(state((0,0,-231)));r.sample(state((0,0,-231-9.81/240)))
    p=r.report(2);assert 'HIGH_SPEED_GRAVITY_COMPATIBLE' in p['first_flags'] and 'CONTACT_FREE_ACCELERATION_ALERT' not in p['first_flags'];return p['status']
test('high_speed_not_mislabeled_divergence',ballistic)
def nonfinite():
    r=TrustedPhysicsHealthRecorder(dt=1/240);s=state();s['qpos'][0]=float('nan');r.sample(s);p=r.report(1)
    assert p['status']=='NONFINITE_PHYSICS';json.dumps(r.private_evidence(),allow_nan=False);return 'captured before strict JSON rejection'
test('nonfinite_private_evidence_serializable',nonfinite)
def attribution():
    r=TrustedPhysicsHealthRecorder(dt=.01);r.sample(state((0,0,-1)));r.sample(state((0,0,-20)));assert 'CONTACT_FREE_ACCELERATION_ALERT' in r.report(2)['first_flags']
    return 'alert only; no score or state mutation'
test('noncontact_acceleration_anomaly',attribution)
def sleeping():
    r=TrustedPhysicsHealthRecorder(dt=.01);r.sample(state(name='NONFINITE_label'));r.sample(state(name='NONFINITE_label'));assert not r.report(2)['flag_counts'];return 'stationary/no contact and labels do not create false alerts'
test('sleeping_static_and_name_false_positive',sleeping)
def bounds():
    r=TrustedPhysicsHealthRecorder(dt=.01,support_xy_bounds=[[-.1,-.1],[.1,.1]],evidence_frames=4)
    for i in range(20):s=state(contact=True);s['objects']['cube']['actor_p_m']=[1,0,1];r.sample(s)
    assert 'OUT_OF_SUPPORT_XY' in r.report(20)['first_flags'] and len(r.ring)==4
    assert r.report(21)['coverage']=='MISSING_OR_INCOMPLETE';return 'explicit actor-point membership and bounded evidence'
test('private_support_and_coverage',bounds)
def trust_hash():
    a=TrustedPhysicsHealthRecorder(dt=.01);b=TrustedPhysicsHealthRecorder(dt=.01);s=state(contact=True)
    a.sample(s,command={'gripper_opening_m':.01});b.sample(s,command={'gripper_opening_m':.02});assert a.report(1)['chain_sha256']!=b.report(1)['chain_sha256']
    try:a.sample({'objects':{}})
    except ValueError:pass
    else:raise AssertionError('untrusted sample accepted')
    return 'command+state bound; untrusted samples rejected'
test('trusted_hash_command_binding',trust_hash)
def contact_velocity():
    r=TrustedPhysicsHealthRecorder(dt=.01);s=state(contact=True);s['contacts'][0]['points'][0]={'separation_m':-.022,'impulse_Ns':[0,0,8]};s['qvel'][0]=17;s['judge_control']={'declared_velocity_limits':[5]*6+[3]*2};r.sample(s)
    assert {'DEEP_CONTACT_ALERT','CONTACT_IMPULSE_ALERT','DECLARED_JOINT_VELOCITY_EXCEEDED'}<=set(r.report(1)['first_flags']);return 'separate contact/physical joint velocity evidence'
test('contact_and_joint_velocity_alerts',contact_velocity)
def real_hook():
    import numpy as np
    from track2_judge.scenario.prototype_env_v2 import make_env
    from track2_judge.scenario.prototype_backend import physical_snapshot
    from track2_judge.toolapi.dispatch import ManiSkillSimulatorAdapter
    # 2026-10-18：健康采样降到 10 Hz（每 24 子步 = 2 个控制步）。原 max_steps=2
    #   只有 24 子步、恰好 1 次采样 —— 会让下面的 frames 断言退化成「只验 1 个样本」。
    #   提到 5 步（60 子步 → 2 次采样）才验得到节流本身。
    a=ManiSkillSimulatorAdapter(make_env,snapshot=physical_snapshot,max_steps=5)
    b=ManiSkillSimulatorAdapter(make_env,snapshot=physical_snapshot,max_steps=5,physics_health=False)
    try:
        oa=a.reset(seed=0);ob=b.reset(seed=0);assert oa==ob
        assert not any('health' in k.lower() for k in oa)
        q=oa['robot']['joint_position'];packet={'joint_position':q[:6],'gripper_opening_m':q[6]-q[7]}
        for _ in range(5):
            a.step(packet);b.step(packet)
        sa=a.get_final_state();sb=b.get_final_state();differences=[]
        for key in ('qpos','qvel','tcp_p_m','tcp_q_wxyz'):
            differences.append(float(np.max(np.abs(np.asarray(sa[key])-np.asarray(sb[key])))))
        assert set(sa['objects'])==set(sb['objects'])
        for name in sa['objects']:
            for key in ('actor_p_m','q_wxyz','linear_velocity_m_s','angular_velocity_rad_s'):
                differences.append(float(np.max(np.abs(np.asarray(sa['objects'][name][key])-np.asarray(sb['objects'][name][key])))))
        assert max(differences)<1e-7,('health observation altered PhysX state',max(differences))
        # 10 Hz：60 子步 → 2 次采样。改的是**密度断言**，不是判定断言。
        h=a.get_physics_health();assert h['coverage']=='COMPLETE' and h['frames']==2 and h['declared_velocity_limits_checked']
        for method in ('get_physics_health','get_physics_health_private_evidence'):
            try:a.dispatch_public(method,{})
            except ValueError:pass
            else:raise AssertionError('judge health exposed through Skill API')
        return {'actual_physx_substeps':60,'physical_state_max_abs_difference':max(differences),'health_RPC':'rejected'}
    finally:a.close();b.close()
test('actual_PhysX_observer_noninterference_privacy',real_hook)
(OUT/'health_tests.json').write_text(json.dumps({'tests':results,'scope':'synthetic detector contracts plus real hook test; no new score policy'},indent=2))
print(json.dumps(results));sys.exit(int(any(r['status']=='FAIL' for r in results)))
