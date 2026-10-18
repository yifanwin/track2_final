"""Reproducible meaningful synthetic contract, RPC, scheduling and scoring checks."""
import argparse, copy, hashlib, json, sys, tempfile, traceback
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from track2_judge.toolapi.dispatch import ManiSkillSimulatorAdapter
from track2_judge.diagnostics.state_export import canonical_root, actor_from_root, export_layout
from track2_judge.scoring.scorer_bridge import v3, FrozenVerifier
from track2_tests.fixtures.synthetic import suite, factory, SETTINGS
from track2_judge.runner.candidate_runner import CandidateRunner
from track2_judge.sandbox.rpc import encode, decode, SkillProcess, SecurityNotReady

def main(output,isolated=False):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True); results=[]
    def test(name,fn):
        try: evidence=fn(); results.append({'name':name,'status':'PASS','evidence':evidence})
        except Exception as exc: results.append({'name':name,'status':'FAIL','error':str(exc),'trace':traceback.format_exc()})
    def reject(fn):
        try:fn()
        except (ValueError,RuntimeError):return
        raise AssertionError('rejection missing')
    def coordinates():
        rng=np.random.default_rng(20261013); errors=[]
        for _ in range(150):
            r=Rotation.random(random_state=rng);q=r.as_quat()[[3,0,1,2]];root=rng.normal(size=3);pivot=rng.normal(size=3);scale=rng.uniform(.1,3,3)
            actor=root+r.apply(pivot*scale); recovered=canonical_root(actor,q,pivot,scale)
            errors.append(float(np.max(abs(recovered-root)))); assert errors[-1]<1e-12
        data=encode({'x':np.float32(.2),'q':np.array([1,0,0,0],dtype=np.float32)});decode(data)
        reject(lambda:encode({'x':np.float32('nan')}));reject(lambda:decode('{"x":NaN}'))
        reject(lambda:canonical_root([0,0,0],[0,0,0,0],[0,0,0]))
        return {'roundtrips':150,'max_error_m':max(errors),'float32':'accepted','NaN':'rejected'}
    test('coordinate_serialization',coordinates)
    def actions():
        task=suite()['tasks'][0];a=factory()(task,0);a.reset(public_task_inputs={'text_instruction':'move'})
        valid={'joint_position':[0,1.2,-1.2,0,.5,0],'gripper_opening_m':.08}
        bad=[{'set_pose':[0,0,0]},dict(valid,translate=[0,0,0]),dict(valid,gripper_opening_m=True),dict(valid,gripper_opening_m=float('nan')),dict(valid,joint_position=[1]*6),dict(valid,joint_position=[0]*5),dict(valid,joint_position=['0',1.2,-1.2,0,.5,0])]
        for x in bad:reject(lambda:a.step(x))
        for method in ('reset','get_final_state','close','env','set_pose'):reject(lambda:a.dispatch_public(method))
        obs=a.observe();obs['task_inputs']['text_instruction']='mutation';assert a.observe()['task_inputs']['text_instruction']=='move'
        reject(lambda:a.reset(public_task_inputs={'initial_scene':{'private':1}}));a.close()
        return {'invalid_actions':len(bad),'judge_methods_blocked':5,'public_fields':'explicit whitelist'}
    test('action_and_observation_boundary',actions)
    def score_match():
        s=suite();task=s['tasks'][0];checked=[]
        for count in range(4):
            final=copy.deepcopy(task['initial_layout'])
            for j in range(count):final['objects'][j]=copy.deepcopy(task['target_layout']['objects'][j])
            direct=v3.score_state(task,final,task['initial_layout'],s['config'],{})
            adapted=FrozenVerifier().verify(task,task['initial_layout'],final,s['config'])
            for key in direct:assert direct[key]==adapted[key]
            assert direct['s']==([0,0,2/3,1][count]);checked.append({'count':count,'completion':direct['completion'],'s':direct['s']})
        # Exactly .6 with unequal weights must survive, old Verifier did not implement weighting.
        t=copy.deepcopy(task); t['predicates'][0]['weight']=3
        final=copy.deepcopy(task['initial_layout']);final['objects'][0]=copy.deepcopy(task['target_layout']['objects'][0])
        assert FrozenVerifier().verify(t,t['initial_layout'],final,s['config'])['s']==.6
        missing=copy.deepcopy(final); missing['objects']=missing['objects'][1:]
        assert v3.score_state(task,missing,task['initial_layout'],s['config'],{})['strict_success']==False
        config=copy.deepcopy(s['config']);config['event_constraints']=['collision']
        reject(lambda:FrozenVerifier().verify(task,final,final,config))
        return {'states':checked,'weighted_threshold':.6,'unsupported_events':'scoring refused'}
    test('frozen_scoring_equivalence_threshold',score_match)
    def chain(kind,count=0,budget=2400):
        result=CandidateRunner(factory(count,budget),isolated=isolated,trusted=True,timeout_s=.35 if kind=='timeout' else 15,max_steps=budget).run(suite(),output/kind,{'text_instruction':kind})
        runs=json.loads((output/kind/'runs.json').read_text());assert len(runs['runs'])==10
        assert [r['seed'] for r in runs['runs'][::2]]==list(range(5))
        initial_hashes=[]
        for i in range(0,10,2):
            a=(output/kind/runs['runs'][i]['initial_layout']).read_bytes()
            b=(output/kind/runs['runs'][i+1]['initial_layout']).read_bytes()
            assert a==b
            initial_hashes.append(hashlib.sha256(a).hexdigest())
        assert len(set(initial_hashes))==5
        assert all(r['attempts']<= (1 if r['mode']=='first' else 3) for r in result['episodes'])
        if kind in ('illegal','private','forge','oversized'):assert all(e['status']=='error' for e in result['episodes'])
        # 2026-10-18：'timeout' 与 'budget' 两条都源自**比赛预算**
        #   （episode 墙钟 / 控制步预算）→ 可评分终态 `budget_timeout`。
        #   'infrastructure_timeout'（链路故障）由 functional.py 的
        #   action 超时用例覆盖。
        elif kind=='timeout':assert all(e['status']=='budget_timeout' for e in result['episodes'])
        elif kind=='budget':assert all(e['status']=='budget_timeout' for e in result['episodes'])
        elif count==3:assert all(e['status']=='success' for e in result['episodes'])
        else:assert all(e['status']=='failure' for e in result['episodes'])
        raw=v3.evaluate(suite(),{'schema_version':1,'runs':runs['runs']},output/kind)
        assert raw['score_100']==result['raw_frozen_score_100']
        return result
    for kind,count in [('move',3),('partial',2),('below',1),('noop',0),('illegal',0),('private',0),('forge',0),('oversized',0),('timeout',0)]:
        test('chain_'+kind,lambda k=kind,c=count:chain(k,c))
    test('chain_action_budget',lambda:chain('budget',0,2))
    def initialization():
        def broken(task,seed): raise RuntimeError('init failed')
        # Factory failures must never produce fabricated states or scores.
        result=CandidateRunner(broken,isolated=False,trusted=True).run(suite(),output/'initfail',{})
        assert result['scoring_status']=='REFUSED: missing trustworthy state'
        assert 'raw_frozen_score_100' not in result
        return 'initialization refused; no scoring; 10 failures recorded'
    test('initialization_failure',initialization)
    def failclosed():
        reject(lambda:SkillProcess(Path(__file__).resolve().parents[1] / 'track2_judge' / 'skills' / 'trusted.py',output/'untrusted',isolated=False,trusted=False))
        return 'untrusted execution refused without namespace isolation'
    test('untrusted_fail_closed',failclosed)
    (output/'test_results.json').write_text(encode({'source':'LOCAL SYNTHETIC, not physical score','results':results}),encoding='utf8')
    for r in results:print(r['status'],r['name'])
    return 1 if any(r['status']=='FAIL' for r in results) else 0

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--isolated',action='store_true');args=p.parse_args()
    sys.exit(main(args.output,args.isolated))
