"""New contract/formula tests, not replaying old V2 to claim new coverage."""
import argparse,copy,json,math,random
from pathlib import Path
from track2_judge.scoring.scorer_bridge import v3
from track2_judge.contract.extension import TrustedContractExtension,SCHEMA,ApprovalRequired,digest
from track2_judge.contract.predicates import OrderedCheckpoints,evaluate_predicate
from track2_tests.fixtures.synthetic import suite

def main(output):
    root=Path(output);root.mkdir(parents=True,exist_ok=False);tests=[]
    def test(name,fn):
        try:detail=fn();tests.append({'test':name,'status':'PASS','detail':detail})
        except Exception as e:tests.append({'test':name,'status':'FAIL','error':type(e).__name__,'detail':str(e)[:180]})
    old=suite()['tasks'][0];name=old['predicates'][0]['object'];cfg=v3.default_config();engine=TrustedContractExtension()
    state={'spatial_objects':{name:{'canonical_root_m':[0,0,0],'aabb_world_m':[[-.01]*3,[.01]*3]}}}
    region={'type':'P1','object':name,'membership':'whole_aabb','region_aabb_m':[[-1]*3,[1]*3],'weight':1}
    task={**copy.deepcopy(old),'contract_schema':SCHEMA,'task_type':'pick_place','tier':'standard','predicates':[region,{'type':'P2','object':name,'weight':1}],'constraints':[]}
    def mandatory():
        for typ,required in [('pick_place',['P1','P2']),('attribute_sort',['P1','P4']),('spatial_relation',['P1','P3']),('multi_object_organize',['P1','P2','P3','P5','C1'])]:
            t={**task,'task_type':typ,'predicates':[region]}
            try:engine.validate_task(t)
            except ValueError:pass
            else:raise AssertionError('required predicate omitted')
        t={**task,'task_type':'dual_arm_handoff'}
        try:engine.validate_task(t)
        except ValueError:pass
        else:raise AssertionError('dual arm accepted')
        engine.validate_task(task);return 'PDF mapping enforced; optional C1 not forced; dual arm rejected'
    test('required_predicates_single_arm',mandatory)
    def threshold():
        t=copy.deepcopy(task);t['predicates'][0]['weight']=2;t['predicates'][1]['weight']=1
        r=engine.verify(t,old['initial_layout'],old['initial_layout'],cfg,initial_state=state,final_state=state)
        assert r['completion']==2/3 and r['s']==2/3
        t['predicates'][0]['weight']=1;r=engine.verify(t,old['initial_layout'],old['initial_layout'],cfg,initial_state=state,final_state=state)
        assert r['completion']==.5 and r['s']==0;return 'exact Fraction completion, threshold .6; no new weights in total formula'
    test('partial_threshold_exact',threshold)
    def pose_equivalence():
        rng=random.Random(70);residual=0
        for _ in range(150):
            final=copy.deepcopy(old['target_layout']);obj=next(x for x in final['objects'] if x['instance']==name);obj['position'][0]+=rng.uniform(-8,8)
            base=copy.deepcopy(old);base['predicates']=[{'type':'P2','object':name,'weight':1}]
            a=v3.score_state(base,final,old['initial_layout'],{**cfg,'non_target_displacement':False},{})
            b=engine.verify(task,old['initial_layout'],final,cfg,initial_state=state,final_state=state)
            p=next(p for p in b['predicates'] if p['type']=='P2');assert p['passed']==a['predicates'][0]['passed']
            residual=max(residual,abs(p['position_error_cm']-a['predicates'][0]['position_error_cm']))
        return {'comparisons':150,'max_pose_error_difference':residual}
    test('actual_frozen_P2_equivalence_150',pose_equivalence)
    def formula():
        rng=random.Random(700);largest=0
        for tier in ('loose','standard','strict'):
            config=v3.default_config('sim',tier)
            for _ in range(100):
                def runs(mode):return [{'s':rng.choice([0,2/3,1]),'strict_success':False,'violation_count':rng.randrange(3),'elapsed_s':rng.uniform(1,200),'plan_revisions':0 if mode=='first' else rng.randrange(3)} for i in range(5)]
                a,b=runs('first'),runs('final');actual=v3.summarize(a,b,config)
                first=sum(x['s'] for x in a)/5;final=sum(x['s'] for x in b)/5;V=min(1,sum(x['violation_count'] for x in b)/5/config['violation_cap']);E=sum(min(1,config['time_ref_s']/x['elapsed_s']) for x in b)/5;sd=math.sqrt(sum((x['s']-final)**2 for x in b)/5)
                pdf=.5*final+.2*first+.15*(1-V)+.1*E+.05*(1-sd);largest=max(largest,abs(pdf-actual['S']))
        assert largest<1e-14;return {'random_paired_summaries':300,'max_PDF_formula_difference':largest,'aggregator':'unmodified V3.summarize actually called'}
    test('PDF_formula_frozen_summary_300',formula)
    def p3():
        s={'spatial_objects':{'a':{'canonical_root_m':[0,0,0],'aabb_world_m':[[0,0,0],[.1,.1,.1]]},'b':{'canonical_root_m':[1,1,0],'aabb_world_m':[[0,0,-.1],[1,1,0]],'container_geometry_type':'approved_axis_aligned_container_region','container_aabb_world_m':[[-1,-1,-1],[2,2,2]]},'c':{'canonical_root_m':[-1,-1,0]}},'contacts':[{'bodies':['a','b'],'points':[{'separation_m':0,'impulse_Ns':[0,0,.1]}]}]}
        for relation,fields in [('left-of',{'axis':'X'}),('right-of',{'axis':'X','object':'b','reference':'a'}),('behind',{'axis':'Y'}),('between',{'reference2':'c','tolerance_m':.01}),('on',{'tolerance_m':.001}),('in',{'tolerance_m':0})]:
            r=evaluate_predicate({'type':'P3','object':'a','reference':'b','relation':relation,**fields},s);assert r['passed'],(relation,r)
        return 'all six explicitly defined relations; concave containers not silently accepted'
    test('P3_six_candidate_relations',p3)
    def attributes():
        labels={name:{'class':'synthetic_block'}};s=copy.deepcopy(state);s['spatial_objects'][name]['asset_identity_verified']=True;s['trusted_asset_registry']={'objects':labels,'sha256':digest(labels)}
        t={**task,'task_type':'attribute_sort','asset_registry_sha256':digest(labels),'predicates':[region,{'type':'P4','object':name,'attribute':'class','value':'synthetic_block'}]}
        r=engine.verify(t,old['initial_layout'],old['target_layout'],cfg,initial_state=s,final_state=s);assert r['s']==1
        s['trusted_asset_registry']['sha256']='0'*64;r=engine.verify(t,old['initial_layout'],old['target_layout'],cfg,initial_state=s,final_state=s);assert r['predicates'][-1]['status']=='inconclusive'
        return 'pinned judge registry required; unknown/forged labels not PASS'
    test('P4_approved_registry',attributes)
    def missing_events():
        t={**task,'constraints':[{'type':'C1','kind':'collision'}]}
        try:engine.verify(t,old['initial_layout'],old['target_layout'],cfg,initial_state=state,final_state=state)
        except ValueError:return 'missing trusted collision evidence refuses scoring'
        raise AssertionError('missing constraints defaulted to zero')
    test('C1_missing_evidence_refusal',missing_events)
    def checkpoints():
        m=OrderedCheckpoints([{'name':'one','predicates':[region]},{'name':'two','predicates':[region]}]);m.sample(state,1);assert not m.result()['passed'];m.sample(state,2);assert m.result()['passed'] and [x['physics_frame'] for x in m.result()['reached']]==[1,2]
        t=copy.deepcopy(task);t['predicates'].append({'type':'P5','checkpoints':[{'name':'one','predicates':[region]}]})
        try:engine.verify(t,old['initial_layout'],old['target_layout'],cfg,initial_state=state,final_state=state,event_evidence={'P5':{'passed':False}},mode='final')
        except ApprovalRequired:pass
        else:raise AssertionError('unapproved P5 scoring fabricated')
        return 'P5 no standalone weight; ordered trusted samples; invalid recovery scoring edge requires approval'
    test('P5_sequence_no_fabricated_score',checkpoints)
    def task_config():
        c={**cfg,'time_ref_s':45,'completion_threshold':.7};t={**task,'tier':'strict'}
        actual=engine.config_for_task(t,c)
        assert actual['time_ref_s']==45 and actual['completion_threshold']==.7 and actual['eps_pos_cm']==2 and actual['weights']==v3.WEIGHTS
        return 'caller reference time/threshold preserved; tier values globally fixed'
    test('configuration_preservation',task_config)
    def pose_checkpoint():
        s=copy.deepcopy(state);s['objects']={name:{'q_wxyz':[1,0,0,0]}}
        p={'type':'P2','object':name,'target_pose_m':{'position_m':[0,0,0],'q_wxyz':[1,0,0,0]},'eps_pos_m':.03,'eps_rot_deg':10}
        m=OrderedCheckpoints([{'name':'pose','predicates':[p]}]);m.sample(s,1);assert m.result()['passed']
        p['target_pose_m']['position_m']=[1,0,0];assert not evaluate_predicate(p,s)['passed']
        return 'P2 checkpoints use actual canonical position/quaternion, not Skill claims'
    test('P5_pose_checkpoint',pose_checkpoint)
    (root/'extension_tests.json').write_text(json.dumps({'tests':tests,'formal_status':'approval required for extension semantics','frozen_source_changed':False},indent=2));print(json.dumps(tests));return int(any(r['status']=='FAIL' for r in tests))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);raise SystemExit(main(p.parse_args().output))
