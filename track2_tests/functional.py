"""Package lifecycle, supplemental predicates, event coverage and public image boundary."""
import argparse,copy,hashlib,json,sys,time,zipfile
from pathlib import Path
import numpy as np
from track2_judge.sandbox.rpc import encode
from track2_tests.fixtures.synthetic import suite,factory
from track2_judge.runner.candidate_runner import CandidateRunner
from track2_tests.fixtures.make_internal_skills import create
from track2_judge.sandbox.package import unpack_skill
from track2_judge.observation.provider import ObservationProvider,rgb_payload,validate_image
from track2_judge.contract.predicates import evaluate_predicate,OrderedCheckpoints
from track2_judge.contract.events import TrustedEventRecorder

def main(output,isolated):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True);results=[]
    def test(name,fn):
        try:r=fn();results.append({'name':name,'status':'PASS','evidence':r})
        except Exception as exc:
            import traceback
            results.append({'name':name,'status':'FAIL','error':type(exc).__name__,'details':str(exc),'trace':traceback.format_exc()})
    def rejected(fn):
        try:fn()
        except (ValueError,RuntimeError):return
        raise AssertionError('missing rejection')
    package=create(output/'internal_skill')
    def lifecycle(kind,count):
        result=CandidateRunner(factory(count),isolated=isolated,trusted=True,package_zip=package,timeout_s=2 if kind=='timeout' else 15).run(suite(),output/kind,{'text_instruction':kind})
        assert len(result['episodes'])==10
        expected='success' if count==3 and kind=='move' else 'error' if kind in ('illegal','private','error') else 'timeout' if kind=='timeout' else 'failure'
        assert all(e['status']==expected for e in result['episodes']),result
        events=json.loads((output/kind/'skill_lifecycle.json').read_text())
        if kind in ('move','noop','forge'):
            assert all(e['events'][0:2]==['ready','initialize'] and e['events'][-1]=='shutdown' for e in events)
        assert result['raw_frozen_score_100']==(100. if expected=='success' else 30.)
        return {'episodes':10,'status':expected,'package_loaded':True,'lifecycle':events[0]['events'],'score_is_judge_only':True}
    for kind,count in [('move',3),('noop',0),('illegal',0),('private',0),('forge',0),('error',0),('timeout',0)]:test('package_'+kind,lambda k=kind,c=count:lifecycle(k,c))
    def zip_rejection():
        for i,(name,symlink) in enumerate([('../escape.py',False),('/absolute.py',False),('C:secret',False),('alias',True)]):
            path=output/f'bad{i}.zip'
            with zipfile.ZipFile(path,'w') as z:
                info=zipfile.ZipInfo(name)
                if symlink:info.external_attr=(0o120777<<16)
                z.writestr(info,'synthetic')
            rejected(lambda:unpack_skill(path,output/f'unpack{i}'))
        path=output/'bad_manifest.zip'
        with zipfile.ZipFile(path,'w') as z:z.writestr('manifest.json',json.dumps({'entrypoint':'/synthetic_judge_private/canary.py','skill_name':'bad','version':'1','api_version':'ctrc-skill-v1'}))
        rejected(lambda:unpack_skill(path,output/'bad_manifest_staging'))
        return {'traversal_absolute_drive_symlink':'rejected'}
    test('safe_package_unpack',zip_rejection)
    def images():
        rgb=np.zeros((24,32,3),dtype=np.uint8);rgb[:,:,0]=120
        image=rgb_payload(rgb,source='SYNTHETIC');validate_image(image)
        p=ObservationProvider(approved_fields=('text_instruction','reference_image','predicates'),camera=lambda _:rgb,synthetic_camera=True)
        data=p.prepare({'reference_image':image,'text_instruction':'synthetic','predicates':[{'type':'P2','object':'a','target_pose':[0,0,0]}]});p.capture_initial(None)
        assert p.observation()['camera_status']=='SYNTHETIC PASS'
        rejected(lambda:p.prepare({'seed':4}))
        rejected(lambda:p.prepare({'predicates':[{'type':'P2','hidden_GT':[1,2,3]}]}))
        rejected(lambda:p.prepare({'predicates':[{'type':'P2','target_pose':{'private_path':'canary'}}]}))
        rejected(lambda:p.prepare({'reference_image':'/private/image.png'}))
        bad=copy.deepcopy(image);bad['sha256']='0'*64;rejected(lambda:validate_image(bad))
        missing=ObservationProvider(camera_required=True);rejected(lambda:missing.capture_initial(None))
        return {'RGB_inline':'PASS','required_camera_missing':'rejected','nested_private_fields':'rejected','real_render':'tested separately'}
    test('approved_observation_and_images',images)
    def blocked_pipe():
        from track2_judge.sandbox.package import PackageExecutor
        packed=unpack_skill(package,output/'blocked_staging');process=PackageExecutor(packed,output/'blocked_work',isolated=isolated,trusted=True,timeout_s=10)
        try:
            process.initialize({'skill_api_version':'ctrc-skill-v1'})
            observation={'task_inputs':{'text_instruction':'blocked_reader'}}
            start=time.monotonic()
            try:process.run(observation,lambda name,payload:{'approved_image':'synthetic'*30000},timeout_s=5)
            except TimeoutError:pass
            else:raise AssertionError('blocked write not rejected')
            elapsed=time.monotonic()-start;assert elapsed<4
            return {'bounded_pipe_seconds':elapsed,'host_not_reading':'killed; no judge hang'}
        finally:process.abort()
    test('bounded_image_response_pipe',blocked_pipe)
    def state():
        objs={n:{'canonical_root_m':p,'aabb_world_m':[np.array(p)-.1,np.array(p)+.1]} for n,p in [('a',[0,0,.2]),('b',[1,0,0]),('c',[-1,0,0]),('table',[0,0,0])]}
        objs['table']['aabb_world_m']=[[-2,-2,-.1],[2,2,.1]]
        objs['b']['container_aabb_world_m']=[[-2,-2,-2],[2,2,2]]
        objs['b']['container_geometry_type']='approved_axis_aligned_container_region'
        records={'a':{'color':'red','class':'block'}};digest=hashlib.sha256(json.dumps(records,sort_keys=True,separators=(',',':')).encode()).hexdigest();objs['a']['asset_identity_verified']=True
        return {'spatial_objects':objs,'trusted_asset_registry':{'objects':records,'sha256':digest},'contacts':[{'bodies':['a','table'],'points':[{'position_m':[0,0,.1],'separation_m':0,'impulse_Ns':[0,0,.01]}]}]}
    def predicates():
        s=state();cases=[{'type':'P1','object':'a','region_aabb_m':[[-.5,-.5,-.5],[.5,.5,.5]],'membership':'point'},
            {'type':'P1','object':'a','region_aabb_m':[[-.5,-.5,-.5],[.5,.5,.5]],'membership':'whole_aabb'},
            {'type':'P3','object':'a','reference':'b','relation':'left-of','axis':'X'},
            {'type':'P3','object':'a','reference':'c','relation':'right-of','axis':'X'},
            {'type':'P3','object':'a','reference':'b','relation':'behind','axis':'X'},
            {'type':'P3','object':'a','reference':'c','reference2':'b','relation':'between','tolerance_m':.3},
            {'type':'P3','object':'a','reference':'b','relation':'in','tolerance_m':0},
            {'type':'P3','object':'a','reference':'table','relation':'on','tolerance_m':.001},
            {'type':'P4','object':'a','attribute':'color','value':'red'}]
        assert all(evaluate_predicate(p,s)['passed'] for p in cases)
        bad=copy.deepcopy(cases[0]);bad['region_aabb_m']=[[2,2,2],[3,3,3]];assert not evaluate_predicate(bad,s)['passed']
        for p in cases:assert evaluate_predicate(p,{'spatial_objects':{}})['status']=='inconclusive'
        corrupt=copy.deepcopy(s);corrupt['trusted_asset_registry']['sha256']='0'*64;assert evaluate_predicate(cases[-1],corrupt)['status']=='inconclusive'
        monitor=OrderedCheckpoints([{'name':'first','predicates':[cases[0]]},{'name':'second','predicates':[bad]}]);monitor.sample(s,1);assert not monitor.result()['passed']
        s['spatial_objects']['a']['canonical_root_m']=[2.5,2.5,2.5];monitor.sample(s,2);assert monitor.result()['passed']
        assert [r['physics_frame'] for r in monitor.result()['reached']]==[1,2]
        # Unsupported predicates are never inserted in frozen tasks/scores.
        from track2_judge.scoring.scorer_bridge import v3
        task=copy.deepcopy(suite()['tasks'][0]);task['predicates'][0]['type']='P1';rejected(lambda:v3.validate_task(task))
        return {'P1_P3_P4_P5':'independently verified diagnostics','frozen_compatibility':'P2 only; unsupported rejected'}
    test('supplemental_predicates',predicates)
    def event_stream():
        policy={'schema':'candidate-event-policy-v2','collision':{'count':'contact_onset','forbidden_pairs':[['a','b']]},'forbidden_zone':{'count':'entry','zones':[{'id':'z','subject':'a','membership':'point','aabb_m':[[0,0,0],[1,1,1]]}]}}
        r=TrustedEventRecorder(policy);s={'trusted_physics_substep':True,'known_contact_bodies':['a','b'],'contacts':[{'bodies':['a','b'],'points':[{'separation_m':-.001,'impulse_Ns':[.01,0,0]}]}],'judge_points_m':{'a':[.5,.5,.5]}}
        r.sample(s);r.sample(s);off=copy.deepcopy(s);off['contacts']=[];off['judge_points_m']['a']=[2,2,2];r.sample(off);r.sample(s)
        result=r.report(['collision','forbidden_zone'],4);assert result['event_counts']=={'collision':2,'forbidden_zone':2}
        overlap=copy.deepcopy(policy);overlap.pop('collision');overlap['forbidden_zone']['zones'][0]['membership']='aabb_overlap';a=TrustedEventRecorder(overlap)
        sample=copy.deepcopy(s);sample['spatial_objects']={'a':{'aabb_world_m':[[.9,.9,.9],[1.1,1.1,1.1]]}};a.sample(sample);assert a.report(['forbidden_zone'],1)['event_counts']['forbidden_zone']==1
        unknown=copy.deepcopy(s);unknown['known_contact_bodies']=[];rejected(lambda:TrustedEventRecorder(policy).sample(unknown))
        rejected(lambda:r.report(['collision'],5));missing=TrustedEventRecorder(policy);rejected(lambda:missing.report(['collision'],0))
        rejected(lambda:missing.sample({'contacts':[]}));return {'SYNTHETIC_ONLY':True,'onset_not_frames':result['event_counts'],'missing_coverage':'refused'}
    test('trusted_event_counting_contract',event_stream)
    (output/'v2_test_results.json').write_text(encode({'results':results,'scope':'synthetic functional tests; actual PhysX and namespace security tested separately'}),encoding='utf8')
    for r in results:print(r['status'],r['name'])
    return int(any(r['status']=='FAIL' for r in results))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--isolated',action='store_true');a=p.parse_args();sys.exit(main(a.output,a.isolated))
