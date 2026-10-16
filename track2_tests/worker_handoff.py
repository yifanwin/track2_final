"""Synthetic offline queue/callback replay; zero network or production access."""
import argparse,json,os
from pathlib import Path
from track2_judge.runner.evidence import OfflineOutbox,job_key,legacy_callback_payload

p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
root=Path(a.output).resolve();root.mkdir(parents=True,mode=0o700,exist_ok=False);os.umask(0o077)
checks={}
try:legacy_callback_payload('synthetic-1',{'candidate_score_100':30,'official_score':None})
except ValueError:checks['candidate_score_callback_blocked']='PASS'
else:raise AssertionError('candidate score entered callback')
from track2_tests.hardening import fixture
verified=fixture(root/'verified_synthetic');payload=legacy_callback_payload('synthetic-1',{'official_score':30.,'official_approval':True},verified_output=verified)
key=job_key('synthetic-1','synthetic-evaluator-hash','synthetic-bank-hash');db=root/'offline_outbox.sqlite'
a1=OfflineOutbox(db);a2=OfflineOutbox(db)
try:
    a1.stage(key,payload);a2.stage(key,payload);checks['duplicate_stage_idempotent']='PASS'
    claim=a1.claim('worker-A');assert claim['job_key']==key and claim['attempt']==1
    assert a2.claim('worker-B') is None;checks['two_workers_one_local_claim']='PASS'
    a1.fail(key,'worker-A');claim=a2.claim('worker-B');assert claim['attempt']==2
    a2.acknowledge(key,'worker-B','SYNTHETIC_ACK');assert a1.claim('worker-A') is None
    checks['failure_retry_and_ack']='PASS'
    conflict={**payload,'score':99.0}
    try:a1.stage(key,conflict)
    except ValueError:checks['conflicting_score_rejected']='PASS'
    else:raise AssertionError('score conflict accepted')
    state=a1.state(key);assert state=={'state':'delivered','attempt':2,'ack':'SYNTHETIC_ACK'}
    second=job_key('synthetic-2','synthetic-evaluator-hash','synthetic-bank-hash')
    a1.stage(second,{**payload,'submitId':'synthetic-2'})
    assert a1.claim('worker-A')['job_key']==second
    a1.close();a1=OfflineOutbox(db)
    assert a2.claim('worker-B') is None
    try:a1.reconcile_claim_after_restart(second,'unknown')
    except ValueError:pass
    else:raise AssertionError('unknown remote callback status auto-replayed')
    a1.reconcile_claim_after_restart(second,'accepted','SYNTHETIC_REMOTE_ACK')
    assert a1.state(second)['state']=='delivered' and a2.claim('worker-B') is None
    checks['crash_requires_authoritative_reconciliation']='PASS'
finally:a1.close();a2.close()
result={'status':'PASS','checks':checks,'network_requests':0,'production_queue_reads':0,'platform_exactly_once':'NOT TESTED: remote callback idempotency/ack contract absent','scope':'synthetic offline Worker handoff; no formal score policy change'}
(root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
