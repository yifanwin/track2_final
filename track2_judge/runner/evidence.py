"""Offline candidate Worker result gate and local callback outbox.

No platform client, queue poller, token, or network call is present. The
production callback's remote idempotency semantics remain unverified.
"""
import hashlib
import json
import math
import sqlite3


def job_key(submit_id, evaluator_sha256, casebank_sha256):
    fields=[submit_id,evaluator_sha256,casebank_sha256]
    if not all(isinstance(v,str) and v for v in fields):raise ValueError('missing job identity')
    return hashlib.sha256(json.dumps(fields,separators=(',',':')).encode()).hexdigest()


def legacy_callback_payload(submit_id,diagnostic,*,verified_output=None):
    if diagnostic.get('official_score') is None:
        raise ValueError('candidate-only score cannot enter platform callback')
    if diagnostic.get('official_approval') is not True:
        raise ValueError('approved scoring contract required')
    if verified_output is None: raise ValueError('ResultGate verified output required')
    from track2_judge.scoring.result_gate import ResultGate
    gate=ResultGate().check(verified_output)
    if gate.get('publishable') is not True: raise ValueError('ResultGate blocked publication')
    if diagnostic.get('candidate_status')=='EXECUTION_FAILURE' or any(e.get('status') not in ('success','failure') for e in diagnostic.get('episodes',[])):
        raise ValueError('execution failure cannot enter callback')
    score=diagnostic['official_score']
    if isinstance(score,bool) or not isinstance(score,(int,float)) or not math.isfinite(score) or not 0<=score<=100:
        raise ValueError('invalid official score')
    if score!=gate['score_100']: raise ValueError('official score differs from verified frozen score')
    return {'submitId':str(submit_id),'executeStatus':1,'score':float(score),'executeMsg':'Track2 trusted evaluation complete'}


class OfflineOutbox:
    def __init__(self,path):
        self.db=sqlite3.connect(path,timeout=5,isolation_level=None)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('CREATE TABLE IF NOT EXISTS callbacks (job_key TEXT PRIMARY KEY, submit_id TEXT NOT NULL, payload TEXT NOT NULL, state TEXT NOT NULL, attempt INTEGER NOT NULL DEFAULT 0, owner TEXT, ack TEXT)')
    def close(self):self.db.close()
    def stage(self,key,payload):
        encoded=json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False)
        self.db.execute('BEGIN IMMEDIATE')
        try:
            row=self.db.execute('SELECT payload FROM callbacks WHERE job_key=?',(key,)).fetchone()
            if row and row[0]!=encoded:raise ValueError('same submission/version key has conflicting score')
            if not row:self.db.execute('INSERT INTO callbacks(job_key,submit_id,payload,state) VALUES(?,?,?,?)',(key,payload['submitId'],encoded,'pending'))
            self.db.execute('COMMIT')
        except Exception:
            self.db.execute('ROLLBACK');raise
    def claim(self,owner):
        if not owner:raise ValueError('worker identity required')
        self.db.execute('BEGIN IMMEDIATE')
        try:
            row=self.db.execute("SELECT job_key,payload,attempt FROM callbacks WHERE state='pending' ORDER BY rowid LIMIT 1").fetchone()
            if row:self.db.execute("UPDATE callbacks SET state='claimed',owner=?,attempt=attempt+1 WHERE job_key=?",(owner,row[0]))
            self.db.execute('COMMIT')
        except Exception:
            self.db.execute('ROLLBACK');raise
        return None if not row else {'job_key':row[0],'payload':json.loads(row[1]),'attempt':row[2]+1}
    def fail(self,key,owner):
        updated=self.db.execute("UPDATE callbacks SET state='pending',owner=NULL WHERE job_key=? AND state='claimed' AND owner=?",(key,owner)).rowcount
        if updated!=1:raise ValueError('callback claim ownership mismatch')
    def acknowledge(self,key,owner,ack):
        if not ack:raise ValueError('platform acknowledgment required')
        updated=self.db.execute("UPDATE callbacks SET state='delivered',ack=? WHERE job_key=? AND state='claimed' AND owner=?",(str(ack),key,owner)).rowcount
        if updated!=1:raise ValueError('callback claim ownership mismatch')
    def reconcile_claim_after_restart(self,key,platform_status,ack=None):
        """Require an external authoritative result before replaying a lost claim."""
        if platform_status not in ('accepted','not_accepted'):
            raise ValueError('authoritative platform status required')
        if platform_status=='accepted' and not ack:
            raise ValueError('accepted callback acknowledgment required')
        if platform_status=='accepted':
            updated=self.db.execute("UPDATE callbacks SET state='delivered',ack=?,owner=NULL WHERE job_key=? AND state='claimed'",(str(ack),key)).rowcount
        else:
            updated=self.db.execute("UPDATE callbacks SET state='pending',owner=NULL WHERE job_key=? AND state='claimed'",(key,)).rowcount
        if updated!=1:raise ValueError('no uncertain callback claim to reconcile')
    def state(self,key):
        row=self.db.execute('SELECT state,attempt,ack FROM callbacks WHERE job_key=?',(key,)).fetchone()
        return None if row is None else {'state':row[0],'attempt':row[1],'ack':row[2]}
