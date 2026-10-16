"""Fail-closed judge-side publication gate. Never publish original scene assets.

Metadata absence cannot prove geometry is free of baked goal poses. An asset
requires an independent approval record, a pinned hash and local-geometry
provenance. This gate cannot certify texture pixels/OCR or reconstruct meshes.
"""
import hashlib,json,struct
from pathlib import Path

def check_glb(path,approval):
    p=Path(path);raw=p.read_bytes()
    if len(raw)>64*1024**2 or len(raw)<20:raise ValueError('asset format/budget rejected')
    magic,version,length=struct.unpack_from('<4sII',raw)
    if (magic,version,length)!=(b'glTF',2,len(raw)):raise ValueError('asset GLB rejected')
    size,kind=struct.unpack_from('<I4s',raw,12)
    if kind!=b'JSON' or size>len(raw)-20:raise ValueError('asset metadata rejected')
    meta=json.loads(raw[20:20+size].rstrip(b' \x00'))
    digest=hashlib.sha256(raw).hexdigest()
    needed={'sha256','local_geometry_reviewed','texture_reviewed','approved_public_name','geometry_provenance'}
    if type(approval) is not dict or set(approval)!=needed or approval['sha256']!=digest or approval['local_geometry_reviewed'] is not True or approval['texture_reviewed'] is not True or approval['geometry_provenance']!='unbaked_actor_local':raise ValueError('public asset approval required')
    def inspect(value):
        if isinstance(value,dict):
            if 'extras' in value or 'extensions' in value:raise ValueError('unreviewed asset metadata rejected')
            for k,v in value.items():
                if k=='name' and v!=approval['approved_public_name']:raise ValueError('unapproved asset label')
                inspect(v)
        elif isinstance(value,list):
            for v in value:inspect(v)
        elif isinstance(value,str):
            if any(s in value.lower() for s in ('judge_only','target_pose','final-poses','initial_layout','/mnt/','d:\\','file:','../')):raise ValueError('private asset label/reference rejected')
    inspect(meta)
    for node in meta.get('nodes',[]):
        if any(k in node for k in ('translation','rotation','matrix','scale')):raise ValueError('scene transform requires independent local conversion')
    for image in meta.get('images',[]):
        if 'uri' in image:raise ValueError('external image paths rejected')
    for buffer in meta.get('buffers',[]):
        if 'uri' in buffer:raise ValueError('external geometry paths rejected')
    return {'status':'PASS: APPROVED COPY ONLY','sha256':digest,'metadata_clean':True,'geometry_provenance':'unbaked_actor_local','texture_review':'independent approval required; not automatically certified'}

def publish_copy(path,approval,destination):
    evidence=check_glb(path,approval);dest=Path(destination)
    payload=Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest()!=evidence['sha256']:raise ValueError('asset changed after review')
    with dest.open('xb') as out:out.write(payload)
    return evidence
