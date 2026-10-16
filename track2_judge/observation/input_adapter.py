"""Versioned public input boundary; neither interpretation mode is selected here."""
import copy
import json
from track2_judge.observation.provider import ObservationProvider, validate_image

CONTRACT_STATUS = 'WAIT_CONFIRM_INPUT_CONTRACT'
SCHEMA = 'track2-input-adapter-v1'
PRIVATE_KEYS = {'gt', 'hidden_gt', 'initial_pose', 'initial_layout', 'final_pose',
                'final_layout', 'target_layout', 'casebank', 'manifest', 'seed',
                'scorer_state', 'judge_state', 'score', 'score_file', 'host_path'}

def validate_envelope(envelope):
    if type(envelope) is not dict or set(envelope) != {'schema', 'mode', 'payload'}:
        raise ValueError('input envelope schema rejected')
    if envelope['schema'] != SCHEMA or envelope['mode'] not in ('A', 'B'):
        raise ValueError('input mode/schema rejected')
    value = copy.deepcopy(envelope)
    def scan(obj, depth=0):
        if depth > 20: raise ValueError('input nesting budget')
        if isinstance(obj, dict):
            if any(not isinstance(k, str) or k.lower() in PRIVATE_KEYS for k in obj):
                raise ValueError('private input field rejected')
            for child in obj.values(): scan(child, depth+1)
        elif isinstance(obj, list):
            for child in obj: scan(child, depth+1)
    scan(value)
    if len(json.dumps(value, allow_nan=False)) > 2000000: raise ValueError('input budget')
    payload = value['payload']
    allowed = {'text_instruction', 'reference_image', 'first_frame', 'predicates', 'constraints'}
    if value['mode'] == 'A': allowed |= {'interpreted_text', 'detections'}
    if type(payload) is not dict or set(payload)-allowed: raise ValueError('input payload rejected')
    for key in ('text_instruction', 'interpreted_text'):
        if key in payload and (not isinstance(payload[key], str) or len(payload[key]) > 16384):
            raise ValueError('text field rejected')
    for key in ('reference_image', 'first_frame'):
        if key in payload and payload[key] is not None: payload[key] = validate_image(payload[key])
    if 'detections' in payload:
        if not isinstance(payload['detections'], list) or len(payload['detections']) > 128:
            raise ValueError('detections budget')
        for item in payload['detections']:
            if type(item) is not dict or set(item)-{'label', 'bbox_xyxy', 'confidence'}:
                raise ValueError('detection fields rejected')
            if not isinstance(item.get('label'), str): raise ValueError('detection label rejected')
            bbox=item.get('bbox_xyxy'); confidence=item.get('confidence')
            if not isinstance(bbox,list) or len(bbox)!=4 or any(type(v) not in (int,float) or not 0<=v<=1 for v in bbox):
                raise ValueError('normalized image bbox rejected')
            if bbox[0]>bbox[2] or bbox[1]>bbox[3] or type(confidence) not in (int,float) or not 0<=confidence<=1:
                raise ValueError('detection value rejected')
    public={k:v for k,v in payload.items() if k in ('text_instruction','reference_image','predicates','constraints')}
    ObservationProvider(approved_fields=tuple(public)).prepare(public)
    return value

class InputAdapter:
    """A takes OS-derived public perception; B takes raw public image/text.

    No model invocation, object pose getters, filesystem paths or new tools.
    An official input producer must be explicitly registered after confirmation.
    """
    contract_status = CONTRACT_STATUS
    def __init__(self, *, mode, producer):
        if mode not in ('A','B') or not callable(producer): raise ValueError('explicit producer/mode required')
        self.mode=mode; self.producer=producer
    def observation(self, public_observation):
        return validate_envelope({'schema':SCHEMA,'mode':self.mode,'payload':self.producer(copy.deepcopy(public_observation))})
