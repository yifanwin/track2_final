"""Explicit approved public input and RGB boundary; no env/object/segmentation escapes."""
import base64, copy, hashlib, json
from io import BytesIO
import numpy as np
from track2_judge.util.common import plain

ALLOWED_FIELDS={'text_instruction','reference_image','predicates','constraints',
               # ★ 裁判侧 Oracle 专用（第 6 步）：GT 规划结果经**唯一入口**下发。
               #   普通选手拿不到 —— `select_case.run()` 只有在显式传
               #   `oracle_plan=` 时才会把它放进公开输入。
               'oracle_plan'}

def rgb_payload(rgb,*,source):
    from PIL import Image
    value=np.asarray(rgb)
    if value.ndim==4 and value.shape[0]==1:value=value[0]
    if value.ndim!=3 or value.shape[2] not in (3,4) or value.shape[0]>512 or value.shape[1]>512 or value.dtype!=np.uint8:raise ValueError('RGB format/budget rejected')
    buffer=BytesIO();Image.fromarray(value[:,:,:3]).save(buffer,format='PNG');blob=buffer.getvalue()
    return {'mime_type':'image/png','width':value.shape[1],'height':value.shape[0],
            'data_base64':base64.b64encode(blob).decode('ascii'),'sha256':hashlib.sha256(blob).hexdigest(),'source':source}

def validate_image(value):
    from PIL import Image
    if type(value) is not dict or set(value)-{'mime_type','width','height','data_base64','sha256','source'}:raise ValueError('image schema rejected')
    if value.get('mime_type')!='image/png' or not isinstance(value.get('data_base64'),str):raise ValueError('image encoding rejected')
    if len(value['data_base64'])>1500000:raise ValueError('image budget exceeded')
    blob=base64.b64decode(value['data_base64'],validate=True)
    if len(blob)>1024**2:raise ValueError('image budget exceeded')
    image=Image.open(BytesIO(blob))
    if image.format!='PNG' or image.mode!='RGB' or image.width>512 or image.height>512:raise ValueError('image size/mode rejected')
    image.load()
    if (value.get('width'),value.get('height'))!=image.size or hashlib.sha256(blob).hexdigest()!=value.get('sha256'):raise ValueError('image integrity failed')
    # Re-encode RGB only: remove PNG text/EXIF/filename metadata from public payload.
    return rgb_payload(np.asarray(image),source='approved_reference')

class ObservationProvider:
    """公开输入的准备 + **相机帧的产出**。

    ## `live`（第 5 步 / 模式 B）

    默认 `False` = 老行为：`capture_initial` 拍一次，之后 `observation()` 一直返回
    那一帧（`deepcopy`）。**21 项回归跑的就是这条**，所以默认值不动。

    `live=True`（runtime 节点用）= 每次 `observation()` **重新渲染**。模式 B 下
    选手每步都要看当前画面，缓存首帧等于给了他一张过期的图。

    ## 相机失败时**不返回陈旧帧**

    `live=True` 且渲染抛异常时：`camera_status='FAIL'` 且 `first_frame=None`。
    **关键**：不能把上一帧留着 —— 那会让选手把陈旧画面当成新的
    （验收 3 的原话：相机失败时不返回陈旧帧）。
    """

    def __init__(self,*,approved_fields=(),camera=None,camera_required=False,
                 synthetic_camera=False,live=False):
        if set(approved_fields)-ALLOWED_FIELDS:raise ValueError('unknown public field')
        self.approved_fields=set(approved_fields);self.camera=camera;self.camera_required=camera_required
        self.synthetic_camera=synthetic_camera;self.first_frame=None;self.camera_status='NOT CONFIGURED'
        # `live` 只影响「要不要每步重拍」；帧的**来源**（camera 回调）不变。
        self.live=bool(live);self._bound_env=None;self.frame_count=0
    def prepare(self,inputs):
        if type(inputs) is not dict or set(inputs)-self.approved_fields:raise ValueError('unapproved public input')
        result=json.loads(json.dumps(plain(inputs),allow_nan=False))
        if 'reference_image' in result:result['reference_image']=validate_image(result['reference_image'])
        # Public predicates/constraints are supplied separately by judge approval;
        # never derive them by copying the hidden suite or complete target_layout.
        for key in ('predicates','constraints'):
            if key in result and not isinstance(result[key],list):raise ValueError('public contract list required')
        allowed_predicate={'type','object','weight','region','target_pose','eps_pos','eps_rot_deg','reference','relation','attribute','value','checkpoints'}
        allowed_constraint={'type','forbidden_zones','delta_max'}
        def primitives(value):
            if isinstance(value,list):return all(primitives(x) for x in value)
            return isinstance(value,(str,int,float,bool)) or value is None
        for name,fields in [('predicates',allowed_predicate),('constraints',allowed_constraint)]:
            for entry in result.get(name,[]):
                if type(entry) is not dict or set(entry)-fields or not all(primitives(v) for v in entry.values()):raise ValueError('unapproved nested public field')
        if len(json.dumps(result))>1500000:raise ValueError('public input budget')
        return result
    @property
    def _source(self):
        return 'SYNTHETIC' if self.synthetic_camera else 'trusted_camera_rgb'

    def capture_initial(self,env):
        """绑定 env。（`live=True` 时不在这里拍首帧，见 `observation`。）"""
        self._bound_env=env
        self.first_frame=None
        if self.camera is None:
            self.camera_status='NOT CONFIGURED'
            if self.camera_required:raise RuntimeError('required first-frame camera absent')
            return
        if self.live:
            # 每步都拍，所以这里只把状态标成 PASS —— 真拍不拍得出来由
            # 第一次 `observation()` 决定（那时 env 已经 reset 完了）。
            self.camera_status='SYNTHETIC PASS' if self.synthetic_camera else 'PASS'
            return
        try:
            self.first_frame=rgb_payload(self.camera(env),source=self._source)
            self.camera_status='SYNTHETIC PASS' if self.synthetic_camera else 'PASS'
        except Exception:
            self.camera_status='FAIL'
            if self.camera_required:raise RuntimeError('first-frame capture failed') from None

    def observation(self):
        """返回 `{'first_frame', 'camera_status'}`。

        `live=True` 时**每次调用都重拍**；拍失败 → `camera_status='FAIL'` +
        `first_frame=None`（**不留上一帧**）。
        """
        if self.live and self.camera is not None and self._bound_env is not None:
            try:
                self.first_frame=rgb_payload(self.camera(self._bound_env),source=self._source)
                self.camera_status='SYNTHETIC PASS' if self.synthetic_camera else 'PASS'
                self.frame_count+=1
            except Exception:
                self.camera_status='FAIL'
                # ★ 不留陈旧帧 —— 否则选手会把过期画面当成新的。
                self.first_frame=None
                if self.camera_required:raise RuntimeError('camera capture failed') from None
        return {'first_frame':copy.deepcopy(self.first_frame),'camera_status':self.camera_status}
