"""Judge-only selected-case backend. No casebank/actor enters Skill RPC.
Source physical properties are unchanged.
Confirmed scoring scope is distinct from production-switch authorization.

D2：不再逐 hull 校验 sha256（发布期哈希门禁已删）。
保留的路径包含 + 符号链接检查仍防路径穿越。
"""
import os,json
from pathlib import Path
import numpy as np,sapien,torch
from mani_skill.envs.sapien_env import BaseEnv
from track2_judge.scenario.piper_robot import PiperPrototype,REST
from track2_judge.scenario.prototype_env import prepare_model
PiperPrototype.urdf_path=str(prepare_model())

# ─────────────────────────────────────── 固定机位（第 5 步 / 模式 B）
#
# ⚠️ **机位不能依赖答案**。原来的 `diagnostic_rgb` 分支从公开输入的 P2 目标位姿
# 取机位中心（`cli/select_case.py`），那是**模式 A** 的做法；模式 B 下公开输入
# 里根本没有 `target_pose`，而「把相机摆到目标上方」这件事本身就是把答案
# 编码进了观测。
#
# 所以改成：**由 `initial_layout` 的物体包围盒 + 台面高度算一个固定的俯视机位**。
# 它只依赖初始场景（选手本来就能从第一帧看到的东西），与 case 内容、与目标
# 都无关，而且**参考图与实时帧用同一个机位**（选手才能把两者对比）。
_CAMERA_FOV_RAD = 1.15         # 视场角（画面是正方形，两方向一样）
_OBJECT_SIZE_PAD_M = 0.18      # 物体尺寸余量：参考点只代表一个角，实体还会往外伸
# 俯角：从竖直方向偏这么多度。**不能接近 0** —— 相机竖直向下时视线与
# `up=[0,0,1]` 几乎平行，`look_at` 的基向量退化，framing 会歪
# （实测：正下方俯视时只拍到两个物体，第三个出界）。
_CAMERA_TILT_RAD = 0.62        # ≈35.5°
_CAMERA_UP = [0.0, 0.0, 1.0]


def _layout_points_cm(bundle, seed, *, tabletop_drop_cm=45.0):
    """某个 seed 的初始布局里，**同一层**物体的位置（场景 cm 坐标）。

    ⚠️ `position` 的语义**不统一**，而且**不能按 `static_objects` 或
    `objects[*].static` 过滤**（实测踩过两次）：

    | 现象 | 为什么 |
    |---|---|
    | 用全部物体算包围盒 → 机位飞到天上、画面里只剩一个巨大的杯子 | 家具（`white_square_table_000`）的 `position` 是**几何中心**（z=76.4cm），任务物体的 `position` 是**底面**（z≈155~160cm）。差 80cm，包围盒被拉扁 |
    | 按 `static_objects` 过滤 → 又漏掉 `black_box_on_table_000` | 这个 case 里 `black_box` **是**参照物（`static_objects` 里），但它就摆在桌面上，少框一个物体画面就歪 |

    **判据改成位置**：只保留「最高的那一层往下 `tabletop_drop_cm` 以内」的物体。
    实测 73 个 case 的 z 分布都有明显断层（家具在地面附近，任务物体在桌面），
    45cm 这个阈值能稳定切开（`work/reports/54_pitfalls.md` F5）。
    """
    layouts = bundle.get('initial_layouts') or []
    index = seed if 0 <= seed < len(layouts) else 0
    layout = layouts[index] if layouts else {'objects': []}
    points = [np.asarray(o['position'], dtype=float) for o in layout['objects']]
    if not points:
        return np.zeros((1, 3))
    points = np.asarray(points)
    cutoff = float(points[:, 2].max()) - float(tabletop_drop_cm)
    kept = points[points[:, 2] >= cutoff]
    return kept if len(kept) else points


def _fit_distance(points, fov, *, pad_m):
    """求能框住 `points` 的最小正俯视距离（米），返回 `(aim, distance)`。

    ## 为什么用包围球，而不是「逐点判视锥」

    逐点判视锥**漏了物体本身的尺寸** —— 按参考点拟合出的距离，
    渲染出来物体的**边角会被切掉**（实测：case 004 的显示器被切成两半）。
    参考点只是物体的一个角，实体从那里向外延伸。

    包围球一次算清：

    ```text
    distance = (包围球半径 + pad_m) / sin(fov/2)
    ```

    ## ⚠️ 一处仍未确认的事实：`fov` 的单位

    `CameraConfig.fov` 的文档只写「float」，**没说弧度还是角度**
    （ManiSkill 源码只做 `float` 校验）。所以我不赌 —— 用**实测反推**：
    当前 `_CAMERA_FOV_RAD = 1.15` 在这台机器上渲染出来的画面，
    框住范围与「1.15 弧度」的预期一致（≈66°）。

    **要复核的话**：拿一个 case 渲染，量画面里物体的像素占比是否
    与 `2*atan(span/2/dist)` 算出来的一致。记在
    `work/reports/54_pitfalls.md` 的 F5。
    """
    aim = points.mean(axis=0)
    radius = float(np.linalg.norm(points - aim, axis=1).max()) + float(pad_m)
    distance = radius / max(np.sin(fov / 2.0), 1e-6)
    return aim, max(distance, 0.35)


def fixed_camera_from_bundle(bundle, seed, *, include_robot_base=True):
    """给 `create_case_env` 用的固定机位规格（或 None = 不开相机）。

    返回 `{'center_m','eye_m','fov_rad','aim_m','distance_m'}`，单位是**米**。

    ## 为什么是「正俯视 + 拟合距离」

    试过带俯角的斜视机位，**在单个 case 上调好、换个 case 就出画**：
    物体散布方向不固定（有的沿 x 铺开、有的沿 y），而斜视会让
    **靠近相机那一侧的物体**被放得极大、其余挤出画面
    （实测：斜视时 `black_box` 离相机只有 8cm，几乎糊满整张图）。
    我在这上面花了几轮（见 `work/reports/54_pitfalls.md` F5）。

    正俯视没有这个问题 —— 所有物体到相机的**竖直**距离都是同一个，
    退化问题（视线与 `up` 平行）用「场景主轴」当 `up` 参考轴解决。

    距离用**在候选值上二分**求「能框住所有目标点的最小高度」，
    而不是套公式：物体的散布方向不固定，公式调好一个 case 就废。

    ## 两条硬约束（模式 B）

    1. **只依赖初始布局**，不碰 `target_layout` —— 把相机摆到目标上方
       等于把答案编码进观测。
    2. **参考图与实时帧用同一个规格** —— 选手才能把两者对比。
    """
    points_cm = _layout_points_cm(bundle, seed)
    if include_robot_base:
        base = bundle.get('robot_base_candidate_m')
        if base is not None:
            points_cm = np.vstack([points_cm, np.asarray(base, dtype=float) * 100.0])
    points = points_cm / 100.0

    low, high = points.min(axis=0), points.max(axis=0)
    # 画面 `up` 参考轴：取场景**跨度更大**的那个水平轴。
    # ⚠️ 不能直接用世界 `[0,0,1]` —— 正俯视时它与视线平行，
    #    `look_at` 的基向量退化、framing 会歪（实测）。
    span_xy = np.array([high[0] - low[0], high[1] - low[1]])
    up_ref = ([1.0, 0.0, 0.0] if span_xy[0] >= span_xy[1] else [0.0, 1.0, 0.0])

    aim, distance = _fit_distance(points, _CAMERA_FOV_RAD, pad_m=_OBJECT_SIZE_PAD_M)
    return {'center_m': aim.tolist(),
            'eye_m': (aim + np.array([0.0, 0.0, distance])).tolist(),
            'fov_rad': _CAMERA_FOV_RAD,
            'aim_m': aim.tolist(),
            'distance_m': distance,
            'up': up_ref}


def _visual_parts(bundle, *, enable, scenes_root=None, k=None):
    """取一例的**真实颜色**视觉块（第 7 步）。

    ⚠️ 三条实测结论（见 `work/reports/57_pitfalls_pvsr.md`）：

    1. `add_visual_from_file(GLB)` 在 SAPIEN 3.0.3 里**忽略 `COLOR_0` 与内嵌材质**，
       一律出灰 —— 所以不能简单地「把 GLB 换成 OBJ 就行」。
    2. 但 `RenderShapeTriangleMesh(V,F,N,UV,mat)` 手工挂 `RenderBodyComponent`
       **是出颜色的** —— 所以颜色要自己搬：按顶点色聚类成单色块。
    3. 换算到局部必须用 **`target_layout`**，不是当前 seed 的 initial 布局。

    ## 为什么是 `target_layout`（这条踩过）

    GLB 顶点烘焙的是**最终优化场景**的世界位姿，即 `V_world = R_target @ V_local
    + P_target`。所以只有用 `target_layout` 反解，得到的才是**物体固有的局部形状**：

    ```text
    V_local = R_targetᵀ ( Cᵀ V_glb_world − P_target )
    ```

    用 initial 布局反解等于「把最终几何按初态位姿又转了一次」，形状被多转一个
    角度。全包实测（448 个物体，视觉 AABB 相对碰撞 hull 的越界量）：

    | 用哪套位姿 | 中位 | p95 | 最大 | 越界 >5cm 的比例 |
    |---|---|---|---|---|
    | `initial_layouts[0]` | 0.0029 | 0.1654 | **0.3372** | **23.2%** |
    | `contract.target_layout` | **0.0000** | **0.0000** | **0.0112** | **0%** |
    | `poses.json`（Y-up 世界）| 1.9407 | 2.5466 | 2.9771 | 100% |

    （第三行不是它的错 —— `poses.json` 给的是世界位姿，直接当局部用当然全错。）

    **局部形状**与 actor 位姿无关，所以挂上去之后画面会**跟着物理走**：
    选手看到的就是当前布局下的真实几何。

    **失败不降级**：拿不到新资产就抛错。静默退回「猜颜色」的经历，
    正是第 7 步要消灭的东西 —— 悄悄退回去会让人以为图是新包的。
    """
    from track2_judge.scenario.visual_asset import (
        DEFAULT_CLUSTERS, VisualAssetError, case_assets,
    )
    if not enable:
        return None
    target = bundle.get('contract', {}).get('target_layout', {}).get('objects')
    if not isinstance(target, list) or not target:
        raise VisualAssetError('bundle has no contract.target_layout to derive '
                               'visual geometry from')
    layout = {row['instance']: row for row in target}
    try:
        return case_assets(bundle.get('case_id') or _case_id_from(bundle), layout,
                           root=scenes_root, k=k or DEFAULT_CLUSTERS)
    except VisualAssetError:
        raise
    except Exception as exc:                                  # noqa: BLE001
        raise VisualAssetError(f"visual asset load failed: {exc!r}") from exc


def _case_id_from(bundle):
    """老 bundle 没有 `case_id` 字段时，从 `source_case_id` 推。"""
    value = bundle.get('source_case_id') or bundle.get('task_id')
    if not isinstance(value, str) or not value:
        raise RuntimeError('bundle carries no case identity for visual assets')
    return value


def _attach_visual(actor, name, visual):
    """把一例的**真实颜色色块**挂到 actor 上（第 7 步）。

    每个色块一个 `RenderShapeTriangleMesh`，全部挂到**同一个**
    `RenderBodyComponent` 里 —— 这样它们在物理上仍是同一个刚体，
    只是画的时候分几次 draw call。

    ⚠️ 挂的是**物体固有的局部几何**（`visual_asset` 用 `target_layout` 反解出来的，
    与 actor 当前位姿无关），所以换 seed / 换布局时画面会**跟着物理走**。
    """
    from sapien.render import (RenderBodyComponent, RenderMaterial,
                               RenderShapeTriangleMesh)
    parts = visual.parts.get(name)
    if not parts:
        return False
    component = RenderBodyComponent()
    for vertices, faces, color in parts:
        component.attach(RenderShapeTriangleMesh(
            vertices.copy(), faces.copy(),
            np.zeros((len(vertices), 3), np.float32),
            np.zeros((len(vertices), 2), np.float32),
            RenderMaterial(base_color=[float(c) for c in color] + [1.0])))
    for entity in (actor._objs if hasattr(actor, '_objs') else [actor]):
        entity.add_component(component)
    return True


def create_case_env(bundle,casebank,seed=0,solver=(30,4),*,allow_unapproved_diagnostics=False,diagnostic_camera=None):
    from track2_judge.contract.profile import PROFILE,validate
    if bundle['contract'].get('scoring_profile')==PROFILE:validate(bundle['contract'])
    elif not allow_unapproved_diagnostics:raise RuntimeError("diagnostic profile explicit opt-in required")
    DATA=Path(casebank).resolve(strict=True)
    if DATA.stat().st_uid!=os.getuid() or DATA.stat().st_mode & 0o077:raise PermissionError("private casebank must be judge-owned 0700")
    LAYOUT={r['instance']:r for r in bundle['initial_layouts'][seed]['objects']}
    # 视觉源（第 7 步）：只在**真要出图**时加载，且**按 case 惰性**。
    # ⚠️ 用的是 `target_layout`（GLB 顶点烘焙的就是它），不是当前 seed 的
    #    initial 布局 —— 见 `_visual_parts` 的说明。
    VISUAL=(_visual_parts(bundle,enable=True,
                          scenes_root=os.environ.get('TRACK2_SCENES_ROOT') or None)
            if diagnostic_camera is not None else None)
    class CaseEnv(BaseEnv):
        SUPPORTED_ROBOTS=[PiperPrototype.uid]
        @property
        def _default_sensor_configs(self):return []
        @property
        def _default_human_render_camera_configs(self):
            if diagnostic_camera is None:return []
            from mani_skill.sensors.camera import CameraConfig
            from mani_skill.utils import sapien_utils
            # 固定机位（第 5 步）：center/height/fov 由 `fixed_camera_from_bundle`
            # 从**初始布局**算出，与目标位姿无关。名字保留 `diagnostic_public_rgb`
            # 是为了不动既有取图代码路径（`camera_rgb()` 按这个名字取）。
            center=np.asarray(diagnostic_camera['center_m'],dtype=float)
            fov=float(diagnostic_camera.get('fov_rad',.9))
            # `eye_m` 由 `fixed_camera_from_bundle` 给出（含俯角）。
            # 没有就退回「正上方一点」——那只是给老调用方的兜底。
            eye=np.asarray(diagnostic_camera.get('eye_m',
                                                (center+np.array([0.,-0.2,0.8])).tolist()),
                           dtype=float)
            up=np.asarray(diagnostic_camera.get('up',_CAMERA_UP),dtype=float)
            return CameraConfig('diagnostic_public_rgb',sapien_utils.look_at(eye,center,up=up),256,256,fov,.01,20.0)
        def evaluate(self):return {}
        def _get_obs_extra(self,info):return {}
        def _load_agent(self,options):super()._load_agent(options,sapien.Pose(bundle['robot_base_candidate_m'],bundle.get('robot_base_candidate_q_wxyz',[1,0,0,0])))
        def _load_scene(self,options):
            self.objects={};self.metadata={};self.load_evidence={}
            initial=LAYOUT
            material=sapien.physx.PhysxMaterial(static_friction=.5,dynamic_friction=.5,restitution=0)
            for name,item in bundle['objects'].items():
                b=self.scene.create_actor_builder()
                for ref in item['hulls']:
                    p=(DATA/ref['file']).resolve(strict=True)
                    if DATA not in p.parents or p.is_symlink():raise ValueError('private asset containment')
                    b.add_convex_collision_from_file(str(p),material=material,density=1000)
                pose=initial[name];b.set_initial_pose(sapien.Pose(np.array(pose['position'])/100,np.array(pose['rotation'])[[3,0,1,2]]))
                a=b.build_static(name=name) if item['static'] else b.build(name=name)
                body=a._bodies[0]
                if len(body.collision_shapes)!=len(item['hulls']):raise RuntimeError('engine collision shape count mismatch')
                if not item['static']:body.mass=item['mass_kg'];body.inertia=item['inertia_kg_m2'];body.set_solver_position_iterations(solver[0]);body.set_solver_velocity_iterations(solver[1])
                if VISUAL is not None:
                    _attach_visual(a,name,VISUAL)
                self.objects[name]=a;self.metadata[name]={'pivot_m':[0,0,0],'pivot_scaled':True}
                # Source canonical pivot is retained in bundle for private inverse
                # provenance; scoring uses this explicitly rebased asset root.
                self.load_evidence[name]={'dynamic':not item['static']}
            # Visible, static scene support fixtures from initial task configuration.
            # Built only during normal scene loading; never exposed as a Skill setter.
            self.spatial_actors={}
            for fixture in bundle.get('initial_static_supports',{}).get(str(seed),[]):
                name=fixture['name']
                if not name.startswith('world_support_') or name in self.objects:raise ValueError('support identity')
                half=np.asarray(fixture['half_size_m'],dtype=float);pos=np.asarray(fixture['position_m'],dtype=float)
                if half.shape!=(3,) or pos.shape!=(3,) or not np.isfinite(np.r_[half,pos]).all() or np.any(half<=0) or np.any(half>2):raise ValueError('support geometry')
                builder=self.scene.create_actor_builder();builder.add_box_collision(half_size=half,material=material)
                if diagnostic_camera is not None:builder.add_box_visual(half_size=half,material=sapien.render.RenderMaterial(base_color=[.55,.55,.55,1]))
                builder.set_initial_pose(sapien.Pose(pos));support=builder.build_static(name=name)
                self.spatial_actors[name]=support;self.metadata[name]={'pivot_m':[0,0,0]}
        def _initialize_episode(self,env_idx,options):self.agent.reset(np.array(bundle.get('initial_robot_qpos',REST),dtype=np.float32))
    return CaseEnv(num_envs=1,robot_uids=PiperPrototype.uid,control_mode='pd_joint_pos',sim_backend='cpu',render_backend='none' if diagnostic_camera is None else 'sapien_cuda:0',obs_mode='none',reward_mode='none',sim_config={'sim_freq':240,'control_freq':20},enhanced_determinism=True)
