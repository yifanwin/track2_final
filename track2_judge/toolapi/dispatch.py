"""Trusted Runner API prototype; JSON data boundary, NOT a process sandbox."""
import copy
from numbers import Real
import numpy as np
from track2_judge.util.common import array,plain

#: health 采样频率（Hz）。2026-10-18 定：**240 → 10**。
#:
#: 240 Hz（逐物理子步）的完整快照占掉真策略 episode 墙钟的 ~62 %，而这套检测器
#: 在 73 例全量里**零告警**、且没有任何判定消费者
#: （`physics_health_diagnostic.json` 无读者）。降到 10 Hz：
#: 相机 ON 配置 163.0 → 64.5 ms/步 ⇒ 单 episode 124.7 s → 49.3 s。
#: 见 `report/plan/Track2_评测加速_代码改动清单.md`。
HEALTH_HZ = 10.0

class PhysicsExecutionError(RuntimeError): pass

class ManiSkillSimulatorAdapter:
    def __init__(self,backend_factory,*,max_steps=2400,max_sim_time_s=120,snapshot=None,metadata=None,scene_settings=None,public_fields=(),observation_provider=None,spatial_provider=None,event_policy=None,checkpoints=(),recovery=False,physics_health=True,initial_validation=None):
        if type(max_steps) is not int or max_steps<=0 or not np.isfinite(max_sim_time_s) or max_sim_time_s<=0:raise ValueError('invalid budget')
        self.__initial_validation=initial_validation;self.__initial_evidence=None
        self.__factory=backend_factory;self.__env=None;self.__steps=0
        self.__max_steps=max_steps;self.__max_time=float(max_sim_time_s);self.__inputs={};self.__previous=None
        self.__snapshot=snapshot;self.__metadata=metadata;self.__settings=scene_settings
        self.__public_fields=set(public_fields)
        self.__observation_provider=observation_provider;self.__spatial_provider=spatial_provider
        self.__event_policy=event_policy;self.__checkpoint_contract=checkpoints;self.__recovery=recovery
        self.__event_recorder=None;self.__original_hook=None;self.__expected_frames=0
        self.__health_enabled=bool(physics_health);self.__health_recorder=None;self.__command=None
    def reset(self,*,seed=20261012,public_task_inputs=None):
        """Judge-only: caller supplies the approved public input contract."""
        if type(seed) is not int:raise ValueError('seed must be integer')
        inputs=copy.deepcopy(public_task_inputs or {})
        if set(inputs)-self.__public_fields:raise ValueError('unapproved public input')
        import json
        inputs=json.loads(json.dumps(plain(inputs),allow_nan=False))
        if self.__observation_provider is not None:inputs=self.__observation_provider.prepare(inputs)
        self.close(); self.__env=self.__factory();self.__env.reset(seed=seed)
        if self.__initial_validation is not None:self.__initial_evidence=self.__initial_validation(self.__env)
        self.__steps=0;self.__inputs=inputs
        q=array(self.__env.agent.robot.get_qpos());self.__previous=np.r_[q[:6],q[6]-q[7]]
        self.__expected_frames=0
        self.__health_tick=0;self.__health_frames=0;self.__expected_health_frames=0
        if self.__observation_provider is not None:self.__observation_provider.capture_initial(self.__env)
        self.__health_recorder=None;self.__command=None;self.__declared_velocity_limits=None
        #: health 的采样节流（2026-10-18）：每 N 个物理子步采一次。
        #: `__health_tick` 是子步计数，`__health_frames` 是**实际**采样次数，
        #: `__expected_health_frames` 是**应有**次数（= 子步数 // N）——
        #: 两者分开是为了让 `coverage` 仍能发现「钩子没挂上」。
        self.__health_decimation=None;self.__health_tick=0
        self.__health_frames=0;self.__expected_health_frames=0
        scene_config=getattr(getattr(self.__env,'sim_config',None),'scene_config',None)
        if self.__health_enabled and hasattr(self.__env,'_after_simulation_step') and scene_config is not None:
            from track2_judge.diagnostics.physics_health import TrustedPhysicsHealthRecorder
            options=copy.deepcopy(getattr(self.__env,'physics_health_diagnostic_options',{}))
            if set(options)-{'speed_alert_m_s','gravity_residual_alert_m_s2','penetration_alert_m','point_impulse_alert_Ns','support_xy_bounds','evidence_frames'}:raise ValueError('invalid private health options')
            # ★ 2026-10-18：`dt` 必须等于**真实采样间隔**，不是仿真步长。
            #   记录器的 `ACCELERATION_ALERT` 用 `(v - v_prev)/dt` 算加速度；
            #   采样间隔 0.1 s 而 dt 仍填 1/240 时残差会被放大 24 倍
            #   → **持续误报** `CONTACT_FREE_ACCELERATION_ALERT`
            #   （合成证明见 `report/plan/Track2_评测加速_代码改动清单.md` §1.1）。
            _sim_freq=float(self.__env.sim_config.sim_freq)
            self.__health_decimation=max(1,int(round(_sim_freq/HEALTH_HZ)))
            self.__health_recorder=TrustedPhysicsHealthRecorder(dt=self.__health_decimation/_sim_freq,gravity=scene_config.gravity,**options)
            try:
                import xml.etree.ElementTree as ET
                tree=ET.parse(self.__env.agent.urdf_path)
                limits={j.get('name'):float(j.find('limit').get('velocity')) for j in tree.findall('joint') if j.find('limit') is not None}
                self.__declared_velocity_limits=[limits[j.name] for j in self.__env.agent.robot.get_active_joints()]
            except (OSError,ValueError,TypeError,AttributeError,KeyError):self.__declared_velocity_limits=None
        if self.__event_policy is not None or self.__checkpoint_contract:
            from track2_judge.contract.events import TrustedEventRecorder
            if not hasattr(self.__env,'_after_simulation_step') or self.__spatial_provider is None:raise RuntimeError('physical substep hook/evidence unavailable')
            self.__event_recorder=TrustedEventRecorder(self.__event_policy,checkpoints=self.__checkpoint_contract,recovery=self.__recovery)
        if self.__health_recorder is not None or self.__event_recorder is not None:
            self.__original_hook=self.__env._after_simulation_step
            # ★ 2026-10-18：两条分支**各走各的节流**，不再共用一次快照构造。
            #
            #   * 事件链的 coverage 判据是 `frames == expected_frames == 子步数`
            #     （`scenario/prototype_env_v2.py:87`），**必须**保持每子步；
            #   * 健康链是纯诊断，降到 10 Hz 即可。
            #
            #   共用一个 `frames` 会把两者绑死 —— 改一个坏另一个。
            def after_step():
                self.__original_hook()
                # ── 事件分支：每子步，语义不变
                if self.__event_recorder is not None:
                    raw=self.__read_physical_state()
                    raw['trusted_physics_substep']=True
                    import json
                    state=json.loads(json.dumps(plain(raw),allow_nan=False))
                    self.__event_recorder.sample(state)
                    return
                if self.__health_recorder is None:
                    return
                # ── 健康分支：每 `__health_decimation` 个子步采一次（10 Hz）。
                #    跳过时**连快照都不算** —— 省的就是这一步。
                self.__health_tick+=1
                if self.__health_tick % self.__health_decimation:
                    return
                if self.__snapshot is None:
                    from track2_judge.scenario.prototype_backend import physical_snapshot
                    raw=physical_snapshot(self.__env)
                else: raw=self.__snapshot(self.__env)
                raw['trusted_physics_substep']=True
                self.__health_frames+=1
                health_state=raw.copy();health_state['judge_control']={}
                robot=self.__env.agent.robot
                if hasattr(robot,'get_drive_targets'):health_state['judge_control']['drive_targets']=array(robot.get_drive_targets())
                if hasattr(robot,'get_qf'):health_state['judge_control']['qf']=array(robot.get_qf())
                if self.__declared_velocity_limits is not None:health_state['judge_control']['declared_velocity_limits']=self.__declared_velocity_limits
                self.__health_recorder.sample(health_state,command=self.__command)
            self.__env._after_simulation_step=after_step
        return self.observe()
    def observe(self, *, include_rgb=True):
        self.__require()
        # CPU/headless prototype exposes proprioception and explicitly approved
        # task inputs only. No object GT or mutable simulator object escapes.
        result=plain({'robot':{'joint_position':array(self.__env.agent.robot.get_qpos()),'joint_velocity':array(self.__env.agent.robot.get_qvel()),'tcp_position_m':array(self.__env.agent.tcp.pose.p),'tcp_quaternion_wxyz':array(self.__env.agent.tcp.pose.q)},'task_inputs':copy.deepcopy(self.__inputs),'control_steps':self.__steps,'simulation_time_s':self.__steps*self.__env.control_timestep})
        result['robot']['joint_limits_rad']=array(self.__env.agent.robot.get_qlimits()).reshape(-1,2)[:6].tolist()
        if include_rgb and self.__observation_provider is not None:
            result.update(self.__observation_provider.observation())
        return result
    # ── 只读件（chunk 的预校验复用它们，避免两份逻辑漂移）
    #
    # ★ 2026-10-18 强制 Action Chunk：chunk 必须在**推进任何物理之前**整条校验，
    #   否则会「前半段动了、后半段越限」→ 半途撞预算/越限 → BLOCKED（见 F1）。
    #   所以把 `step()` 里的四个判据抽出来，`step()` 与 chunk 都用同一份。

    @property
    def control_timestep(self):
        self.__require()
        return float(self.__env.control_timestep)

    def current_command_target(self):
        """当前命令目标（上一帧下发值），形如 6 关节 + 夹爪开度。"""
        self.__require()
        return None if self.__previous is None else list(self.__previous)

    @property
    def control_steps(self):
        """**episode 累计**推进的控制步数（从 `reset` 起算）。

        ★ 2026-10-19：`control_steps` 这个字段名在三个入口上的语义**不一致**，
        这条 q 是统一它们的基础设施。见 `docs/Track2_提交到评测_全链路架构与数据流.md`
        的「字段语义」节与 `plan/Track2_字段语义统一_修复方案.md`。

        | 入口 | 修前 | 修后 |
        |---|---|---|
        | `observe` | 累计 | 累计（不变） |
        | `joint_trajectory` | **本次增量** | 累计 |
        | `gripper_control` | 累计（走 `observe()`） | 累计（不变） |

        选手侧不需要它 —— 它是给 gateway/adapter 内部把增量换算成累计用的。
        """
        self.__require()
        return int(self.__steps)

    def assert_within_limits(self,raw,opening):
        """限位 + 形状 + 有限性。不通过抛 `ValueError`。"""
        self.__require()
        if not isinstance(raw,(list,tuple,np.ndarray)) or np.ndim(raw)!=1 or any(not isinstance(x,Real) or isinstance(x,(bool,np.bool_)) for x in raw) or not isinstance(opening,Real) or isinstance(opening,(bool,np.bool_)):raise ValueError('numeric values required')
        try:q=np.asarray(raw,dtype=float);g=float(opening)
        except (ValueError,TypeError):raise ValueError('numeric values required') from None
        if q.shape!=(6,) or not np.isfinite(q).all() or not np.isfinite(g):raise ValueError('dimension or finite check failed')
        lim=array(self.__env.agent.robot.get_qlimits()).reshape(-1,2)[:6]
        if np.any(q<lim[:,0]) or np.any(q>lim[:,1]) or not 0<=g<=.1:raise ValueError('outside Piper limits')
        return q,g

    def assert_within_slew(self,previous,raw,opening):
        """步间限幅。`previous` 为 7 维（6 关节 + 夹爪），来自 `current_command_target()`。"""
        q,g=self.assert_within_limits(raw,opening)
        if previous is None:return q,g
        target=np.r_[q,g];dt=float(self.__env.control_timestep)
        # Conservative internal command slew budget, below URDF 5 rad/s, 3m/s.
        # 步间限幅 = 1.0 rad/s × dt(=1/20 s) = **0.05 rad/步**（臂）；
        # 夹爪 0.12 m/s × dt = 0.006 m/步。
        if np.any(np.abs(target-np.asarray(previous,dtype=float))>np.r_[[1.]*6,.12]*dt+1e-7):
            import sys as _s
            _d=np.abs(target-np.asarray(previous,dtype=float)); _l=np.r_[[1.]*6,.12]*dt
            print("[track2-slew] dims=%s diff=%s lim=%s prev=%s tgt=%s"%(
                [int(i) for i in np.where(_d>_l+1e-7)[0]],
                np.round(_d,4).tolist(), np.round(_l,4).tolist(),
                np.round(np.asarray(previous,dtype=float),4).tolist(), np.round(target,4).tolist()),
                file=_s.stderr, flush=True)
            raise ValueError('target slew exceeded')
        return q,g

    def assert_budget_allows(self,count=1):
        """剩余预算是否容得下 `count` 步。容不下抛 `RuntimeError('control budget exhausted')`。"""
        self.__require()
        dt=float(self.__env.control_timestep)
        if self.__steps+count>self.__max_steps or (self.__steps+count)*dt>self.__max_time+1e-9:raise RuntimeError('control budget exhausted')

    def step(self,action):
        self.__require()
        if not isinstance(action,dict) or set(action)!={'joint_position','gripper_opening_m'}:raise ValueError('only robot joint targets accepted')
        raw=action['joint_position']; opening=action['gripper_opening_m']
        q,g=self.assert_within_slew(self.__previous,raw,opening)
        self.assert_budget_allows(1)
        target=np.r_[q,g]
        expected=int(getattr(self.__env,'_sim_steps_per_control',0))
        self.__expected_frames+=expected
        # health 的应有采样数按**同一个节流**算（2026-10-18）。
        # ⚠️ **不要**把 expected 直接填成实际采样数 —— 那会让 `coverage` 退化成
        #    恒真，丢掉「钩子还在不在」这条信息。
        if self.__health_decimation:
            self.__expected_health_frames=self.__expected_frames//self.__health_decimation
        self.__command={'joint_position':q.tolist(),'gripper_opening_m':g}
        try:
            self.__env.step(np.r_[q,g/2,-g/2].astype(np.float32))
            self.__steps+=1;self.__previous=target.copy()
            # Action responses never expose RGB. Rendering both cameras on each
            # internal control step wastes the action deadline (25x per chunk).
            # The next explicit observe captures the current physical state.
            return self.observe(include_rgb=False)
        except Exception: raise PhysicsExecutionError('physical control/observation failed') from None
    def __read_physical_state(self):
        self.__require()
        if self.__snapshot is None:
            from track2_judge.scenario.prototype_backend import physical_snapshot
            snapshot=physical_snapshot
        else:snapshot=self.__snapshot
        import json
        value=snapshot(self.__env)
        if self.__spatial_provider is not None:value=self.__spatial_provider(self.__env,value)
        return value
    def get_final_state(self):
        """Judge-only API. No private health or state crosses Skill RPC."""
        import json
        return json.loads(json.dumps(plain(self.__read_physical_state()),allow_nan=False))
    def get_initialization_evidence(self):return copy.deepcopy(self.__initial_evidence)
    def get_physics_health(self):
        if self.__health_recorder is None:return {'status':'NOT CONFIGURED','coverage':'NOT AVAILABLE','threshold_scope':'no synthetic health claimed'}
        return self.__health_recorder.report(self.__expected_health_frames)
    def get_physics_health_private_evidence(self):
        return self.__health_recorder.private_evidence() if self.__health_recorder is not None else {}
    def get_event_evidence(self,enabled):
        if self.__event_recorder is None:
            if enabled:raise RuntimeError('trusted event collector absent')
            return {'event_counts':{},'coverage':'NOT CONFIGURED','P5':{'status':'NOT CONFIGURED'}}
        return self.__event_recorder.report(enabled,self.__expected_frames)
    def get_layout(self):
        from track2_judge.diagnostics.state_export import export_layout
        return export_layout(self.get_final_state()['objects'],self.__metadata,self.__settings)
    def close(self):
        if self.__env is not None and self.__original_hook is not None:self.__env._after_simulation_step=self.__original_hook
        if self.__env is not None:self.__env.close();self.__env=None
        self.__original_hook=None;self.__event_recorder=None
    def dispatch_public(self,method,payload=None):
        """Future RPC allowlist; process isolation is a separate unfinished task."""
        if method=='observe' and payload is None:return self.observe()
        if method=='step':return self.step(payload)
        if method=='piper_joint_control':return self.step(payload)
        # ★ 2026-10-18 强制 Action Chunk：本地 adapter 也认 `joint_trajectory`
        #   （`payload={'targets':[...]}`）。它与 gateway 侧走同一条语义：
        #   整段先校验、整段执行。旧名 `joint_control` **不再接受**。
        if method=='joint_trajectory':
            targets=(payload or {}).get('targets')
            if not isinstance(targets,list):raise ValueError('joint_trajectory requires targets list')
            executed=0
            for item in targets:
                self.step({'joint_position':item['joint_position'],
                           'gripper_opening_m':item['gripper_opening_m']})
                executed+=1
            return {'control_steps':executed,
                    'simulation_time_s':executed*float(self.__env.control_timestep)}
        if method=='piper_gripper_control':
            if not isinstance(payload,dict) or set(payload)!={'gripper_opening_m'}:raise ValueError('gripper schema rejected')
            return self.step({'joint_position':self.__previous[:6].tolist(),'gripper_opening_m':payload['gripper_opening_m']})
        raise ValueError('method is judge-only or unsupported')
    def __require(self):
        if self.__env is None:raise RuntimeError('adapter not reset or already closed')


# ───────────────────────────────────────── Gateway 的执行体适配

class GatewayExecutor:
    """把 `ManiSkillSimulatorAdapter` 适配成 `toolapi.gateway.ToolExecutor`。

    职责只有一件：**把 adapter 的原始观测/动作结果，整理成 ToolSpec 声明的形状**。
    校验与账务都在 Gateway 侧，这里不重复做。

    `is_ready()` 供 Gateway 的 `/tools/{id}/context` 判断 readiness ——
    adapter 没 reset 过时 PAOS 会看到 `ready: false`。
    """

    def __init__(self, adapter):
        self._adapter = adapter
        self._observations = 0

    def is_ready(self) -> bool:
        return getattr(self._adapter, "_ManiSkillSimulatorAdapter__env", None) is not None

    def observe(self, arguments=None):
        arguments = arguments or {}
        raw = self._adapter.observe()
        return self._as_observation(raw, include_rgb=arguments.get("include_rgb", True))

    def step(self, arguments):
        q = list(arguments["joint_position"])
        g = float(arguments["gripper_opening_m"])
        returned = self._adapter.step({"joint_position": q, "gripper_opening_m": g})
        return self._as_action_result(returned)

    def gripper_step(self, arguments):
        # 只动夹爪：关节目标保持在**上一次的目标**上（adapter 的 __previous 语义）。
        g = float(arguments["gripper_opening_m"])
        returned = self._adapter.dispatch_public("piper_gripper_control",
                                                 {"gripper_opening_m": g})
        return self._as_action_result(returned)

    # ── 形状整理

    def _as_observation(self, raw, *, include_rgb):
        """把 adapter 的原始观测收窄成 ToolSpec 声明的形状。

        adapter 的 `observe()` 在挂了 `ObservationProvider` 时会多返回
        `camera_status`（图像到底拍没拍到）。那是**运行事实**，不是目标答案，
        所以保留它；但 `output_schema` 是 `additionalProperties: false`，
        只搬声明过的字段。
        """
        self._observations += 1
        out = {
            "observation_id": "obs_%06d" % self._observations,
            "captured_at_s": _monotonic_s(),
            "sim_time_s": float(raw.get("simulation_time_s", 0.0)),
            "control_steps": int(raw.get("control_steps", 0)),
            "robot": raw.get("robot", {}),
        }
        if raw.get("task_inputs") is not None:
            out["task_inputs"] = raw["task_inputs"]
        if include_rgb:
            images = {}
            # 内部原始观测的键仍是 `first_frame`（provider 的缓存名），
            # 对选手暴露的 ToolSpec 键是 `camera`。
            if raw.get("first_frame") is not None:
                images["camera"] = raw["first_frame"]
            if raw.get("camera_status") is not None:
                # 相机没配好时也要让选手看见 —— 否则他会以为图像是空的。
                images["camera_status"] = str(raw["camera_status"])
            if raw.get("wrist_frame") is not None:
                images["wrist_camera"] = raw["wrist_frame"]
            if raw.get("wrist_camera_status") is not None:
                images["wrist_camera_status"] = str(raw["wrist_camera_status"])
            if images:
                out["images"] = images
        return out

    def _as_action_result(self, raw):
        """把 adapter 的动作结果收窄成 **ToolSpec 声明的形状**。

        ★ 2026-10-19 修正两处（都在 `plan/Track2_字段语义统一_修复方案.md`）：

        1. **`control_steps` 一律是 episode 累计**（与 `observe` 同义）。
           修前 `joint_trajectory` 返回本次增量、`gripper_control` 返回累计 ——
           同一字段名两种语义，选手没法用它记账。
        2. **按 ToolSpec 决定带不带 `executed_steps`**。
           `joint_trajectory` 的 schema 声明了它、而 `gripper_control` 没有；
           两个 spec 都是 `additionalProperties: false`，所以多带一个键
           会被 `validate_output` **直接拒绝** —— 修前 `gripper_control`
           的每次调用都会因此失败（踩到过真 bug，见 §「既有 bug」）。

        这里按 `raw` 里**是否真的有**该键来带：adapter 的 gripper 路径
        （`dispatch_public` → `step()` → `observe()`）本来就不产出它。
        """
        out = {
            "control_steps": int(raw.get("control_steps", 0)),
            "sim_time_s": float(raw.get("simulation_time_s", 0.0)),
        }
        if raw.get("executed_steps") is not None:
            out["executed_steps"] = int(raw["executed_steps"])
        return out


def _monotonic_s():
    import time
    return time.monotonic()


SimulatorAdapter = ManiSkillSimulatorAdapter
