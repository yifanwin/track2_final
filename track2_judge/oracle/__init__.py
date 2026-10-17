"""裁判侧 Oracle 的 GT 规划（第 6 步从 `experiments/` 搬进来）。

## 为什么单独开一个包

原来的 Oracle 住在 `experiments/Frozen73_Final_Readiness_20261014/`，靠
`sys.path.insert` 拉一批**平铺模块**（`case_backend` / `adapter` / `common` …）。
第 6 步要让它走 `release_entry.py`，就不能再依赖那套平铺 import。

搬进来的只有**纯规划逻辑**（几何 + IK + 轨迹），以及它的两个依赖：

| 来源 | 去向 |
|---|---|
| `oracle_planner.py` | `oracle_planner.py`（只改 import） |
| `oracle_geometry.py` | `oracle_geometry.py`（只改 import） |
| `common.array` | → `track2_judge.util.common.array` |
| `kinematics`（原在 SOURCE 根） | 已搬进 `track2_judge/scenario/kinematics.py` |

**没有搬的**（也不需要）：`oracle_driver.py` 里那套 `PrivateOracleInput` /
`OracleAdapter` —— 那正是「绕过入口」的部分，第 6 步要把它去掉。
"""
