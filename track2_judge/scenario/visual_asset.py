"""新视觉资产源（第 7 步）：`zhangyexin/scenes/*/scene.glb` 的真实颜色。

## 为什么需要这个模块

旧 casebank 的 hull **只是碰撞几何**（`.obj`，无颜色）。当时的图是靠
**物体名字里的颜色词**上色的：

```python
for word, rgb in [('blue',…),('green',…),('red',…),('white',…)]:
    if word in name.lower(): color = rgb; break
```

模式 B 下这是硬伤 —— 选手要从图像认出目标物体，而所有白色物体都是同一个死白。
新包（PVSR 交付包）的 `scene.glb` 带**真实顶点颜色**（`COLOR_0`）。

## ⚠️ 实测：SAPIEN **不吃** GLB 的颜色（17 个探针的结论）

| 进法 | 结果 |
|---|---|
| `add_visual_from_file(*.glb)`，不传 material | 灰 |
| `add_visual_from_file(*.glb)`，传 `RenderMaterial` | **含 `COLOR_0` 时灰** / 不含时有色 |
| `add_visual_from_file(*.obj)`，传 `RenderMaterial` | 有色（旧代码一直走这条） |
| GLB 内嵌 `baseColorFactor` / `baseColorTexture` | 灰（被忽略） |
| `RenderShapeTriangleMesh(V,F,N,UV,mat)` 手工挂 | **有色** |
| `set_camera_shader_dir("vertex_color")` | 报错（Material 尺寸不符），不可用 |
| `RenderBodyComponent.shading_mode = 0..4` | 全灰，不是这个开关 |

**所以颜色只能自己搬**：读 GLB 的 `COLOR_0`，按**面平均色**聚类成 k 个单色块，
每块一个 `RenderShapeTriangleMesh` + 平色材质。新包的价值在于它给出了
「真实颜色」这个**真值**。

## 坐标：GLB 顶点在**世界**（Y-up、米），要转到 actor 局部

```text
C      = [[1,0,0],[0,0,1],[0,-1,0]]      # 行向量右乘：v @ C 把 (x,y,z) 变成 (x, −z, y)
V_local = ( V_glb_world @ C − P_role ) @ R_roleᵀ
```

（核对：把新包的 up `(0,1,0)` 送进去得到 `(0,0,1)` —— 正是 Z-up 的 up。）

### ⚠️ `R_role` / `P_role` 必须取 **`target_layout`**

GLB 顶点烘焙的是**最终优化场景**的世界位姿，即 `V_world = R_target V_local
+ P_target`。只有用 `target_layout` 反解，得到的才是**物体固有的局部形状**：

```text
V_local = ( V_glb_world @ C − P_target ) @ R_targetᵀ
```

用 initial 布局反解等于「把最终几何按初态位姿又转了一次」，形状会被多转一个
角度。全包实测（448 个物体，视觉 AABB 相对碰撞 hull 的越界量）：

| 用哪套位姿 | 中位 | p95 | 最大 | 越界 >5cm |
|---|---|---|---|---|
| `initial_layouts[0]` | 0.0029 | 0.1654 | 0.3372 | 23.2% |
| `contract.target_layout` | 0.0000 | 0.0000 | **0.0112** | **0%** |

**局部形状与 actor 位姿无关**，所以挂到 actor 上之后画面会**跟着物理走** ——
选手看到的就是当前布局下的真实几何（初始布局或目标布局都行）。

推导时 `source_canonical_pivot_m` 恰好抵消 —— **不需要它**。

**不能**把 GLB 直接当局部件挂上去，也**不能**再叠 `poses.json`
（它的位姿已烘进顶点）—— 两者都会让画面错位。

## 惰性加载与释放

一包 78 个 GLB 共 6 GB，不可能全驻留。这里**只保留最近一个 case**，
切 case 时显式丢弃（`_CACHE` 最多一条），并把 trimesh 场景对象解引用。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["SCENES_ROOT", "SceneAssets", "VisualAssetError",
           "case_assets", "clusters_for", "scenes_root"]

#: 新交付包的位置。可用 `TRACK2_SCENES_ROOT` 覆盖（交付/搬迁时用）。
_DEFAULT_SCENES_ROOT = os.environ.get(
    "TRACK2_SCENES_ROOT",
    "/mnt/SSD_3TB/jusheng/track2_maniskill_staging_20261010/zhangyexin/scenes")

#: Y-up → Z-up（右乘形式）。`v @ C` 把 (x,y,z) 变成 (x, z, −y)。
C_YUP_TO_ZUP = np.array([[1.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0],
                         [0.0, -1.0, 0.0]], dtype=np.float64)

#: 每个物体拆成多少个单色块。实测 5 块能保住「白桌子的深色桌腿」
#: 「显示器深色边框」这类可辨识特征，而 1 块（平均色）会抹平。
DEFAULT_CLUSTERS = 5

#: k-means 用的面采样上限。全量 2.9M 面直接迭代要 6.5 s，
#: 采样到 4 万面后 < 0.4 s，而簇心几乎不变（颜色分布是低维的）。
_KMEANS_SAMPLE = 40_000
_KMEANS_ITERS = 8


class VisualAssetError(RuntimeError):
    """视觉资产不可用。调用方决定是降级还是让这一例失败。"""


def scenes_root(root: Any = None) -> Path:
    """新交付包的根目录（含 `<case_id>/scene.glb`）。"""
    return Path(root if root is not None else _DEFAULT_SCENES_ROOT)


# ────────────────────────────────────────────── 色块聚类


def _kmeans_faces(face_color: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """对**面颜色**做 k-means，返回 `(labels, centers)`。

    采样初始化 + 采样迭代，最后把**全部**面指派到簇 —— 这是本模块唯一
    的耗时热点，所以刻意不依赖 sklearn（标定环境里没装）。
    """
    n = len(face_color)
    k = max(1, min(k, n))
    if k == 1:
        return np.zeros(n, np.int32), face_color.mean(0, keepdims=True)

    rng = np.random.default_rng(0)
    step = max(1, n // _KMEANS_SAMPLE)
    sample = face_color[::step].astype(np.float32)
    centers = sample[rng.choice(len(sample), size=min(k, len(sample)), replace=False)].copy()

    for _ in range(_KMEANS_ITERS):
        lab = ((sample[:, None, :] - centers[None]) ** 2).sum(-1).argmin(1)
        moved = np.stack([sample[lab == i].mean(0) if np.any(lab == i) else centers[i]
                          for i in range(len(centers))])
        if np.allclose(moved, centers, atol=0.5):
            centers = moved
            break
        centers = moved

    # 全量指派（一次矩阵运算，比逐面循环快两个数量级）
    labels = ((face_color[:, None, :] - centers[None]) ** 2).sum(-1).argmin(1).astype(np.int32)
    return labels, centers


def clusters_for(vertices: np.ndarray, faces: np.ndarray, colors: np.ndarray,
                 *, k: int = DEFAULT_CLUSTERS) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """把（带顶点色的）网格拆成 `k` 个**单色块**。

    `colors` 是逐顶点 uint8 RGB；先按面取平均（三角形在真实网格里颜色本就接近常量），
    再聚类。返回 `[(顶点, 面, 颜色 float 0..1), ...]`，按面数从多到少。

    每个色块的顶点做了「只保留用到的顶点」压缩 —— 3.0M 面的物体
    如果每块都带全量顶点，光上传显存就要命。
    """
    face_color = colors[faces].mean(1).astype(np.float32)
    labels, centers = _kmeans_faces(face_color, k)
    parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for i in np.argsort(-np.bincount(labels, minlength=len(centers))):
        sel = labels == i
        if not sel.any():
            continue
        used, inverse = np.unique(faces[sel], return_inverse=True)
        parts.append((np.asarray(vertices, np.float32)[used],
                      inverse.reshape(-1, 3).astype(np.uint32),
                      (centers[i] / 255.0).astype(np.float32)))
    return parts


# ────────────────────────────────────────────── GLB 读取


class SceneAssets:
    """一个 case 的视觉资产：`{物体名: [色块, ...]}`。

    **只在需要渲染时才建**（`create_case_env(..., diagnostic_camera=...)`），
    建完由 `env_factory` 交给 SAPIEN；本类不持有 GPU 资源。
    """

    __slots__ = ("case_id", "parts", "source", "faces", "seconds")

    def __init__(self, case_id: str, parts: dict[str, list], source: str,
                 faces: int, seconds: float):
        self.case_id = case_id
        self.parts = parts
        self.source = source
        self.faces = faces
        self.seconds = seconds

    def __contains__(self, name: str) -> bool:
        return name in self.parts

    def __len__(self) -> int:
        return len(self.parts)


_CACHE: dict[str, SceneAssets] = {}
_CACHE_ORDER: list[str] = []


def _evict(keep: str) -> None:
    """只保留最近一个 case —— 6 GB 不可能全驻留。"""
    for case_id in list(_CACHE):
        if case_id != keep:
            _CACHE.pop(case_id, None)
    _CACHE_ORDER[:] = [c for c in _CACHE_ORDER if c == keep]


def case_assets(case_id: str, layout: dict[str, dict], *, root: Any = None,
                k: int = DEFAULT_CLUSTERS, use_cache: bool = True) -> SceneAssets:
    """读 `<root>/<case_id>/scene.glb`，按 `layout` 转到 actor 局部后聚类。

    `layout` 是 `{物体名: {'position': [cm,cm,cm], 'rotation': [x,y,z,w]}}` ——
    与 casebank 的 `initial_layouts[i]['objects']` / `contract.target_layout
    ['objects']` 同一个形状（`position` 是**厘米**，`rotation` 是 **xyzw**）。

    没有 GLB 的 case（例如新包里的 5 个新增、或尚未迁移的 case）抛
    `VisualAssetError` —— **不静默降级**，因为降级会让人以为图是新包的。
    """
    import time

    if use_cache and case_id in _CACHE:
        return _CACHE[case_id]

    glb = scenes_root(root) / case_id / "scene.glb"
    if not glb.is_file():
        raise VisualAssetError(f"no scene.glb for {case_id}: {glb}")

    import scipy.spatial.transform as st
    import trimesh

    started = time.time()
    scene = trimesh.load(str(glb), force="scene")
    parts: dict[str, list] = {}
    faces = 0
    for name, item in layout.items():
        geometry = scene.geometry.get(name)
        if geometry is None or not hasattr(geometry, "faces"):
            continue
        vertices = np.asarray(geometry.vertices, np.float64)
        rotation = np.asarray(item["rotation"], np.float64)
        # casebank 的 `rotation` 就是 xyzw（旧的 `[[3,0,1,2]]` 是 wxyz 重排）
        R = st.Rotation.from_quat(rotation).as_matrix()
        P = np.asarray(item["position"], np.float64) / 100.0          # 厘米 → 米
        local = (vertices @ C_YUP_TO_ZUP - P) @ R
        triangles = np.asarray(geometry.faces, np.int64)
        colors = np.asarray(geometry.visual.vertex_colors, np.uint8)[:, :3]
        parts[name] = clusters_for(local, triangles, colors, k=k)
        faces += len(triangles)
    # 显式解引用：trimesh 的大数组不会自己走
    del scene

    assets = SceneAssets(case_id, parts, source="pvsr_scene_glb", faces=faces,
                         seconds=time.time() - started)
    if use_cache:
        _evict(case_id)
        _CACHE[case_id] = assets
        _CACHE_ORDER.append(case_id)
    return assets


def cached_cases() -> list[str]:
    """当前驻留的 case（诊断用；正常情况下 ≤1）。"""
    return list(_CACHE_ORDER)
