import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]   # track2_judge 包根
def array(x):
    return np.asarray(x.detach().cpu() if hasattr(x, 'detach') else x).reshape(-1)
def plain(x):
    if isinstance(x, dict): return {str(k): plain(v) for k,v in x.items()}
    if isinstance(x, (list, tuple)): return [plain(v) for v in x]
    if isinstance(x, np.ndarray): return plain(x.tolist())
    if isinstance(x, np.generic): return x.item()
    return x
def save(path, data):
    p=ROOT/path; p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(plain(data), indent=2, ensure_ascii=False, allow_nan=False)+'\n',encoding='utf-8')
