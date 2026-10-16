"""Offline installation checks; never install packages or use network.

D3：冻结评分器哈希与环境锁检查已删除（不再在运行时重算）。
"""
import importlib.util, json, shutil, sys
from pathlib import Path

def main():
    results={k:bool(importlib.util.find_spec(k)) for k in ('numpy','scipy','PIL','mani_skill','sapien','torch')}
    results['linux']=sys.platform=='linux';results['unshare']=bool(shutil.which('unshare'));results['mount']=bool(shutil.which('mount'))
    print(json.dumps({'checks':results,'local_synthetic_ready':results['numpy'] and results['scipy'],
                      'physical_dependencies_ready':all(results.values()),'production_ready':False}))
    return 0 if results['numpy'] and results['scipy'] and results['PIL'] else 1

if __name__=='__main__':sys.exit(main())
