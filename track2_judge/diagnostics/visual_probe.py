"""Bounded actual CPU-render attempt; records native failure honestly."""
import argparse,json,os,subprocess,sys
from pathlib import Path
from track2_judge.sandbox.rpc import encode

def child(output):
    from track2_judge.scenario.prototype_env_v2 import make_env
    from track2_judge.observation.provider import ObservationProvider
    env=None
    try:
        env=make_env(render=True);env.reset(seed=0)
        provider=ObservationProvider(camera=lambda e:e.render_rgb_array(),camera_required=True)
        provider.capture_initial(env);frame=provider.observation()['first_frame']
        import base64,numpy as np
        from PIL import Image
        from io import BytesIO
        rgb=np.asarray(Image.open(BytesIO(base64.b64decode(frame['data_base64']))))
        assert rgb.std()>1
        result={'status':'PASS','actual_render_backend':'cpu','first_frame_shape':list(rgb.shape),'nonblank_std':float(rgb.std()),'RGB_only':True,'GT_fields_returned':False}
    except Exception as exc:result={'status':'FAIL','actual_render_backend':'cpu','error':type(exc).__name__,'message':'CPU first-frame render unavailable','synthetic_interface_test':'separate'}
    finally:
        if env is not None:env.close()
    (output/'visual_result.json').write_text(encode(result),encoding='utf8')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--child',action='store_true');a=p.parse_args();root=Path(a.output).resolve();root.mkdir(parents=True,exist_ok=True)
    if a.child:child(root)
    else:
        import resource
        resource.setrlimit(resource.RLIMIT_CORE,(0,0))
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
        try:
            with (root/'internal_render_console.log').open('w') as log:
                code=subprocess.run([sys.executable,'-B',__file__,'--child','--output',str(root)],env=env,stdout=log,stderr=log,timeout=180).returncode
        except subprocess.TimeoutExpired:code='TIMEOUT'
        path=root/'visual_result.json'
        if not path.exists():path.write_text(encode({'status':'FAIL','actual_render_backend':'cpu','child_exit':code,'message':'real rendering attempt failed; no synthetic image represented as camera output'}),encoding='utf8')
        print(path.read_text(encoding='utf8'))
