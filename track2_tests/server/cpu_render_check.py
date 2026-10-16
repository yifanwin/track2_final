"""Pin Vulkan to software ICD only; never enumerate or allocate physical GPUs."""
import json,os,pathlib,subprocess,sys
work=pathlib.Path(sys.argv[1]).resolve(strict=True)
root=pathlib.Path(os.environ['TRACK2_STAGING_ROOT']).resolve(strict=True)
assert work.is_relative_to(root) and work!=root and work.stat().st_uid==os.getuid()
os.umask(0o077)
out=work/'cpu_render';out.mkdir(mode=0o700,exist_ok=False)
icd=pathlib.Path('/usr/share/vulkan/icd.d/lvp_icd.x86_64.json')
config=json.loads(icd.read_text());lib=pathlib.Path(config['ICD']['library_path']);assert lib.is_file() and 'lvp' in lib.name
env=os.environ.copy();env.update(VK_ICD_FILENAMES=str(icd),VK_DRIVER_FILES=str(icd),CUDA_VISIBLE_DEVICES='',
    XDG_RUNTIME_DIR=str(out),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
os.nice(10)
p=subprocess.run(['vulkaninfo','--summary'],env=env,capture_output=True,text=True,timeout=25)
(out/'software_vulkan.log').write_text(p.stdout+'\n'+p.stderr)
software_ok=p.returncode==0 and 'PHYSICAL_DEVICE_TYPE_CPU' in p.stdout and 'llvmpipe' in p.stdout.lower()
result={'software_vulkan':'PASS' if software_ok else 'FAIL','physical_gpu_used':False,'icd':str(icd),'camera':'NOT TESTED'}
if software_ok:
    source=work/'source/Track2_ManiSkill_Evaluator_Candidate_V2'
    with (out/'primitive_camera.log').open('w') as log:
        try:
            p=subprocess.run([str(root/'env_ready/bin/python'),'-B','visual_probe_v2.py','--output',str(out/'primitive')],cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120)
            result['probe_exit_code']=p.returncode
        except subprocess.TimeoutExpired:result['probe_exit_code']='TIMEOUT'
    path=out/'primitive/visual_result.json'
    result['camera']=json.loads(path.read_text()) if path.exists() else {'status':'FAIL','reason':'bounded child produced no result'}
(out/'cpu_render_result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
