"""Synthetic adversary; only attacks a specifically supplied synthetic canary."""
import ctypes, json, os, socket, sys
message=json.loads(sys.stdin.readline())
obs=message['observation'];path=obs['canary_path']; report={}
for name,p,mode in [('private_read',path,'r'),('private_write',path,'w'),('proc_escape','/proc/1/root'+path,'r'),('skill_modify','/skill/main.py','w')]:
    try:
        with open(p,mode) as f:
            if mode=='w':f.write('tamper')
            else:f.read()
        report[name]='FAIL'
    except OSError:report[name]='PASS'
try: socket.socket();report['network']='FAIL'
except OSError: report['network']='PASS'
try: os.fork();report['fork']='FAIL'
except OSError:report['fork']='PASS'
libc=ctypes.CDLL(None,use_errno=True)
report['chroot_capability']='PASS' if libc.chroot(b'/')==-1 else 'FAIL'
report['environment']='PASS' if 'CANARY_SECRET' not in os.environ else 'FAIL'
for name,p in obs.get('canary_paths',{}).items():
    mode='w' if name.endswith('tamper') else 'r'
    try:
        with open(p,mode) as f:
            if mode=='w':f.write('tamper')
            else:f.read()
        report[name]='FAIL'
    except OSError:report[name]='PASS'
report['mount_escape']='PASS' if libc.mount(b'none',b'/tmp',b'tmpfs',0,None)==-1 else 'FAIL'
report['namespace_change']='PASS' if libc.unshare(0x10000000)==-1 else 'FAIL'
report['ptrace']='PASS' if libc.ptrace(0,0,None,None)==-1 else 'FAIL'
open_fds=[]
for fd in range(3,32):
    try:os.fstat(fd);open_fds.append(fd)
    except OSError:pass
report['inherited_private_fd']='PASS' if not open_fds else 'FAIL'
print(json.dumps({'method':'security_report','payload':report}),flush=True)
sys.stdin.readline()
print(json.dumps({'done':True}),flush=True)
