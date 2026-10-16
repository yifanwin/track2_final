"""Linux rootless user/mount/network/PID namespace with minimal chroot.
Only a stdlib Python runtime and the selected Skill file enter the sandbox.
No host home, judge root, procfs, env_ready or assets are mounted.
"""
import ctypes, os, pathlib, resource, subprocess, sys

def command(*args):
    subprocess.run(args, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def main():
    root = pathlib.Path(sys.argv[1]).resolve()
    skill = pathlib.Path(sys.argv[2]).resolve()
    package = pathlib.Path(sys.argv[3]).resolve() if len(sys.argv)>3 else None
    entry = sys.argv[4] if package is not None else None
    command('mount','--make-rprivate','/')
    command('mount','-t','tmpfs','-o','size=32m,nosuid,nodev','tmpfs',str(root))
    runtime=pathlib.Path(os.environ['TRACK2_SANDBOX_RUNTIME']).resolve(strict=True)
    for name in ('usr','lib','lib64','bin'):
        source=runtime/name
        if not source.exists():continue
        dest=root/name;dest.mkdir(exist_ok=True)
        # Private copied tree: never recursive-bind host directories or mounts.
        command('mount','--bind',str(source),str(dest))
        command('mount','-o','remount,bind,ro,nosuid,nodev',str(dest))
    (root/'skill').mkdir()
    if package is None:
        (root/'skill/main.py').touch()
        command('mount','--bind',str(skill),str(root/'skill/main.py'))
        command('mount','-o','remount,bind,ro,nosuid,nodev',str(root/'skill/main.py'))
        launch=['python3','-I','-B','/skill/main.py']
    else:
        command('mount','--bind',str(package),str(root/'skill'))
        command('mount','-o','remount,bind,ro,nosuid,nodev',str(root/'skill'))
        (root/'host').mkdir();(root/'host/main.py').touch()
        command('mount','--bind',str(skill),str(root/'host/main.py'))
        command('mount','-o','remount,bind,ro,nosuid,nodev',str(root/'host/main.py'))
        launch=['python3','-I','-B','/host/main.py','--entrypoint','/skill/'+entry]
    (root/'tmp').mkdir(); (root/'etc').mkdir(); (root/'dev').mkdir()
    (root/'dev/null').touch()
    command('mount','--bind','/dev/null',str(root/'dev/null'))
    os.chroot(root); os.chdir('/tmp')
    libc=ctypes.CDLL(None,use_errno=True)
    # Drop all bounding capabilities and all effective/permitted/inheritable caps.
    for cap in range(41): libc.prctl(24,cap,0,0,0)
    class Header(ctypes.Structure): _fields_=[('version',ctypes.c_uint32),('pid',ctypes.c_int)]
    class Data(ctypes.Structure): _fields_=[('effective',ctypes.c_uint32),('permitted',ctypes.c_uint32),('inheritable',ctypes.c_uint32)]
    header=Header(0x20080522,0); data=(Data*2)()
    if libc.capset(ctypes.byref(header),ctypes.byref(data)) != 0: raise RuntimeError('capset failed')
    if libc.prctl(38,1,0,0,0) != 0: raise RuntimeError('no_new_privs failed')
    # No descendants, namespace changes, kernel mounts, host networking or ptrace.
    sec=ctypes.CDLL('libseccomp.so.2',use_errno=True)
    sec.seccomp_init.restype=ctypes.c_void_p
    sec.seccomp_rule_add.argtypes=[ctypes.c_void_p,ctypes.c_uint32,ctypes.c_int,ctypes.c_uint]
    sec.seccomp_load.argtypes=[ctypes.c_void_p]
    sec.seccomp_syscall_resolve_name.argtypes=[ctypes.c_char_p]
    ctx=sec.seccomp_init(0x7fff0000)
    if not ctx: raise RuntimeError('seccomp unavailable')
    for name in ('fork','vfork','clone','clone3','unshare','setns','mount','umount2','pivot_root','chroot',
                 'ptrace','process_vm_readv','process_vm_writev','open_by_handle_at','socket','connect',
                 'bind','listen','accept','accept4','bpf','perf_event_open','keyctl','add_key','request_key'):
        nr=sec.seccomp_syscall_resolve_name(name.encode())
        if nr>=0 and sec.seccomp_rule_add(ctx,0x50001,nr,0)!=0: raise RuntimeError('seccomp rule failure')
    if sec.seccomp_load(ctx)!=0: raise RuntimeError('seccomp load failure')
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    resource.setrlimit(resource.RLIMIT_AS,(256*1024**2,256*1024**2))
    resource.setrlimit(resource.RLIMIT_CPU,(30,30))
    resource.setrlimit(resource.RLIMIT_NOFILE,(32,32))
    resource.setrlimit(resource.RLIMIT_FSIZE,(1024**2,1024**2))
    os.execve('/usr/bin/python3',launch,{'PATH':'/usr/bin:/bin','HOME':'/tmp','TMPDIR':'/tmp','PYTHONIOENCODING':'utf-8'})

if __name__=='__main__': main()
