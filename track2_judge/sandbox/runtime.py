"""Judge-side stdlib-only runtime copy. No host runtime directory mounts.

Linux namespaces remain required. This is a candidate runtime, not an approved
PhyAgentOS SDK. Build in a private judge directory before accepting submissions.
"""
import hashlib, json, os, pathlib, shutil, subprocess, sys

def _prepare_runtime_unlocked():
    if sys.platform != 'linux': raise RuntimeError('Linux isolation required')
    base = pathlib.Path(__file__).resolve().parent / 'cache' / 'sandbox_runtime'
    manifest = base / 'runtime_manifest.json'
    def verify():
        entries = json.loads(manifest.read_text())
        actual = {str(p.relative_to(base)) for p in base.rglob('*') if p.is_file() and p != manifest}
        if actual != set(entries): raise RuntimeError('runtime inventory mismatch')
        for rel, digest in entries.items():
            p = base / rel
            if p.is_symlink() or hashlib.sha256(p.read_bytes()).hexdigest() != digest:
                raise RuntimeError('runtime integrity rejected')
    if manifest.exists():
        verify(); return base
    base.mkdir(parents=True, mode=0o700, exist_ok=False)
    os.chmod(base.parent, 0o700)
    version = subprocess.check_output(['/usr/bin/python3','-I','-c','import sys;print("%d.%d"%sys.version_info[:2])'],text=True).strip()
    stdlib = pathlib.Path('/usr/lib/python' + version)
    if not stdlib.is_dir(): raise RuntimeError('unsupported system Python layout')
    def copy(src, rel=None):
        src=pathlib.Path(src)
        target=base/(rel if rel is not None else str(src).lstrip('/'))
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(src.resolve(strict=True),target)
        target.chmod(0o555 if os.access(src,os.X_OK) else 0o444)
    copy('/usr/bin/python3','usr/bin/python3')
    for src in stdlib.rglob('*'):
        if src.is_file() and not set(src.relative_to(stdlib).parts)&{'__pycache__','test','tests','ensurepip','dist-packages','site-packages'}:
            if src.suffix in ('.py','.so','.txt','.pickle'): copy(src)
    libs=set()
    binaries=[base/'usr/bin/python3',*base.rglob('*.so')]
    lookup=subprocess.check_output(['/sbin/ldconfig','-p'],text=True)
    for line in lookup.splitlines():
        if 'libseccomp.so.2 ' in line and '=>' in line:libs.add(line.split('=>',1)[1].strip());break
    else:raise RuntimeError('libseccomp missing')
    for binary in binaries:
        output=subprocess.run(['/usr/bin/ldd',str(binary)],capture_output=True,text=True,check=True).stdout
        for line in output.splitlines():
            for word in line.split():
                if word.startswith('/') and pathlib.Path(word).is_file():libs.add(word)
    for lib in sorted(libs):copy(lib)
    entries={str(p.relative_to(base)):hashlib.sha256(p.read_bytes()).hexdigest() for p in base.rglob('*') if p.is_file()}
    manifest.write_text(json.dumps(entries,sort_keys=True));manifest.chmod(0o400)
    verify();return base

def prepare_runtime():
    # Serialize a shared candidate cache's initialization AND verification.
    # Namespace/seccomp setup, copied inventory and hash checks are unchanged.
    if sys.platform!='linux':raise RuntimeError('Linux isolation required')
    import fcntl
    cache=pathlib.Path(__file__).resolve().parent/'cache'
    cache.mkdir(mode=0o700,parents=True,exist_ok=True)
    if cache.is_symlink() or cache.stat().st_uid!=os.getuid() or cache.stat().st_mode&0o077:raise PermissionError('private cache ownership/mode')
    fd=os.open(cache/'sandbox_runtime.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW|os.O_CLOEXEC,0o600)
    try:
        fcntl.flock(fd,fcntl.LOCK_EX)
        return _prepare_runtime_unlocked()
    finally:os.close(fd)

if __name__=='__main__':
    root=prepare_runtime()
    print(json.dumps({'status':'PASS','runtime_files':len(json.loads((root/'runtime_manifest.json').read_text())),'scope':'private copied stdlib runtime; no host recursive mounts'}))
