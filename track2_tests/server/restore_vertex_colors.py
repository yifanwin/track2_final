"""Judge-side temporary rendering conversion; preserve original geometry and colors.
No original asset is modified. Per-triangle linear color is baked into a padded
texture, avoiding the non-faithful palette-UV interpolation shortcut.
"""
import os,sys,json,hashlib
from pathlib import Path
import numpy as np,trimesh
from PIL import Image
WORK=Path(sys.argv[1]).resolve(strict=True)
ROOT=Path(os.environ['TRACK2_STAGING_ROOT']).resolve(strict=True)
assert WORK.is_relative_to(ROOT) and WORK.stat().st_uid==os.getuid()
os.umask(0o077);OUT=WORK/'private_visual_cache';OUT.mkdir(mode=0o700,exist_ok=False)
rows=[]
for folder in (ROOT/'assets').iterdir():
    source=folder/'visual.glb'
    if not source.is_file():continue
    scene=trimesh.load(source,force='scene',process=False)
    assert len(scene.geometry)==1
    mesh=next(iter(scene.geometry.values()));color=mesh.visual.vertex_colors
    assert color.shape==(len(mesh.vertices),4) and np.all(color[:,3]==255)
    ids=mesh.faces.reshape(-1);vertices=np.asarray(mesh.vertices)[ids];normals=np.asarray(mesh.vertex_normals)[ids]
    triangles=np.arange(len(ids)).reshape(-1,3);c=color[ids,:3].astype(np.float32).reshape(-1,3,3)/255
    n=len(triangles);side=int(np.ceil(np.sqrt(n)));tile=6;width=side*tile
    texture=np.zeros((width,width,4),dtype=np.uint8);texture[:,:,3]=255
    cols=(np.arange(n)%side)*tile;rows_y=(np.arange(n)//side)*tile
    max_error=0
    for dy in range(tile):
        for dx in range(tile):
            u=(dx-1)/2;v=(dy-1)/2
            linear=np.clip(c[:,0]+u*(c[:,1]-c[:,0])+v*(c[:,2]-c[:,0]),0,1)
            srgb=np.where(linear<=.0031308,12.92*linear,1.055*np.power(linear,1/2.4)-.055)
            encoded=np.clip(np.round(srgb*255),0,255).astype(np.uint8)
            texture[rows_y+dy,cols+dx,:3]=encoded
            if dx in (1,2,3) and dy in (1,2,3) and dx+dy<=4:
                f=encoded.astype(float)/255;decoded=np.where(f<=.04045,f/12.92,np.power((f+.055)/1.055,2.4))
                max_error=max(max_error,float(np.max(np.abs(decoded-linear))))
    uv=np.zeros((n,3,2));uv[:,0,:]=np.column_stack(((cols+1.5)/width,1-(rows_y+1.5)/width));uv[:,1,:]=np.column_stack(((cols+3.5)/width,1-(rows_y+1.5)/width));uv[:,2,:]=np.column_stack(((cols+1.5)/width,1-(rows_y+3.5)/width))
    material=trimesh.visual.material.PBRMaterial(baseColorTexture=Image.fromarray(texture),baseColorFactor=np.array([1.,1.,1.,1.]),metallicFactor=1.,roughnessFactor=1.)
    visual=trimesh.visual.texture.TextureVisuals(uv=uv.reshape(-1,2),material=material)
    converted=trimesh.Trimesh(vertices=vertices,faces=triangles,vertex_normals=normals,visual=visual,process=False)
    payload=converted.export(file_type='glb');dest=OUT/(folder.name+'.glb');dest.write_bytes(payload)
    reloaded=trimesh.load(dest,force='scene',process=False);other=next(iter(reloaded.geometry.values()))
    actual=np.asarray(other.vertices)[other.faces.reshape(-1)]
    expected=vertices[triangles.reshape(-1)];geometry_error=float(np.max(np.abs(actual-expected)))
    world_bounds_error=float(np.max(np.abs(scene.bounds-reloaded.bounds)))
    assert geometry_error<1e-7 and world_bounds_error<1e-7 and max_error<.005
    row={'object':folder.name,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'derived_sha256':hashlib.sha256(payload).hexdigest(),'geometry_max_error_m':geometry_error,'world_bounds_max_error_m':world_bounds_error,'linear_color_quantization_max_error':max_error,'triangles':n,'texture_size':[width,width],'original_modified':False,'collision_modified':False}
    rows.append(row);print(json.dumps(row),flush=True)
(WORK/'evidence/vertex_color_conversion_sanitized.json').write_text(json.dumps({'status':'PASS','scope':'temporary original COLOR_0 restoration; not a new official asset','objects':rows},indent=2))
