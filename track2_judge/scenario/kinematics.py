"""Cached Piper URDF kinematics, physical finger transforms and bounded IK."""
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares

def transform(p,r):
    t=np.eye(4);t[:3,:3]=r;t[:3,3]=p;return t
class Kinematics:
    def __init__(self,path):
        self.names=['joint'+str(i) for i in range(1,7)]
        self.joints=[];limits={}
        for j in ET.parse(path).getroot().findall('joint'):
            o=j.find('origin');xyz=np.zeros(3) if o is None else np.fromstring(o.get('xyz','0 0 0'),sep=' ');rpy=np.zeros(3) if o is None else np.fromstring(o.get('rpy','0 0 0'),sep=' ')
            a=j.find('axis');axis=np.array([1.,0,0]) if a is None else np.fromstring(a.get('xyz'),sep=' ')
            self.joints.append((j.get('name'),j.get('type'),j.find('parent').get('link'),j.find('child').get('link'),transform(xyz,Rotation.from_euler('xyz',rpy).as_matrix()),axis))
            lim=j.find('limit')
            if lim is not None:limits[j.get('name')]=[float(lim.get('lower')),float(lim.get('upper'))]
        self.limits=np.array([limits[n] for n in self.names])
    def poses(self,q):
        values=dict(zip(self.names+['gripper_joint1','gripper_joint2'],q));poses={'world':np.eye(4)}
        for name,kind,parent,child,origin,axis in self.joints:
            t=origin
            if kind=='revolute':t=t@transform([0,0,0],Rotation.from_rotvec(axis*values.get(name,0)).as_matrix())
            if kind=='prismatic':t=t@transform(axis*values.get(name,0),np.eye(3))
            poses[child]=poses[parent]@t
        return poses
    def fk(self,q):return self.poses(q)['diagnostic_tcp']
    def ik(self,p,r,start,seed_count=12):
        def residual(q):
            t=self.fk(q);return np.r_[t[:3,3]-p,.12*Rotation.from_matrix(r@t[:3,:3].T).as_rotvec()]
        rng=np.random.default_rng(812);seeds=[start,[0,1.3,-1.3,0,.5,0],[0,.6,-1.7,0,1.,0],[0,2.,-.8,0,-.5,0]]
        seeds+=list(rng.uniform(self.limits[:,0],self.limits[:,1],size=(max(0,seed_count-4),6)))
        best=None
        for seed in seeds[:seed_count]:
            sol=least_squares(residual,np.clip(seed,self.limits[:,0]+1e-6,self.limits[:,1]-1e-6),bounds=(self.limits[:,0]+1e-7,self.limits[:,1]-1e-7),max_nfev=140,ftol=1e-9,xtol=1e-9,gtol=1e-9)
            err=np.linalg.norm(residual(sol.x))
            if best is None or err<best[0]:best=(err,sol.x)
            if err<1e-7:break
        t=self.fk(best[1]);return best[1],{'weighted_residual':float(best[0]),'position_error_m':float(np.linalg.norm(t[:3,3]-p)),'rotation_error_rad':float(np.linalg.norm(Rotation.from_matrix(r@t[:3,:3].T).as_rotvec()))}
