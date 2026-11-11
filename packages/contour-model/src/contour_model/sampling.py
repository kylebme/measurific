"""Geometry-only adaptive resolution for complex silhouettes."""
import numpy as np
from .data import prepare,ring_geometry

VERSION='adaptive-turn-v1'
LOCAL_VERSION='adaptive-local-v2'

def adaptive_counts(contours,density=1.):
    xy,offsets,hole=ring_geometry(contours);counts=[]
    for is_hole,lo,hi in zip(hole,offsets[:-1],offsets[1:]):
        q=xy[lo:hi];d=np.roll(q,-1,axis=0)-q;length=np.linalg.norm(d,axis=1);s=np.r_[0,np.cumsum(length)]
        # Uniform geometry probe avoids nominal primitive seams and raw density leakage.
        probe=min(8192,max(512,len(q)))
        pos=np.arange(probe)*s[-1]/probe;closed=np.vstack([q,q[0]])
        p=np.c_[np.interp(pos,s,closed[:,0]),np.interp(pos,s,closed[:,1])]
        a=p-np.roll(p,2,axis=0);b=np.roll(p,-2,axis=0)-p
        cross=a[:,0]*b[:,1]-a[:,1]*b[:,0];dot=(a*b).sum(1)
        turn=np.abs(np.arctan2(cross,dot)).sum()/2
        # Smooth circles need few tokens; many shoulders/teeth need proportionally more.
        base=max(64 if is_hole else 768,turn*8)
        counts.append(int(np.ceil(base*density/32))*32)
    counts=np.array(counts,dtype=np.int64)
    if counts.sum()>65536:raise ValueError('Contour exceeds the 65,536-interval operating envelope')
    return counts

def adaptive_grids(contours,counts):
    xy,offsets,hole=ring_geometry(contours);grids=[]
    for is_hole,lo,hi,n in zip(hole,offsets[:-1],offsets[1:],counts):
        q=xy[lo:hi]
        area=np.sum(q[:,0]*np.roll(q[:,1],-1)-q[:,1]*np.roll(q[:,0],-1))
        if (area<0)!=is_hole:q=q[::-1]
        lengths=np.linalg.norm(np.roll(q,-1,axis=0)-q,axis=1);arc=np.r_[0,np.cumsum(lengths)]
        size=min(65536,max(512,len(q)*4));grid=np.linspace(0,arc[-1],size+1);closed=np.vstack([q,q[0]])
        positions=(grid[:-1]+grid[1:])/2
        p=np.c_[np.interp(positions,arc,closed[:,0]),np.interp(positions,arc,closed[:,1])]
        a=p-np.roll(p,2,axis=0);b=np.roll(p,-2,axis=0)-p
        curvature=np.abs(np.arctan2(a[:,0]*b[:,1]-a[:,1]*b[:,0],(a*b).sum(1)))
        # Periodic smoothing spreads corner mass onto both adjacent surfaces.
        from scipy.ndimage import gaussian_filter1d
        curvature=gaussian_filter1d(curvature,1.5,mode='wrap')
        mass=.35/size+.65*curvature/max(curvature.sum(),1e-12)
        cdf=np.r_[0,np.cumsum(mass)];cdf/=cdf[-1]
        target=np.interp(np.linspace(0,1,int(n)+1),cdf,grid);target[0]=0.;target[-1]=arc[-1]
        grids.append(target)
    return grids


def prepare_adaptive(contours,density=1.,version=VERSION):
    counts=adaptive_counts(contours,density)
    if version not in (VERSION,LOCAL_VERSION):raise ValueError(f'Unsupported sampling version: {version}')
    grids=adaptive_grids(contours,counts) if version==LOCAL_VERSION else None
    return prepare(contours,ring_counts=counts,ring_grids=grids)
