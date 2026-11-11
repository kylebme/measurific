"""Label-blind smooth form-error augmentation; nominal targets stay unchanged."""
import numpy as np
from .data import encode_geometry


def deform(sample, rng, strength=.003):
    """A shared smooth spatial field, independent of labels, rings and boundaries.

    Coordinates are normalized by part radius. Each wave has displacement at
    most strength, wavelength .04--.8 radii, and Lipschitz bound .075. Two waves
    therefore cannot fold space or create new intersections. Zero/weak/strong
    examples are mixed by the caller; no nominal coordinates enter the encoder.
    """
    out={k:v.copy() for k,v in sample.items()}
    x=out['x']; q=x[:,:2].astype(float); displacement=np.zeros_like(q)
    for _ in range(2):
        theta=rng.uniform(-np.pi,np.pi); axis=np.array([np.cos(theta),np.sin(theta)])
        theta=rng.uniform(-np.pi,np.pi); direction=np.array([np.cos(theta),np.sin(theta)])
        wavelength=np.exp(rng.uniform(np.log(.04),np.log(.8)))
        amplitude=min(strength*rng.uniform(.25,1),.075*wavelength/(2*np.pi))
        displacement+=amplitude*np.sin(2*np.pi*(q@axis)/wavelength+rng.uniform(-np.pi,np.pi))[:,None]*direction
    offsets=out['offsets']; hole=np.array([bool(x[lo,5]) for lo in offsets[:-1]])
    lengths=x[:,4].copy()
    for lo,hi in zip(offsets[:-1],offsets[1:]):
        old=np.roll(q[lo:hi],-1,axis=0)-np.roll(q[lo:hi],1,axis=0)
        changed=(q+displacement)[lo:hi]
        new=np.roll(changed,-1,axis=0)-np.roll(changed,1,axis=0)
        lengths[lo:hi]*=np.linalg.norm(new,axis=1)/np.maximum(np.linalg.norm(old,axis=1),1e-12)
    out['x']=encode_geometry(q+displacement,offsets,hole,lengths)[0]
    return out
