"""Explicit angular tooth spans, including exact involute flanks."""
import numpy as np
from .geometry import Chain


def angular_teeth(part, radius, count, depth, hole=False, style='involute'):
    pitch=2*np.pi/count
    depth=min(depth,radius*pitch*.35)
    fid=part.feature('tooth_profile',count=count,domain='angular',style=style,
                     angular_pitch=pitch,standard_match=None)
    c=Chain(part,hole);sign=-1 if hole else 1
    def point(r,a):return r*np.array([np.cos(sign*a),np.sin(sign*a)])
    if style=='smooth':
        for i in range(count):
            t=np.linspace(i*pitch,(i+1)*pitch,max(17,int(np.ceil(pitch*(radius+depth)/c.spacing))+1))
            r=radius+depth*(1-np.cos(t*count))/2
            c.add(np.c_[r*np.cos(sign*t),r*np.sin(sign*t)],fid,'freeform',profile='sinusoidal_tooth')
    else:
        rb=radius*np.cos(np.deg2rad(20))
        inv=lambda r:np.sqrt((r/rb)**2-1)-np.arctan(np.sqrt((r/rb)**2-1))
        delta=float(inv(radius+depth)-inv(radius)) if style=='involute' else .18*pitch
        for i in range(count):
            a=i*pitch
            c.arc([0,0],radius,sign*(a-.17*pitch),sign*(a+.17*pitch),fid)
            if style=='involute':
                r=np.linspace(radius,radius+depth,max(12,int(np.ceil(depth/c.spacing))+1))
                angle=a+.17*pitch+inv(r)-inv(radius)
                c.add(np.c_[r*np.cos(sign*angle),r*np.sin(sign*angle)],fid,'freeform',
                      profile='involute',base_radius=float(rb))
            else:c.line(point(radius,a+.17*pitch),point(radius+depth,a+.17*pitch+delta),fid)
            end=a+.83*pitch-delta
            c.arc([0,0],radius+depth,sign*(a+.17*pitch+delta),sign*end,fid)
            if style=='involute':
                r=r[::-1];angle=a+.83*pitch-(inv(r)-inv(radius))
                c.add(np.c_[r*np.cos(sign*angle),r*np.sin(sign*angle)],fid,'freeform',
                      profile='involute',base_radius=float(rb))
            else:c.line(point(radius+depth,end),point(radius,a+.83*pitch),fid)
    c.close()
    return fid
