"""Torch-free mapping of token outputs back onto observed contour edges."""
import numpy as np


def runs(mask):
    """Closed-ring connected components, including the seam."""
    n=len(mask)
    if not np.any(mask):return []
    if np.all(mask):return [np.arange(n)]
    result=[]
    for start in np.flatnonzero(mask & ~np.roll(mask,1)):
        r=[];i=int(start)
        while mask[i]:
            r.append(i);i=(i+1)%n
        result.append(np.array(r))
    return result


def describe_selection(indices,contours):
    mask=np.zeros(len(contours['actual_xy_mm']),bool);mask[indices]=True
    spans=[]
    for ri,(lo,hi) in enumerate(zip(contours['ring_offsets'][:-1],contours['ring_offsets'][1:])):
        for run in runs(mask[lo:hi]):
            spans.append({'ring_index':ri,'start_edge':int(run[0]),'edge_count':len(run),
                          'wraps':bool(run[-1]<run[0]),'closed_ring':bool(len(run)==hi-lo)})
    return spans


def mapped_probabilities(prepared,values):
    area=prepared.overlap.multiply(prepared.weights[:,None])
    weight=np.asarray(area.sum(0)).ravel()
    return np.asarray(area.T@values)/np.maximum(weight[:,None],1e-12)


def mapped_boundary_events(prepared,values):
    """A boundary is a point event, not a class occupying an interval's length."""
    overlap=prepared.overlap.tocoo()
    result=np.zeros(prepared.overlap.shape[1],dtype=values.dtype)
    np.maximum.at(result,overlap.col,values[overlap.row])
    return result


def split_surface_run(indices, peaks, ring_size):
    """Split at observed corners without inventing a boundary at the array seam."""
    if len(indices)==ring_size:
        cuts=sorted(set(int(i) for i in peaks))
        if not cuts:return [indices]
        return [indices[np.arange(a,b if b>a else b+len(indices))%len(indices)]
                for a,b in zip(cuts,cuts[1:]+cuts[:1])]
    cuts=sorted(set([0,len(indices)]+[int(i) for i in peaks if 2<=i<len(indices)-2]))
    return [indices[a:b] for a,b in zip(cuts[:-1],cuts[1:])]


def observed_turn(q,arc,indices,window):
    closed=np.vstack([q,q[0]])
    def at(position):
        position=position%arc[-1]
        return np.c_[np.interp(position,arc,closed[:,0]),np.interp(position,arc,closed[:,1])]
    before=q[indices]-at(arc[indices]-window);after=at(arc[indices]+window)-q[indices]
    cosine=np.sum(before*after,axis=1)/np.maximum(np.linalg.norm(before,axis=1)*np.linalg.norm(after,axis=1),1e-12)
    return np.arccos(np.clip(cosine,-1,1))


def surface_run_turn(q,arc,run,window):
    # A short land's corner test must not reach into the neighboring deep notches.
    if len(run)<len(q):window=min(window,.2*np.diff(arc)[run].sum())
    return observed_turn(q,arc,run,window)
