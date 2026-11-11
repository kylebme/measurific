import numpy as np
import pytest
from contour_model.pae_corpus import transfer,project


def source():
    xy=np.array([[0,0],[20,0],[20,20],[0,20]],float)
    ids=np.arange(1,5)
    return dict(actual_xy_mm=xy,ring_offsets=np.array([0,4]),edge_base_class=np.ones(4,int),
        edge_base_instance=ids,edge_compound_class=np.full(4,4),edge_compound_instance=np.ones(4,int),
        edge_feature_id=np.ones(4,int),edge_anomaly_class=np.zeros(4,int))


def test_transfer_retains_same_class_boundaries_winding_and_seam():
    raw=source();ref=raw['actual_xy_mm']
    dense=np.concatenate([a+np.arange(20)[:,None]/20*(b-a) for a,b in zip(ref,np.roll(ref,-1,axis=0))])
    observed=np.roll(dense[::-1],13,axis=0)
    out,_=transfer(raw,ref,observed,np.array([0,80]))
    assert set(out['edge_base_instance'])=={1,2,3,4}
    assert (out['edge_compound_instance']==1).all()
    assert (out['edge_base_class']==1).all()
    assert len(np.flatnonzero(out['edge_base_instance']!=np.roll(out['edge_base_instance'],1)))==4


def test_long_edge_projection_not_nearest_vertex_guess():
    ref=np.array([[0,0],[100,0],[100,1],[0,1]],float)
    idx,_,distance,_=project(np.array([[50,.1],[50,.9]]),ref)
    np.testing.assert_array_equal(idx,[0,2]);np.testing.assert_allclose(distance,.1)


def test_transfer_rejects_lost_identity():
    raw=source();raw['edge_base_instance']=np.array([1,2,3,4])
    # Observed triangle skips one entire side; cannot be treated as intact truth.
    with pytest.raises(ValueError):transfer(raw,raw['actual_xy_mm'],raw['actual_xy_mm'][:3],np.array([0,3]))
