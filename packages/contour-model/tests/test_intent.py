import numpy as np
import torch
from contour_model.data import encode_geometry
from contour_model.intent import deform
from contour_model.model import ContourTransformer, ModelConfig
from contour_model.recache_intent import corrected_annotations
from contour_model.train_primitives import initialize_from_previous_schema


def test_deformation_keeps_nominal_targets_and_is_label_blind():
    q=np.c_[np.cos(np.linspace(0,2*np.pi,256,endpoint=False)),np.sin(np.linspace(0,2*np.pi,256,endpoint=False))]
    x=encode_geometry(q,[0,256],[False],np.ones(256))[0]
    a=dict(x=x,offsets=np.array([0,256]),primary=np.ones((256,4)),boundary=np.zeros(256))
    b=deform(a,np.random.default_rng(9)); poisoned=dict(a,primary=np.zeros((256,4)),boundary=np.ones(256))
    c=deform(poisoned,np.random.default_rng(9))
    np.testing.assert_array_equal(b['x'],c['x'])
    np.testing.assert_array_equal(a['primary'],b['primary'])
    np.testing.assert_array_equal(a['boundary'],b['boundary'])
    assert not np.array_equal(a['x'],b['x']) and np.isfinite(b['x']).all()


def test_global_context_backward_and_old_checkpoint_compatibility():
    torch.set_num_threads(2)
    config=dict(width=16,global_width=32,local_depth=1,global_depth=1,pooled_tokens=8,queries=4,label_schema='primitives-compounds-v2')
    old=ContourTransformer(ModelConfig(**config)).eval()
    new=ContourTransformer(ModelConfig(**config,direct_global_context=True)).eval()
    initialize_from_previous_schema(new,old.state_dict())
    x=torch.randn(128,12); offsets=[0,64,128]
    a=old(x,offsets); b=new(x,offsets)
    torch.testing.assert_close(a['primary'],b['primary'])
    b['primary'].square().mean().backward()
    assert new.context_gate.grad is not None and torch.isfinite(new.context_gate.grad)
    with torch.no_grad(): new.context_gate.fill_(.2)
    new.zero_grad();new(x,offsets)['primary'].square().mean().backward()
    assert new.context_attention.q_proj_weight.grad.abs().sum()>0


def test_freeform_relabel_uses_nominal_not_waviness():
    q=np.array([[0,0],[2,0],[4,0],[6,0],[8,0],[10,0],[10,5],[0,5]],float)
    raw=dict(nominal_xy_mm=q,actual_xy_mm=q+np.random.default_rng(2).normal(0,.2,q.shape),
        ring_offsets=np.array([0,8]),edge_primitive_id=np.array([1,1,1,1,1,2,3,4]),
        edge_base_class=np.array([2,2,2,2,2,1,1,1]),edge_base_instance=np.array([1,1,1,1,1,2,3,4]),
        edge_compound_class=np.zeros(8,int),edge_compound_instance=np.zeros(8,int))
    metadata=dict(primitives=[dict(id=1,kind='freeform')],compound_instances=[])
    a,_=corrected_annotations(metadata,raw)
    assert (a['edge_base_class'][:5]==1).all()
    raw['actual_xy_mm']*=20
    b,_=corrected_annotations(metadata,raw)
    np.testing.assert_array_equal(a['edge_base_class'],b['edge_base_class'])


def test_closed_lobed_freeform_is_not_a_circle():
    t=np.linspace(0,2*np.pi,240,endpoint=False);r=10+np.cos(3*t)
    q=np.c_[r*np.cos(t),r*np.sin(t)]; n=len(q)
    raw=dict(nominal_xy_mm=q,ring_offsets=np.array([0,n]),edge_primitive_id=np.ones(n,int),
        edge_base_class=np.full(n,3),edge_base_instance=np.ones(n,int),
        edge_compound_class=np.zeros(n,int),edge_compound_instance=np.zeros(n,int))
    a,_=corrected_annotations(dict(primitives=[dict(id=1,kind='freeform')],compound_instances=[]),raw)
    assert 3 not in a['edge_base_class'] and len(np.unique(a['edge_base_instance']))>1
