import numpy as np
import torch
from inspection_shapes.geometry import Part, Chain, circle, slot
from inspection_shapes.primitive_labels import labels, LABEL_VERSION
from inspection_shapes.teeth import angular_teeth
from inspection_shapes.expanded import NEW_FAMILIES, expanded_nominal
from contour_model.corpus_primitives import targets
from contour_model.data import prepare
from contour_model.model import ModelConfig, ContourTransformer
from contour_model.train_primitives import losses


def record(p):
    return {'features':p.features,'primitives':p.primitives}, {
        'nominal_xy_mm':np.concatenate([r.nominal for r in p.rings]),
        'actual_xy_mm':np.concatenate([r.nominal for r in p.rings]),
        'ring_offsets':np.r_[0,np.cumsum([len(r.nominal) for r in p.rings])],
        'edge_primitive_id':np.concatenate([r.primitive for r in p.rings]),
        'edge_feature_id':np.concatenate([r.feature for r in p.rings]),
        'edge_anomaly_class':np.zeros(sum(len(r.nominal) for r in p.rings),np.uint8)}


def test_circle_slot_and_straight_surface_identity():
    p=Part('test');circle(p,[0,0],20,'external_profile')
    slot(p,[0,0],12,4)
    m,c=record(p);a,groups=labels(m,c)
    n=len(p.rings[0].nominal)
    assert np.all(a['edge_base_class'][:n]==3)
    assert set(a['edge_base_class'][n:])=={1,2}
    assert groups[0]['class']=='full_round_slot'
    assert len(groups[0]['base_instance_ids'])==4
    # The base instances survive compound membership.
    assert len(np.unique(a['edge_base_instance'][n:]))==4


def test_square_thread_maps_to_trapezoidal_and_templates_restricted():
    styles=set()
    for seed in range(12):
        p=expanded_nominal('multi_thread_shaft',np.random.default_rng(seed))
        m,c=record(p);a,groups=labels(m,c)
        for f in p.features:
            if f['semantic']=='thread':
                style=f['nominal_parameters']['style'];styles.add(style)
                assert np.all(a['edge_compound_class'][c['edge_feature_id']==f['id']]==(2 if style=='v' else 3))
    assert styles=={'v','trapezoid','square'}
    assert 'keyhole_panel' not in NEW_FAMILIES


def test_involute_flanks_are_assumed_arcs_and_crowns_are_arcs():
    p=Part('test');angular_teeth(p,15,24,1,style='involute')
    m,c=record(p);a,groups=labels(m,c)
    assert set(a['edge_base_class'])=={2}
    assert groups[0]['class']=='tooth_profile'
    assert sum(v['kind']=='freeform' for v in p.primitives)==48


def test_same_class_collinear_construction_is_one_instance():
    p=Part('test');c=Chain(p);f=p.feature('flat')
    for a,b in [([0,0],[3,0]),([3,0],[6,0]),([6,0],[6,4]),([6,4],[0,4]),([0,4],[0,0])]:c.line(a,b,f)
    c.close();m,raw=record(p);a,_=labels(m,raw)
    assert len(np.unique(a['edge_base_instance']))==4


def test_compounds_own_whole_observable_base_instances():
    from contour_model.corpus import mixed_shaft
    for seed in range(8):
        m,c=record(mixed_shaft(np.random.default_rng(seed)))
        a,groups=labels(m,c)
        for ident in np.unique(a['edge_base_instance']):
            assert len(np.unique(a['edge_compound_instance'][a['edge_base_instance']==ident]))==1
        for group in groups:
            np.testing.assert_array_equal(a['edge_compound_instance']==group['id'],
                                          np.isin(a['edge_base_instance'],group['base_instance_ids']))


def test_two_stage_model_backward_without_repeats():
    torch.set_num_threads(1)
    p=Part('test');circle(p,[0,0],20,'external_profile');slot(p,[0,0],12,4)
    m,c=record(p);prepared=prepare(c,128)
    t,_,_=targets(m,c,prepared);t['weights']=prepared.weights
    model=ContourTransformer(ModelConfig(width=32,global_width=64,queries=8,label_schema=LABEL_VERSION))
    out=model(torch.from_numpy(prepared.x),prepared.offsets,instances=True)
    loss=losses(out,{k:torch.as_tensor(v) for k,v in t.items()},torch.ones(4),torch.ones(5))
    loss.backward()
    assert torch.isfinite(loss)
    assert out['primary'].shape==(len(prepared.x),4)
    assert out['compound'].shape==(len(prepared.x),5)
    assert model.compound[0].weight.grad.abs().sum()>0


def test_prediction_is_label_blind_and_complete_circles_remain_whole():
    from contour_model.inference import predict
    p=Part('test');circle(p,[0,0],20,'external_profile');circle(p,[0,0],4,'hole',True)
    _,c=record(p)
    model=ContourTransformer(ModelConfig(width=32,global_width=64,queries=8,
        adaptive_sampling=True,sampling_version='adaptive-local-v2',label_schema=LABEL_VERSION)).eval()
    with torch.no_grad():
        model.primary.weight.zero_();model.primary.bias[:]=-10;model.primary.bias[3]=10
        model.compound[-1].weight.zero_();model.compound[-1].bias[:]=-10;model.compound[-1].bias[0]=10
        model.query_class.weight.zero_();model.query_class.bias[:]=-10;model.query_class.bias[0]=10
    clean={k:c[k] for k in ('actual_xy_mm','ring_offsets')}
    result,a=predict(model,clean,instances=True)
    poisoned={**clean,'nominal_xy_mm':np.zeros_like(c['actual_xy_mm']),
              'edge_base_class':np.full(len(c['actual_xy_mm']),3)}
    other,b=predict(model,poisoned,instances=True)
    assert result==other
    np.testing.assert_array_equal(a['primary'],b['primary'])
    assert len(result['features'])==2
    assert all(f['semantic']=='circle' and f['contour_spans'][0]['closed_ring'] for f in result['features'])
    assert not result['compound_features']


def test_thread_pairing_rejects_neighboring_zones_and_survives_rotation():
    from contour_model.inference_primitives import compatible_thread_sides
    x=np.linspace(0,12,100)
    top=np.c_[x,3+.25*np.sin(x*5)]
    bottom=top*[1,-1]
    neighbor=bottom+[15,0]
    q=np.vstack([top,bottom,neighbor]);weights=np.ones(len(q))
    features=[{'raw_edge_indices':list(range(i*100,(i+1)*100))} for i in range(3)]
    for angle in (0,.83,2.7):
        r=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        raw=q@r.T+[91,-73]
        assert compatible_thread_sides([0],[1],features,raw,weights)
        assert not compatible_thread_sides([0],[2],features,raw,weights)


def test_partly_recognized_thread_side_pairs_and_extends_over_the_zone():
    from contour_model.inference_primitives import compatible_thread_sides,extend_thread_side
    # One closed silhouette: top side left to right, then bottom side back.
    x=np.linspace(0,12,120)
    top=np.c_[x,3+.25*np.sin(x*5)];bottom=(top*[1,-1])[::-1]
    raw=np.vstack([top,bottom]);offsets=np.array([0,len(raw)])
    features=[{'raw_edge_indices':list(range(i,i+10)),'semantic_id':1,
               'contour_spans':[{'ring_index':0}]} for i in range(0,len(raw),10)]
    owner=np.repeat(np.arange(len(features)),10)
    full,partial=list(range(12)),[16,17,18,19]
    assert compatible_thread_sides(full,partial,features,raw,np.ones(len(raw)))
    extended=extend_thread_side(partial,full,features,raw,owner,offsets,set(full))
    assert sorted(extended)==list(range(12,24))


def test_confident_compound_posterior_cannot_turn_one_span_into_a_thread():
    from contour_model.inference import predict
    p=Part('test');circle(p,[0,0],20,'external_profile')
    _,c=record(p)
    model=ContourTransformer(ModelConfig(width=32,global_width=64,queries=8,
        adaptive_sampling=True,sampling_version='adaptive-local-v2',label_schema=LABEL_VERSION)).eval()
    with torch.no_grad():
        model.primary.weight.zero_();model.primary.bias[:]=-10;model.primary.bias[2]=10
        model.boundary.weight.zero_();model.boundary.bias[:]=-10
        model.compound[-1].weight.zero_();model.compound[-1].bias[:]=-10
        model.query_class.weight.zero_();model.query_class.bias[:]=-10;model.query_class.bias[0]=10
        for cls in (2,3):
            model.compound[-1].bias[:]=-10;model.compound[-1].bias[cls]=10
            result,_=predict(model,{k:c[k] for k in ('actual_xy_mm','ring_offsets')},instances=True)
            assert len(result['features'])==1
            assert not result['compound_features']


def test_labyrinth_straights_are_lines_and_buffer_arcs_keep_their_radius():
    for seed in range(8):
        p=expanded_nominal('labyrinth_plate',np.random.default_rng(seed))
        m,c=record(p);a,_=labels(m,c)
        for primitive in p.primitives:
            if primitive['kind']=='circular_arc':
                assert primitive['nominal_parameters']['radius']<=1.000001
        for lo,hi in zip(c['ring_offsets'][1:-1],c['ring_offsets'][2:]):
            q=c['nominal_xy_mm'][lo:hi];length=np.linalg.norm(np.roll(q,-1,axis=0)-q,axis=1)
            assert length[a['edge_base_class'][lo:hi]==1].sum()/length.sum()>.9


def test_label_repair_preserves_original_observations():
    from inspection_shapes.repair_primitives import repair
    p=Part('buffered');f=p.feature('rounded_profile');c=Chain(p)
    c.add([[0,0],[10,0],[10.05,.003]],f,'circular_arc',center=[5,10],radius=10,start_angle=0,end_angle=1)
    for a,b in [([10.05,.003],[10,5]),([10,5],[0,5]),([0,5],[0,0])]:c.line(a,b,f)
    c.close();m,raw=record(p)
    raw['actual_xy_mm']=raw['actual_xy_mm']+[.013,-.017]
    before=raw['actual_xy_mm'].copy();assert repair(m,raw)==1
    np.testing.assert_array_equal(before,raw['actual_xy_mm'])
    a,_=labels(m,raw);assert a['edge_base_class'][0]==1
