import numpy as np
import pytest
import torch
from inspection_shapes.geometry import generate
from contour_model.data import prepare,targets,adapted_labels,augment
from contour_model.model import ContourTransformer,ModelConfig
from contour_model.losses import segmentation_loss,instance_loss,tensor_targets
from contour_model.inference import runs,predict,describe_selection
from contour_model.measurement import line_fit,circle_fit,slot_fit,MeasurementRouter


def fixture(family='slotted_bracket',seed=12):
    p=generate(family,seed)
    c={'actual_xy_mm':np.concatenate([r.actual for r in p.rings]),'ring_offsets':np.r_[0,np.cumsum([len(r.actual) for r in p.rings])],
       'edge_feature_id':np.concatenate([r.feature for r in p.rings]),'edge_anomaly_class':np.concatenate([r.defect for r in p.rings])}
    from inspection_shapes.render import feature_bits
    c['edge_semantic_bits']=feature_bits(p)[c['edge_feature_id']]
    return {'features':p.features},c


def test_resampling_label_blind_and_overlap_conserves():
    m,c=fixture();p=prepare(c,256);d={k:v.copy() for k,v in c.items()};d['edge_feature_id'][:]=999;d['edge_semantic_bits'][:]=0
    q=prepare(d,256);np.testing.assert_array_equal(p.x,q.x)
    np.testing.assert_allclose(p.overlap.sum(1),1,atol=1e-6)
    assert (np.asarray(p.overlap.sum(0))>0).all()
    t=targets(m,c,p);np.testing.assert_allclose(t['primary'].sum(1),1,atol=1e-6)


def test_transform_and_ring_winding():
    _,c=fixture();p=prepare(c,512);d={**c,'actual_xy_mm':c['actual_xy_mm']*3+[15,9]}
    q=prepare(d,512);np.testing.assert_allclose(p.x,q.x,atol=2e-5)
    points=[]
    for a,b in zip(c['ring_offsets'][:-1],c['ring_offsets'][1:]):points.append(c['actual_xy_mm'][a:b][::-1])
    rev=prepare({**c,'actual_xy_mm':np.concatenate(points)},512)
    assert np.isfinite(rev.x).all();assert np.array_equal(rev.hole,p.hole)


def test_augmentation_scale_and_targets():
    m,c=fixture();p=prepare(c,256);a={'x':p.x,'offsets':p.offsets,'weights':p.weights,**targets(m,c,p)}
    b=augment(a,np.random.default_rng(4),noise=0)
    np.testing.assert_allclose(sorted(b['x'][:,4]),sorted(a['x'][:,4]),rtol=1e-5)
    np.testing.assert_allclose(np.sort(b['semantic'].sum(1)),np.sort(a['semantic'].sum(1)))


def test_tab_grouping_and_external_ancestry():
    for family in ['slotted_bracket','polygon_spacer','cam','toothed_disk']:
        m,c=fixture(family,22);ids,cl,sem=adapted_labels(m,c)
        end=c['ring_offsets'][1];assert np.all(sem[:end,0]==1)
        for semantic in [15,16]:
            for run in runs(cl[:end]==semantic):assert len(np.unique(ids[run]))==1


def test_models_forward_backward_and_permuted_rings():
    torch.set_num_threads(2)
    m,c=fixture();p=prepare(c,256);t=targets(m,c,p)
    a={'weights':p.weights,**t};tt=tensor_targets(a,'cpu')
    for backbone in ['transformer','conv']:
        model=ContourTransformer(ModelConfig(width=32,global_width=64,global_depth=1,queries=32,backbone=backbone))
        x=torch.tensor(p.x);out=model(x,p.offsets,True)
        loss,_=segmentation_loss(out,tt);loss=loss+instance_loss(out,tt);loss.backward()
        assert np.isfinite(float(loss.detach()));assert model.embed[0].weight.grad is not None
        assert out['semantic'].shape==(len(x),16)
        model.eval()
        order=list(range(len(p.offsets)-1))[::-1];perm=np.concatenate([np.arange(p.offsets[i],p.offsets[i+1]) for i in order]);off=np.r_[0,np.cumsum(np.diff(p.offsets)[order])]
        with torch.no_grad():
            expected=model(x,p.offsets)['primary'][perm];actual=model(x[perm],off)['primary']
        torch.testing.assert_close(expected,actual,atol=1e-5,rtol=1e-5)


def test_inference_ignores_ground_truth():
    torch.set_num_threads(2);_,c=fixture('washer');model=ContourTransformer(ModelConfig(width=32,global_width=64,global_depth=1))
    model.eval();a,pa=predict(model,c,tokens=128)
    d={'actual_xy_mm':c['actual_xy_mm'],'ring_offsets':c['ring_offsets']};b,pb=predict(model,d,tokens=128)
    np.testing.assert_array_equal(pa['primary'],pb['primary']);assert a['coordinate_hash']==b['coordinate_hash']


def test_wrapped_spans():
    import json
    _,c=fixture('washer');n=c['ring_offsets'][1];ind=np.r_[0,1,n-2,n-1]
    s=describe_selection(ind,c);assert len(s)==1 and s[0]['wraps'] and s[0]['edge_count']==4
    assert json.loads(json.dumps(s))==s
    whole=describe_selection(np.arange(n),c)
    assert json.loads(json.dumps(whole))[0]['closed_ring'] is True


def test_classical_fits():
    t=np.linspace(0,2*np.pi,600,endpoint=False);q=np.c_[4+3*np.cos(t),-2+3*np.sin(t)]
    result=circle_fit(q);assert result['diameter']==pytest.approx(6,abs=1e-6)
    line=line_fit(np.c_[np.linspace(-2,2,100),np.linspace(-2,2,100)*.4+3]);assert line['rms_profile_residual']<1e-10
    from inspection_shapes.geometry import Part,slot
    p=Part('test');slot(p,[2,3],12,4,angle=.4)
    result=slot_fit(p.rings[0].nominal);assert result['width']==pytest.approx(4,abs=.001);assert result['overall_length']==pytest.approx(12,abs=.001)


def test_router_abstains():
    result=MeasurementRouter().measure({}, {'id':'x','semantic':'hole','units':'mm','status':'abstain'})
    assert result['status']=='abstain'


def test_thread_reference_pitch_and_raw_selection():
    from contour_model.measurement import thread_fit,selected_points
    from inspection_shapes.geometry import generate
    for seed in range(5):
        p=generate('threaded_shaft',seed,severity=0)
        f=next(f for f in p.features if f['semantic']=='thread');ring=p.rings[0]
        c={'actual_xy_mm':ring.actual,'ring_offsets':np.array([0,len(ring.actual)])}
        indices=np.flatnonzero(ring.feature==f['id'])
        feature={'raw_edge_indices':indices.tolist(),'contour_spans':describe_selection(indices,c)}
        result=thread_fit(c,feature)
        assert result['pitch']==pytest.approx(f['nominal_parameters']['pitch'],rel=.025)
        assert len(selected_points(c,feature))>=len(indices)


def test_bridge_transfer_rejects_topology_change():
    from contour_model.bridge import transfer
    m,c=fixture('washer');c['actual_xy_px']=c['actual_xy_mm']*[20,-20]
    bad={'actual_xy_mm':c['actual_xy_mm'][:c['ring_offsets'][1]]*20,'ring_offsets':c['ring_offsets'][:2]}
    with pytest.raises(ValueError,match='topology'):transfer(c,bad)


def test_surface_splits_do_not_create_array_seam_boundary():
    from contour_model.inference import split_surface_run
    indices=np.arange(100,140)
    pieces=split_surface_run(indices,[5,15,25,35],40)
    assert len(pieces)==4
    assert all(len(p)==10 for p in pieces)
    np.testing.assert_array_equal(pieces[-1],np.r_[135:140,100:105])
    np.testing.assert_array_equal(np.sort(np.concatenate(pieces)),indices)
    assert len(split_surface_run(indices,[],40))==1
    np.testing.assert_array_equal(split_surface_run(indices,[5],40)[0],np.r_[105:140,100:105])
    assert [len(p) for p in split_surface_run(indices,[5,15,25,35],50)]==[5,10,10,10,5]


def test_export_roundtrip_and_overwrite_guard(tmp_path):
    from contour_model.export import export_model
    from contour_model.train import load_model
    from contour_model.data import LABEL_VERSION
    model=ContourTransformer(ModelConfig(width=32,global_width=64,global_depth=1))
    source=tmp_path/'training.pt';destination=tmp_path/'model.pt'
    torch.save({'model':model.state_dict(),'model_config':model.metadata(),'step':12,
                'label_version':LABEL_VERSION,'instances_trained':True,'optimizer':{'discard':123}},source)
    export_model(source,destination,256)
    loaded,artifact=load_model(destination)
    assert 'optimizer' not in artifact and artifact['settings']['tokens']==256
    assert artifact['input_contract']['required_token_budget']==256
    from contour_model.inference import predict_file
    with pytest.raises(ValueError,match='requires --tokens 256'):
        predict_file(destination,tmp_path/'nonexistent.npz',tmp_path/'out.json',tokens=512)
    for a,b in zip(model.parameters(),loaded.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
    with pytest.raises(FileExistsError):export_model(source,destination)


def test_invalid_training_settings_fail_before_creating_run(tmp_path):
    from contour_model.train import train
    with pytest.raises(ValueError,match='either'):train(tmp_path,tmp_path/'run',init='a',resume='b')
    with pytest.raises(ValueError,match='positive'):train(tmp_path,tmp_path/'run',steps=0)
    assert not (tmp_path/'run').exists()


def test_dense_grammar_and_adaptive_sampling_without_label_leakage():
    from inspection_shapes.expanded import expanded_nominal
    from inspection_shapes.geometry import valid
    from contour_model.sampling import adaptive_counts
    p=expanded_nominal('perforated_grid_plate',np.random.default_rng(47),'extreme')
    assert len(p.rings)==577 and valid(p,False)
    c={'actual_xy_mm':np.concatenate([r.nominal for r in p.rings]),'ring_offsets':np.r_[0,np.cumsum([len(r.nominal) for r in p.rings])]}
    counts=adaptive_counts(c);assert len(counts)==577 and (counts>=32).all() and counts.sum()<=65536
    altered={**c,'edge_feature_id':np.full(len(c['actual_xy_mm']),999)}
    np.testing.assert_array_equal(counts,adaptive_counts(altered))
    p=expanded_nominal('multi_step_shaft',np.random.default_rng(47),'extreme')
    assert valid(p,False) and len(np.unique(p.rings[0].feature))>=384


def test_topological_instances_exceed_learned_query_capacity():
    from inspection_shapes.geometry import Part,plate_outline,circle,imperfect
    from inspection_shapes.render import feature_bits
    from contour_model.sampling import prepare_adaptive
    p=Part('grid');plate_outline(p,28,28,1)
    for y in range(-8,9,4):
        for x in range(-8,9,4):circle(p,[x,y],.8,'hole',True)
    imperfect(p,np.random.default_rng(3),0)
    c={'actual_xy_mm':np.concatenate([r.actual for r in p.rings]),'ring_offsets':np.r_[0,np.cumsum([len(r.actual) for r in p.rings])],
       'edge_feature_id':np.concatenate([r.feature for r in p.rings]),'edge_anomaly_class':np.concatenate([r.defect for r in p.rings])}
    c['edge_semantic_bits']=feature_bits(p)[c['edge_feature_id']]
    prep=prepare_adaptive(c);t=targets({'features':p.features},c,prep)
    model=ContourTransformer(ModelConfig(width=32,global_width=64,global_depth=1,queries=16,adaptive_sampling=True,topology_instances=True))
    x=torch.tensor(prep.x);out=model(x,prep.offsets,True)
    loss=instance_loss(out,tensor_targets({'weights':prep.weights,**t},'cpu'));loss.backward();assert torch.isfinite(loss)
    model.eval()
    with torch.no_grad():out=model(x,prep.offsets,True)
    assert out['query_masks'].shape==(16+25,len(x))
    for i,(lo,hi) in enumerate(zip(prep.offsets[1:-1],prep.offsets[2:])):
        selected=out['query_masks'][16+i]>0
        assert selected.sum()==hi-lo and selected[lo:hi].all()


def test_reject_single_point_touching_holes_and_shell():
    from contour_model.data import ring_geometry
    outer=np.array([[-5,-5],[5,-5],[5,5],[-5,5]])
    first=np.array([[-2,0],[-1,-1],[0,0],[-1,1]])
    second=first+[2,0]
    with pytest.raises(ValueError,match='Touching'):
        ring_geometry({'actual_xy_mm':np.concatenate([outer,first,second]),'ring_offsets':np.array([0,4,8,12])})
    touching=first+[5,0]
    with pytest.raises(ValueError):
        ring_geometry({'actual_xy_mm':np.concatenate([outer,touching]),'ring_offsets':np.array([0,4,8])})


def test_nonuniform_projection_preserves_physical_arc_weight():
    from types import SimpleNamespace
    from scipy.sparse import csr_matrix
    from contour_model.inference import mapped_probabilities
    prepared=SimpleNamespace(overlap=csr_matrix([[1.,0.],[.25,.75]]),weights=np.array([1.,4.]))
    result=mapped_probabilities(prepared,np.array([[0.],[1.]]))
    np.testing.assert_allclose(result[:,0],[.5,1.])
    assert float(result[:,0]@np.array([2.,3.]))==pytest.approx(4.)


def test_local_sampling_geometry_only_and_arc_conservation():
    from contour_model.sampling import prepare_adaptive,LOCAL_VERSION
    m,c=fixture('slotted_bracket',22)
    p=prepare_adaptive(c,version=LOCAL_VERSION)
    d={**c,'edge_feature_id':np.zeros(len(c['actual_xy_mm']),np.int64),'edge_semantic_bits':np.zeros(len(c['actual_xy_mm']),np.uint32)}
    other=prepare_adaptive(d,version=LOCAL_VERSION)
    np.testing.assert_array_equal(p.x,other.x)
    np.testing.assert_allclose(p.overlap.sum(1),1,atol=1e-6)
    raw_lengths=np.concatenate([np.linalg.norm(np.roll(c['actual_xy_mm'][lo:hi],-1,axis=0)-c['actual_xy_mm'][lo:hi],axis=1) for lo,hi in zip(c['ring_offsets'][:-1],c['ring_offsets'][1:])])
    area=np.asarray(p.overlap.multiply(p.weights[:,None]).sum(0)).ravel()
    np.testing.assert_allclose(area,raw_lengths,rtol=1e-6,atol=1e-8)
    t=targets(m,c,p);np.testing.assert_allclose(t['primary'].sum(1),1,atol=1e-6)


def test_cache_sampling_mismatch_is_explicit(tmp_path):
    import json
    from contour_model.train import validate_cache_sampling
    model=ContourTransformer(ModelConfig(width=32,global_width=64,global_depth=1,adaptive_sampling=True,sampling_version='adaptive-local-v2'))
    (tmp_path/'config.json').write_text(json.dumps({'sampling':'adaptive-turn-v1'}))
    with pytest.raises(ValueError,match='does not match'):validate_cache_sampling(model,tmp_path)
    (tmp_path/'config.json').write_text(json.dumps({'sampling':'adaptive-local-v2'}))
    validate_cache_sampling(model,tmp_path)


def test_boundary_event_survives_dense_local_sampling():
    from types import SimpleNamespace
    from scipy.sparse import csr_matrix
    from contour_model.inference import mapped_boundary_events
    prepared=SimpleNamespace(overlap=csr_matrix([[1.,0.],[1.,0.],[0.,1.]]))
    np.testing.assert_allclose(mapped_boundary_events(prepared,np.array([.9,.1,.2])),[.9,.2])
    refined=SimpleNamespace(overlap=csr_matrix([[1.,0.],[1.,0.],[1.,0.],[0.,1.]]))
    np.testing.assert_allclose(mapped_boundary_events(refined,np.array([.9,.1,.1,.2])),[.9,.2])


def test_short_thread_zone_axis_comes_from_continuous_side():
    from contour_model.measurement import thread_fit
    u=np.linspace(0,8,801);top=np.c_[u,15+.25*np.cos(2*np.pi*u)]
    bottom=top*[1,-1];xy=np.vstack([top,bottom[::-1]])
    c={'actual_xy_mm':xy,'ring_offsets':np.array([0,len(xy)])}
    indices=np.r_[np.arange(800),np.arange(801,1601)]
    feature={'raw_edge_indices':indices.tolist(),'contour_spans':describe_selection(indices,c)}
    assert thread_fit(c,feature)['pitch']==pytest.approx(1.,abs=.005)


def test_repeat_checkpoint_selection_prioritizes_semantic_recognition():
    from contour_model.train import checkpoint_score
    surface={'macro_f1':.982,'instances':{'precision_iou50':.783,'recall_iou50':.878}}
    forgotten={'macro_f1':.907,'instances':{'precision_iou50':.920,'recall_iou50':.851}}
    assert checkpoint_score(surface,True,'repeats')[0]>checkpoint_score(forgotten,True,'repeats')[0]
    assert checkpoint_score(surface,False)[0]==.982


def test_observable_reference_combines_only_collinear_flat_subdivisions():
    from contour_model.validation import observable_reference_labels
    nominal=np.array([[0.,0.],[1.,0.],[2.,0.],[2.,1.],[0.,1.]])
    actual=nominal.copy();actual[1,1]=.01
    c={'actual_xy_mm':actual,'nominal_xy_mm':nominal,'ring_offsets':np.array([0,5]),
       'edge_feature_id':np.arange(1,6),'edge_semantic_bits':np.full(5,1<<4,np.uint32)}
    ids,classes,_=observable_reference_labels({'features':[{'semantic_id':4} for _ in range(5)]},c)
    assert ids[0]==ids[1] and len(np.unique(ids))==4
    np.testing.assert_array_equal(classes,np.full(5,4))


def test_short_land_corner_test_does_not_reach_neighboring_notches():
    from contour_model.inference import observed_turn,surface_run_turn
    vertices=np.array([[-1.,-4.],[-1.,0.],[0.,0.],[0.,-4.],[2.,-4.],[2.,2.],[-2.,2.],[-2.,-4.]])
    q=np.concatenate([np.linspace(a,b,int(round(np.linalg.norm(b-a)*100))+1)[:-1] for a,b in zip(vertices,np.roll(vertices,-1,axis=0))])
    arc=np.r_[0,np.cumsum(np.linalg.norm(np.roll(q,-1,axis=0)-q,axis=1))]
    run=np.flatnonzero((q[:,1]==0)&(q[:,0]>=-1)&(q[:,0]<0));middle=len(run)//2
    assert observed_turn(q,arc,run,1.5)[middle]>.25
    assert surface_run_turn(q,arc,run,1.5)[middle]<1e-6
