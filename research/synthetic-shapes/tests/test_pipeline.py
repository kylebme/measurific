import json
import numpy as np
import pytest
from inspection_shapes.geometry import generate,valid
from inspection_shapes.render import render,random_camera,Camera
from inspection_shapes.taxonomy import PARTS
from inspection_shapes.dataset import write_dataset,iter_samples,split_for
from inspection_shapes.validate import validate_dataset

@pytest.mark.parametrize("family",PARTS)
def test_topology_ownership_and_correspondence(family):
    for seed in range(12):
        p=generate(family,seed,severity=2.5)
        assert valid(p) and valid(p,False)
        for ring in p.rings:
            assert ring.actual.shape==ring.nominal.shape
            assert len(ring.feature)==len(ring.primitive)==len(ring.actual)
            assert np.max(np.abs(ring.actual-ring.nominal))>0
            xy=ring.nominal
            area=np.sum(xy[:,0]*np.roll(xy[:,1],-1)-xy[:,1]*np.roll(xy[:,0],-1))/2
            assert (area<0)==ring.hole


def test_zero_severity_and_determinism():
    for family in PARTS:
        p=generate(family,491,severity=0)
        q=generate(family,491,severity=0)
        for a,b in zip(p.rings,q.rings):
            np.testing.assert_array_equal(a.nominal,a.actual)
            np.testing.assert_array_equal(a.actual,b.actual)
            assert not a.defect.any()


def test_circle_has_form_error_and_bore_offset():
    p=generate("washer",561,severity=2)
    q=p.rings[1].actual
    center=q.mean(axis=0)
    assert np.linalg.norm(center)>.001
    radius=np.linalg.norm(q-center,axis=1)
    assert np.ptp(radius)>.01


def test_independent_line_form_errors():
    p=generate("drilled_plate",771,severity=2)
    errors=p.imperfections["rings"][0]["primitive_bow_skew_mm"]
    assert len(errors)>=4 and np.std([e[2] for e in errors])>.001
    assert abs(p.imperfections["affine_shear"])>.00001


def test_render_alignment_distortion_and_ancestor_labels():
    p=generate("slotted_bracket",871,severity=1)
    camera=random_camera(p,np.random.default_rng(71),480,270,2)
    camera.k1=.025;camera.p1=.001
    a,px,timing=render(p,camera,88)
    b,_,_=render(p,camera,88)
    np.testing.assert_array_equal(a["image"],b["image"])
    assert a["image"].shape==(270,480)
    assert (a["boundary_instance"]>0).any()
    assert a["image"][a["material"]>0].mean()<60
    assert a["image"][a["material"]==0].mean()>170
    # Closed hole/slot centers must be background material and feature-labeled regions.
    for ring in p.rings[1:]:
        uv=camera.project(ring.actual.mean(axis=0,keepdims=True))[0]
        x,y=np.rint(uv).astype(int)
        assert a["material"][y,x]==0 and a["region_instance"][y,x]==ring.feature[0]
    assert a["semantic_bits"].dtype==np.uint32


def test_dataset_roundtrip_resume_config_and_leakage(tmp_path):
    kwargs=dict(output=tmp_path,count=4,shard_size=2,workers=1,width=192,height=108,views_per_design=2)
    first=write_dataset(**kwargs)
    again=write_dataset(**kwargs)
    assert [s["sha256"] for s in first]==[s["sha256"] for s in again]
    records=list(iter_samples(tmp_path))
    np.testing.assert_array_equal(records[0][1]["contours"]["actual_xy_mm"],records[1][1]["contours"]["actual_xy_mm"])
    assert records[0][0]["split"]==records[1][0]["split"]
    assert validate_dataset(tmp_path)["samples"]==4
    with pytest.raises(ValueError,match="configuration"):
        write_dataset(**{**kwargs,"seed":1})
    shard=tmp_path/first[0]["file"]
    with shard.open("ab") as f:f.write(b"corrupt")
    with pytest.raises(ValueError,match="Corrupt"):
        write_dataset(**kwargs)


def test_nonparallelism_changes_opposite_surface_slopes():
    p=generate("drilled_plate",771,severity=2)
    ring=p.rings[0]
    pids=np.unique(ring.primitive)
    slopes=[]
    for pid in (pids[0],pids[4]):
        q=ring.actual[ring.primitive==pid]
        slopes.append(np.polyfit(q[:,0],q[:,1],1)[0])
    assert abs(slopes[0]-slopes[1])>.0001


def test_shrink_and_partial_shard_overlap_rejected(tmp_path):
    kwargs=dict(output=tmp_path,count=3,shard_size=2,workers=1,contours_only=True)
    write_dataset(**kwargs)
    with pytest.raises(ValueError,match="shrink"):
        write_dataset(**{**kwargs,"count":2})
    with pytest.raises(ValueError,match="overlap"):
        write_dataset(**{**kwargs,"count":4})


def test_contour_only_can_be_rendered_later(tmp_path):
    from inspection_shapes.dataset import part_from_record
    write_dataset(tmp_path,count=1,shard_size=1,workers=1,width=192,height=108,contours_only=True)
    metadata,arrays=next(iter_samples(tmp_path))
    assert 'image' not in arrays
    part=part_from_record(metadata,arrays['contours'])
    a,_,_=render(part,Camera(**metadata['camera']),metadata['appearance_seed'])
    assert a['image'].shape==(108,192)
    np.testing.assert_array_equal(part.rings[0].actual,arrays['contours']['actual_xy_mm'])
    part.rings[0].actual[0]+=1
    assert not np.array_equal(part.rings[0].actual,arrays['contours']['actual_xy_mm'])


def test_subpixel_raster_coverage_preserves_continuous_area_and_boundary_cells():
    from inspection_shapes.render import rasterize_coverage
    # Fractional vertices expose integer truncation and inclusive fill bias.
    for ss in [1, 2, 3, 4]:
        for dx, dy in [(0., 0.), (.17, .29), (.53, .71)]:
            q = np.array([[10.2, 10.3], [24.7, 10.3], [24.7, 28.8], [10.2, 28.8]])+[dx, dy]
            coverage = rasterize_coverage([q], [False], 40, 40, ss)
            assert abs(float(coverage.sum())/ss**2-14.5*18.5) < 5e-5
            # Exact occupancy of selected boundary cells, independently clipped.
            from shapely.geometry import Polygon, box
            polygon = Polygon(q*ss+(ss-1)/2)
            for y, x in np.argwhere((coverage > 0) & (coverage < 1))[::7]:
                expected = polygon.intersection(box(x-.5, y-.5, x+.5, y+.5)).area
                assert abs(coverage[y, x]-expected) < 1e-6
