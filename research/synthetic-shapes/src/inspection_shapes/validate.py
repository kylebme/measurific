"""Dataset checks, including topology, label consistency, and image correspondence."""
from collections import Counter
import json
from pathlib import Path
import numpy as np
from shapely.geometry import Polygon
from .dataset import digest, iter_samples
from .render import Camera


def validate_dataset(output):
    output=Path(output);manifest=json.loads((output/"manifest.json").read_text())
    for entry in manifest["shards"]:
        if digest(output/entry["file"])!=entry["sha256"]:raise ValueError(f'Checksum mismatch: {entry["file"]}')
    families=Counter();splits=Counter();classes=Counter();severities=Counter();seen=set();design_splits={};anomalies=0
    for m,a in iter_samples(output):
        sid=m["sample_id"]
        assert sid not in seen,(sid,"duplicate sample");seen.add(sid)
        assert design_splits.setdefault(m["design_id"],m["split"])==m["split"],"Design leaks across splits"
        c=a["contours"];offsets=c["ring_offsets"];actual=c["actual_xy_mm"]
        assert offsets[0]==0 and offsets[-1]==len(actual) and (np.diff(offsets)>=3).all()
        rings=[actual[l:r] for l,r in zip(offsets[:-1],offsets[1:])]
        assert Polygon(rings[0],rings[1:]).is_valid,(sid,"invalid topology")
        np.testing.assert_allclose(Camera(**m["camera"]).project(actual),c["actual_xy_px"],atol=1e-8)
        for key in ("edge_feature_id","edge_primitive_id","edge_semantic_bits","edge_anomaly_class"):
            assert len(c[key])==len(actual)
        assert c["edge_feature_id"].min()>0 and c["edge_feature_id"].max()<=len(m["features"])
        assert c["edge_primitive_id"].min()>0 and c["edge_primitive_id"].max()<=len(m["primitives"])
        if "image" in a:
            shape=(m["camera"]["height"],m["camera"]["width"])
            assert all(a[k].shape==shape for k in ("image","material","region_semantic","boundary_instance","anomaly"))
            assert a["image"].dtype==np.uint8
            feature_lookup=np.array([0]+[f["semantic_id"] for f in m["features"]])
            np.testing.assert_array_equal(feature_lookup[a["boundary_instance"]],a["boundary_semantic"])
            np.testing.assert_array_equal(feature_lookup[a["region_instance"]],a["region_semantic"])
            bits=a["semantic_bits"]["bits"]
            for k in ("region_semantic","boundary_semantic"):
                mask=a[k]>0
                assert np.all((bits[mask] & (np.uint32(1)<<a[k][mask].astype(np.uint32)))!=0)
            interior=a["boundary_instance"]==0
            dark=a["image"][(a["material"]>0)&interior];bright=a["image"][(a["material"]==0)&interior]
            assert len(dark)>0 and len(bright)>0 and dark.mean()+80<bright.mean(),(sid,"image polarity/alignment")
            assert not np.any((a["anomaly"]>0)&(a["boundary_instance"]==0))
        families[m["family"]]+=1;splits[m["split"]]+=1;severities[str(m["imperfections"]["severity"])]+=1
        classes.update(f["semantic"] for f in m["features"]);anomalies+=len(m["anomalies"])
    assert len(seen)==sum(s["count"] for s in manifest["shards"])
    result=dict(samples=len(seen),families=dict(families),splits=dict(splits),features=dict(classes),
                severity_counts=dict(severities),anomaly_instances=anomalies,checks="passed")
    (output/"validation.json").write_text(json.dumps(result,indent=2)+"\n")
    return result
