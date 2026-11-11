"""Score reference and predicted feature selections against independent fixture truth."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import gzip
import hashlib
import html
import json
from pathlib import Path
import time
import traceback

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import linear_sum_assignment
import torch

from known_geometry import generate, FAMILIES, VERSION
from inspection_measurement import Contours, evaluate
from contour_model.inference import predict
from contour_model.train import load_model
from inspection_shapes.geometry import Part, Ring
from inspection_shapes.render import Camera, render


def write(path, data):
    if path.suffix == '.gz':
        with gzip.open(path, 'wt') as stream: json.dump(data, stream, allow_nan=False)
    else: path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def read(path):
    if path.suffix == '.gz':
        with gzip.open(path, 'rt') as stream: return json.load(stream)
    return json.loads(path.read_text())


def match_request(case, contours, prediction, threshold=.5):
    targets = [f for f in case['request']['features'] if 'selection' in f and not f.get('compose_selections_from')]
    primitive_schema = prediction.get('label_version') == 'primitives-compounds-v1'
    candidates = [f for f in prediction['features'] + prediction.get('compound_features', []) if f['status'] == 'accepted']
    # Construction provenance uses the old 16 labels. Match the new namespaces
    # by geometry identity; never equate a base ID with a compound/legacy ID.
    identities = {
        'line': ('base', {'line'}), 'arc': ('base', {'arc'}),
        'circle': ('base', {'circle'}), 'ellipse': ('base', {'arbitrary_profile'}),
        'slot': ('compound', {'full_round_slot'}),
        'repeated_profile': ('compound', {'tooth_profile'}),
    }

    weights = np.empty(len(contours.xy))
    for lo, hi in zip(contours.offsets[:-1], contours.offsets[1:]):
        q = contours.xy[lo:hi]; weights[lo:hi] = np.linalg.norm(np.roll(q,-1,axis=0)-q,axis=1)
    matrix = np.zeros((len(targets), len(candidates)))
    for i, f in enumerate(targets):
        a = np.asarray(f['selection']['edge_indices'],int)
        for j, g in enumerate(candidates):
            if primitive_schema:
                level, names = identities.get(f['type'], ('unsupported', set()))
                if g.get('label_level') != level or g.get('semantic') not in names: continue
            elif f['semantic_id'] != g['semantic_id']: continue
            b = np.asarray(g['raw_edge_indices'],int)
            intersection = weights[np.intersect1d(a,b)].sum()
            union = weights[a].sum()+weights[b].sum()-intersection
            matrix[i,j] = intersection/max(union,1e-12)
    rows, cols = linear_sum_assignment(-matrix)
    matches = {targets[i]['id']: {'prediction_id':candidates[j]['id'], 'iou':float(matrix[i,j]),
               'indices':candidates[j]['raw_edge_indices'], 'confidence':candidates[j]['confidence']}
               for i,j in zip(rows,cols) if matrix[i,j]>=threshold}
    req = deepcopy(case['request']); available = set(matches)
    for f in req['features']:
        if 'selection' not in f or f.get('compose_selections_from'): continue
        f['source'] = 'model'
        if f['id'] in matches:
            m = matches[f['id']];f.update(selection={'edge_indices':m['indices']},status='accepted',confidence=m['confidence'])
        else:
            # Remove reference support completely: an unmatched prediction cannot
            # accidentally inherit the known-good labels.
            f.pop('selection',None); f.update(status='abstain')
    for f in req['features']:
        sources = f.get('compose_selections_from')
        if not sources: continue
        f['source']='model'
        if all(ident in matches for ident in sources):
            f['selection']={'edge_indices':sorted({i for ident in sources for i in matches[ident]['indices']})}
            f['status']='accepted';available.add(f['id'])
        else:
            f.pop('selection',None);f['status']='abstain'
    return req, {'target_features':len(targets),'accepted_predictions':len(candidates),
                 'matched_target_features':len(matches),'unmatched_target_ids':[f['id'] for f in targets if f['id'] not in matches],
                 'matches':{k:{name:value for name,value in v.items() if name!='indices'} for k,v in matches.items()}}


def score(truth, report):
    measurements={m['id']:m for m in report['measurements']};out=[]
    for t in truth:
        row={**t};m=measurements[t['measurement_id']]
        if m['status']!='measured':
            row.update(status='unavailable' if m['status']=='unavailable' else 'measurement_failed',
                       reason=m.get('reason',m['status']),measured=None,error=None)
        else:
            try:
                a=np.asarray(m[t['field']],float);b=np.asarray(t['expected'],float)
                if a.shape!=b.shape: raise ValueError('Measured and expected dimensions differ')
                delta=np.abs(a-b)
                if 'period' in t:
                    delta=delta%t['period'];delta=np.minimum(delta,t['period']-delta)
                error=float(np.max(delta));relative=error/max(float(np.max(np.abs(b))),1e-15)
                if not np.isfinite(error):raise ValueError('Nonfinite measurement')
                row.update(status='pass' if error<=t['absolute_tolerance'] else 'outside_tolerance',
                           measured=m[t['field']],error=error,relative_error=relative)
            except (KeyError,ValueError,TypeError) as exception:
                row.update(status='measurement_failed',reason=str(exception),measured=None,error=None)
        out.append(row)
    return out


def render_case(case, contours):
    features=[{'id':i+1,'semantic_id':2 if hole else 1,'parent':0}
              for i,hole in enumerate(case['ring_is_hole'])]
    part=Part(case['family'],features=features)
    for i,(lo,hi) in enumerate(zip(contours.offsets[:-1],contours.offsets[1:])):
        q=contours.xy[lo:hi]*[1,-1];zero=np.zeros(len(q),int)
        part.rings.append(Ring(q,np.full(len(q),i+1),zero,case['ring_is_hole'][i],q,zero,zero))
    camera=Camera(width=1024,height=768,supersample=2,pixels_per_mm=1,translation_x=0,translation_y=0,
                  sigma_px=.8,read_noise_dn=.1,prnu=0,electrons_per_dn=10000)
    return render(part,camera,case['seed'])[0]['image']


def preview(image, case, reference, predicted, ref_scores, pred_scores):
    canvas=Image.new('RGB',(1740,850),'#111820');canvas.paste(Image.fromarray(image).convert('RGB'),(0,65))
    draw=ImageDraw.Draw(canvas);font=ImageFont.load_default(size=15)
    draw.text((16,14),case['id']+' | independently known measurement truth',fill='white',font=font)
    draw.text((16,38),'Cyan: reference support | orange: model support | images are for review, not measurement inputs',fill='#b9c8d8',font=font)
    # Highlight first failed predicted measurement, otherwise first plan feature.
    failed=next((s for s in pred_scores if s['status']!='pass'),pred_scores[0])
    req=next(m for m in case['request']['measurements'] if m['id']==failed['measurement_id']);refs=set(req['feature_refs'])
    for report,color in [(reference,'#00c2ff'),(predicted,'#ffaf51')]:
        for f in report['features']:
            if f['id'] not in refs:continue
            for q,closed in zip(f.get('support',[]),f.get('closed',[])):
                points=[(p[0],p[1]+65) for p in q]
                if closed:points.append(points[0])
                draw.line(points,fill=color,width=2)
    chosen=[];seen=set()
    for i in [j for j,s in enumerate(pred_scores) if s['status']!='pass']+list(range(len(case['truth']))):
        if i in seen:continue
        seen.add(i);chosen.append(i)
        if len(chosen)==13:break
    draw.text((1040,65),'Measurement: truth | reference | model',fill='white',font=font)
    for j,i in enumerate(chosen):
        t=case['truth'][i];a=ref_scores[i];b=pred_scores[i]
        def fmt(value):
            if value is None:return 'unavailable'
            if np.isscalar(value):return f'{value:.6g}'
            return '['+', '.join(f'{x:.4g}' for x in np.asarray(value).ravel()[:3])+']'
        y=94+j*54
        draw.text((1040,y),(t['measurement_id']+'.'+t['field'])[:75],fill='white',font=font)
        draw.text((1040,y+18),f"{fmt(t['expected'])} | {fmt(a['measured'])} | {fmt(b['measured'])}"[:79],fill='#b9c8d8',font=font)
        draw.text((1040,y+35),f"reference {a['status']} | model {b['status']}",fill='#6ae5a5' if a['status']==b['status']=='pass' else '#ff8890',font=font)
    return canvas


def summarize(root, manifest):
    samples=[json.loads(s) for s in (root/'samples.jsonl').read_text().splitlines()]
    aggregate={};families={};rows_by_lane={};cards=[]
    for lane in ('reference','predicted'):
        rows=[]
        for sample in samples:
            rows += [{**r,'sample_id':sample['id'],'family':sample['family']}
                     for r in read(root/'cases'/sample['id']/(lane+'-scores.json'))]
        rows_by_lane[lane]=rows;counts=Counter(r['status'] for r in rows)
        byop={};byfamily={}
        for name,selection in [('operator',byop),('family',byfamily)]:
            for key in sorted({r[name] for r in rows}):
                subset=[r for r in rows if r[name]==key];errors=[r['error'] for r in subset if r['error'] is not None]
                selection[key]={'checks':len(subset),'status':dict(Counter(r['status'] for r in subset)),
                                'median_error':float(np.median(errors)) if errors else None,
                                'p95_error':float(np.quantile(errors,.95)) if errors else None,
                                'maximum_error':float(max(errors)) if errors else None}
        worst=sorted([r for r in rows if r['error'] is not None],key=lambda r:r['error']/max(r['absolute_tolerance'],1e-15),reverse=True)[:30]
        aggregate[lane]={'checks':len(rows),'status':dict(counts),'by_operator':byop,'by_family':byfamily,
                         'pass_rate_all_requested':counts['pass']/max(1,len(rows)),
                         'worst_by_tolerance_ratio':worst}
    summary={'protocol':manifest,'samples':len(samples),'lanes':aggregate,
             'requested_feature_targets':sum(s['matching']['target_features'] for s in samples),
             'matched_feature_targets':sum(s['matching']['matched_target_features'] for s in samples),
             'measurement_timing_seconds':{lane:{'median':float(np.median([s[lane+'_seconds'] for s in samples])),
                 'p95':float(np.quantile([s[lane+'_seconds'] for s in samples],.95))}
                 for lane in ('reference','predicted') if all(lane+'_seconds' in s for s in samples)}}
    write(root/'summary.json',summary)
    text=['# Independent known-truth measurement benchmark','',
          f"{len(samples)} deterministic analytic parts, {len(FAMILIES)} fixture families. Ground truth was saved before measurement or model inference.",'',
          'The same fixed inspection plan is evaluated using reference selections and model-selected support; both are compared with independent construction formulas. Model matches are established only after inference by taxonomy-aware geometry identity and arc-length IoU >=0.5. Missing matches remain unavailable and stay in the denominator. The model is never given truth dimensions, labels, or parameters.','',
          '| Selection lane | Requested checks | Within declared error bound | Outside bound | Unavailable | Measurement failed |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for lane,s in aggregate.items():
        c=s['status'];text.append(f"| {lane} | {s['checks']} | {c.get('pass',0)} | {c.get('outside_tolerance',0)} | {c.get('unavailable',0)} | {c.get('measurement_failed',0)} |")
    text += ['','## Reference-selection correctness by operator','','| Operator | Checks | Pass | Outside bound | Failed/unavailable | Max absolute error* |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for op,s in aggregate['reference']['by_operator'].items():
        c=s['status'];text.append(f"| {op} | {s['checks']} | {c.get('pass',0)} | {c.get('outside_tolerance',0)} | {c.get('unavailable',0)+c.get('measurement_failed',0)} | {s['maximum_error']} |")
    text += ['','*An operator may report multiple fields with different units; per-check JSON includes exact units, expected values, formulas, errors and bounds. Summary maxima are diagnostic, not a pooled dimensional error metric.','',
             '## Interpretation','',
             '- Exact line/circle/slot/ellipse and associated arc quantities use independently prescribed parameters.',
             '- Finite polygon area/perimeter are compared with polygon formulas, not smooth circumference/area.',
             '- Circle-to-circle finite-polyline clearance has a predefined chord-sagitta error bound; linear repetition pitch has a predefined resampling-grid bound.',
             '- Bowed-line straightness and RMS, and symmetric lobed-circle radial variation/RMS, are nonzero analytically known form errors.',
             '- Pass means within the benchmark numerical/discretization bound. These bounds are not production part tolerances or instrument uncertainty.',
             '- All measurements use stored contour points directly. Renderer images are review artifacts; PAE, optics and calibration are excluded.',
             '- Feature type is specified by the inspection plan in both lanes; the model supplies selected edge indices. No reference points are substituted when model matching fails.',
             '- The new fixtures differ from the training distribution. Predicted-lane results combine recognition/selection errors with measurement behavior; reference-lane results isolate measurement correctness on the defined geometry.','',
             '## Artifacts','',
             '`generator_snapshot.py`, `manifest.json` and every case `truth.json` preserve the exact generator, seed, construction parameters, formulas and error bounds. Each case also contains contours, predictions, requests, measured reports, scored comparisons, and an annotated preview. `index.html` shows every preview.']
    (root/'REPORT.md').write_text('\n'.join(text)+'\n')
    for sample in samples:
        ident=sample['id'];stats=' | '.join(f"{lane}: {sample[lane]}" for lane in ('reference','predicted'))
        cards.append(f'<article><h2>{html.escape(ident)}</h2><p>{html.escape(stats)}</p><a href="cases/{ident}/annotated.png"><img loading="lazy" src="cases/{ident}/annotated.png"></a></article>')
    (root/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Known measurement truth</title><style>body{background:#111820;color:#ddd;font:14px sans-serif;margin:20px}main{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}img{width:100%}h2{font-size:16px}article{background:#1c2630;padding:10px}</style><h1>Independent known-truth measurement evaluation</h1><p>Reference and predicted selections are each compared with construction truth. Up to 13 checks shown per preview; full scores are retained.</p><main>'+''.join(cards)+'</main>')
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--per-family',type=int,default=25);p.add_argument('--seed',type=int,default=20260908)
    p.add_argument('--checkpoint',type=Path,default=Path('packages/contour-model/release-v2/model.pt'))
    p.add_argument('--reuse',type=Path,help='Reuse immutable fixtures and predictions from a prior run')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'manifest.json').exists():raise FileExistsError('Choose a fresh output directory')
    source=Path(__file__).with_name('known_geometry.py')
    manifest={'generator_version':VERSION,'generator_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
              'base_seed':a.seed,'per_family':a.per_family,'families':FAMILIES,
              'checkpoint':str(a.checkpoint.resolve()),'checkpoint_sha256':hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
              'measurement_source_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(Path('packages/measurement/src/inspection_measurement').glob('*.py'))},
              'truth_independence':'Generator imports only numpy and standard library, never measurement fitters/operators. All expected results are written before inference/evaluation.',
              'matching':'Accepted predictions, taxonomy-aware geometry identity in separate base/compound namespaces, one-to-one weighted edge IoU>=0.5; fixed inspection-plan types; missing predictions are not replaced.',
              'matching_version':'primitives-compounds-v1', 'input':'Analytic pixel contours; no image extraction', 'cases':[]}
    if a.reuse:
        oldmanifest=read(a.reuse/'manifest.json')
        for key in ('generator_sha256','base_seed','per_family','checkpoint_sha256'):
            if manifest[key]!=oldmanifest[key]:raise ValueError('Reuse mismatch: '+key)
        manifest['reuse_from']=str(a.reuse.resolve())
    (a.output/'generator_snapshot.py').write_bytes(source.read_bytes())
    # Freeze all expected values before loading model or calling measurement code.
    for i,family in enumerate(FAMILIES):
        for j in range(a.per_family):
            seed=a.seed+i*10000+j;case=generate(family,seed);ident=case['id'];directory=a.output/'cases'/ident;directory.mkdir(parents=True)
            c=case.pop('contours');np.savez_compressed(directory/'contours.npz',xy=np.asarray(c['xy']),offsets=np.asarray(c['offsets']),closed=np.asarray(c['closed']))
            write(directory/'truth.json',case)
            manifest['cases'].append({'id':ident,'family':family,'seed':seed,'truth_sha256':hashlib.sha256((directory/'truth.json').read_bytes()).hexdigest(),
                                      'contours_sha256':hashlib.sha256((directory/'contours.npz').read_bytes()).hexdigest()})
    write(a.output/'manifest.json',manifest)
    torch.set_num_threads(4);model,ckpt=load_model(a.checkpoint,'cpu');start=time.perf_counter()
    with (a.output/'samples.jsonl').open('w') as log:
        for index,e in enumerate(manifest['cases'],1):
            directory=a.output/'cases'/e['id'];case=read(directory/'truth.json')
            with np.load(directory/'contours.npz') as arr:c=Contours(arr['xy'],arr['offsets'],arr['closed'])
            previous=a.reuse/'cases'/e['id'] if a.reuse else None
            if previous and (previous/'prediction.json.gz').exists():prediction=read(previous/'prediction.json.gz')
            else:prediction,_=predict(model,c.model_arrays(),threshold=.7,instances=ckpt['instances_trained'],units='px')
            write(directory/'prediction.json.gz',prediction)
            # Exercise the exact GUI handoff as well as the fixed inspection plan.
            handoff = evaluate(c, prediction)
            write(directory/'handoff-report.json.gz', handoff)
            predicted_req,matching=match_request(case,c,prediction)
            before=time.perf_counter();reference=evaluate(c,case['request']);reference_seconds=time.perf_counter()-before
            before=time.perf_counter();predicted=evaluate(c,predicted_req);predicted_seconds=time.perf_counter()-before
            ref_scores=score(case['truth'],reference);pred_scores=score(case['truth'],predicted)
            for lane,req,report,scores in [('reference',case['request'],reference,ref_scores),('predicted',predicted_req,predicted,pred_scores)]:
                write(directory/(lane+'-request.json'),req);write(directory/(lane+'-report.json.gz'),report);write(directory/(lane+'-scores.json'),scores)
            image=np.asarray(Image.open(previous/'input.png')) if previous and (previous/'input.png').exists() else render_case(case,c)
            Image.fromarray(image).save(directory/'input.png');preview(image,case,reference,predicted,ref_scores,pred_scores).save(directory/'annotated.png')
            record={'id':e['id'],'family':e['family'],'reference':dict(Counter(r['status'] for r in ref_scores)),
                    'predicted':dict(Counter(r['status'] for r in pred_scores)),'matching':matching,
                    'reference_seconds':reference_seconds,'predicted_seconds':predicted_seconds, 'handoff_summary':handoff['summary']}
            log.write(json.dumps(record)+'\n');log.flush()
            print(f"{index}/{len(manifest['cases'])} {e['id']} reference={record['reference']} predicted={record['predicted']}",flush=True)
    summary=summarize(a.output,manifest)
    print(json.dumps({'seconds':time.perf_counter()-start,'lanes':{k:v['status'] for k,v in summary['lanes'].items()}}),flush=True)


if __name__=='__main__':main()
