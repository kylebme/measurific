import argparse
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description='Contour-only semantic recognition and raw-edge measurement handoff')
    sub=p.add_subparsers(dest='command',required=True)
    x=sub.add_parser('export');x.add_argument('checkpoint');x.add_argument('output');x.add_argument('--tokens',type=int,default=1024)
    g=sub.add_parser('corpus');g.add_argument('output');g.add_argument('--count',type=int,default=50000);g.add_argument('--seed',type=int,default=8173);g.add_argument('--tokens',type=int,default=1024);g.add_argument('--workers',type=int,default=4);g.add_argument('--migrate-layout',action='store_true')
    c=sub.add_parser('prepare');c.add_argument('source');c.add_argument('output');c.add_argument('--tokens',type=int,default=2048);c.add_argument('--limit',type=int)
    t=sub.add_parser('train');t.add_argument('cache');t.add_argument('output');t.add_argument('--steps',type=int,default=2000)
    t.add_argument('--device',choices=['auto','cpu','cuda','mps'],default='auto');t.add_argument('--seed',type=int,default=42)
    t.add_argument('--train-limit',type=int,default=0);t.add_argument('--val-limit',type=int,default=100)
    t.add_argument('--eval-every',type=int,default=200);t.add_argument('--lr',type=float,default=3e-4)
    t.add_argument('--accumulate',type=int,default=1);t.add_argument('--no-augmentation',dest='augmentation',action='store_false')
    t.add_argument('--freeze-backbone',action='store_true')
    t.add_argument('--instances',action='store_true');t.add_argument('--init');t.add_argument('--resume');t.add_argument('--width',type=int,default=96)
    t.add_argument('--backbone',choices=['transformer','conv'],default='transformer')
    t.add_argument('--adaptive',action='store_true');t.add_argument('--queries',type=int)
    t.add_argument('--repeat-queries',action='store_true')
    t.add_argument('--relative-noise',action='store_true')
    e=sub.add_parser('evaluate');e.add_argument('checkpoint');e.add_argument('cache');e.add_argument('--split',default='test',choices=['train','val','test','ood','all']);e.add_argument('--device',default='cpu');e.add_argument('--output',required=True)
    i=sub.add_parser('predict');i.add_argument('checkpoint');i.add_argument('contours');i.add_argument('output');i.add_argument('--device',default='cpu');i.add_argument('--tokens',type=int);i.add_argument('--threshold',type=float,default=.7);i.add_argument('--units',choices=['mm','px'],default='mm');i.add_argument('--measure',action='store_true')
    a=sub.add_parser('segment', help='Training-free line/arc/circle recognition')
    a.add_argument('contours');a.add_argument('output');a.add_argument('--units',choices=['mm','px'],default='mm')
    a.add_argument('--tolerance',type=float,help='Maximum deviation in input units (default .01 mm or .75 px)')
    a.add_argument('--no-compounds',dest='compounds',action='store_false')
    m=sub.add_parser('measure-evaluate');m.add_argument('checkpoint');m.add_argument('source');m.add_argument('output');m.add_argument('--split',default='test');m.add_argument('--limit',type=int,default=150);m.add_argument('--tokens',type=int,default=1024)
    b=sub.add_parser('bridge');b.add_argument('source');b.add_argument('output');b.add_argument('--tokens',type=int,default=1024);b.add_argument('--limit',type=int,default=300)
    v=sub.add_parser('review');v.add_argument('checkpoint');v.add_argument('source');v.add_argument('destination');v.add_argument('--split',default='test');v.add_argument('--limit',type=int,default=12);v.add_argument('--tokens',type=int,default=1024)
    args=vars(p.parse_args());cmd=args.pop('command')
    if cmd=='export':
        from .export import export_model
        print(export_model(**args))
    elif cmd=='segment':
        from .geometric import predict_file
        predict_file(**args)
    elif cmd=='measure-evaluate':
        from .validation import selection_report
        selection_report(**args)
    elif cmd=='bridge':
        from .bridge import build_bridge
        print(build_bridge(**args))
    elif cmd=='review':
        from .visualize import review
        review(**args)
    elif cmd=='corpus':
        from .corpus import build_corpus
        build_corpus(**args)
    elif cmd=='prepare':
        from .data import build_cache
        build_cache(**args)
    elif cmd=='train':
        from .train import train
        train(**args)
    elif cmd=='evaluate':
        import torch
        torch.set_num_threads(4)
        from .train import load_model,evaluate,model_device,validate_cache_sampling
        dev=model_device(args['device']);model,ckpt=load_model(args['checkpoint'],dev)
        validate_cache_sampling(model,args['cache'])
        indexed=json.loads((Path(args['cache'])/'index.json').read_text())
        entries=indexed if args['split']=='all' else [e for e in indexed if e['split']==args['split']]
        if model.config.label_schema in ('primitives-compounds-v1','primitives-compounds-v2'):
            from .train_primitives import evaluate as evaluate_primitives
            result=evaluate_primitives(model,entries,args['cache'],dev)
        else:result=evaluate(model,entries,args['cache'],dev,ckpt['instances_trained'])
        Path(args['output']).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    else:
        from .inference import predict_file
        predict_file(**args)

if __name__=='__main__':main()
