import argparse
import json
from pathlib import Path
from .dataset import write_dataset,preview,iter_samples,digest

def main():
    parser=argparse.ArgumentParser(description="Contour-first synthetic inspection datasets")
    sub=parser.add_subparsers(dest="command",required=True)
    g=sub.add_parser("generate")
    g.add_argument("--output",type=Path,required=True)
    g.add_argument("--count",type=int,default=1200)
    g.add_argument("--seed",type=int,default=20260905)
    g.add_argument("--workers",type=int,default=4)
    g.add_argument("--shard-size",type=int,default=100)
    g.add_argument("--width",type=int,default=1920);g.add_argument("--height",type=int,default=1080)
    g.add_argument("--supersample",type=int,default=2)
    g.add_argument("--views-per-design",type=int,default=1)
    g.add_argument("--contours-only",action="store_true")
    p=sub.add_parser("preview");p.add_argument("output",type=Path);p.add_argument("destination",type=Path);p.add_argument("--limit",type=int,default=24)
    v=sub.add_parser("validate");v.add_argument("output",type=Path)
    args=vars(parser.parse_args());command=args.pop("command")
    if command=="generate":write_dataset(**args)
    elif command=="preview":preview(**args)
    else:
        from .validate import validate_dataset
        print(json.dumps(validate_dataset(args["output"]),indent=2))

if __name__=="__main__":main()
