"""Measure CPU rendering (PNG compression and dataset I/O excluded)."""
import json
import time
import numpy as np
from inspection_shapes.geometry import generate
from inspection_shapes.render import random_camera,render


def main():
    part=generate('threaded_shaft',123)
    camera=random_camera(part,np.random.default_rng(42))
    timings=[]
    for _ in range(4):
        start=time.perf_counter()
        _,_,stages=render(part,camera,55)
        timings.append(time.perf_counter()-start)
    print(json.dumps(dict(cpu_seconds=float(np.median(timings[1:])),last_stage_seconds=stages),indent=2))

if __name__=='__main__':main()
