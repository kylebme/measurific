"""A single forward camera transform is shared by image and label generation."""
from dataclasses import dataclass, asdict
from time import perf_counter
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy.ndimage import gaussian_filter, distance_transform_edt
from .taxonomy import SEMANTICS


def rasterize_coverage(pixels, holes, width, height, supersample):
    """Integrate polygon area in each subpixel, without integer vertex snapping.

    Pillow fills interiors quickly; only cells intersecting a boundary need
    geometric clipping. Subpixel centers follow the same convention as project().
    """
    import shapely
    from shapely.geometry import Polygon, GeometryCollection

    ss = supersample
    material = Image.new("L", (width*ss, height*ss), 0)
    boundary = Image.new("L", material.size, 0)
    fill, edges = ImageDraw.Draw(material), ImageDraw.Draw(boundary)
    geometry = GeometryCollection()
    for uv, hole in zip(pixels, holes):
        high = uv*ss+(ss-1)/2
        points = [tuple(x) for x in high]
        fill.polygon(points, fill=0 if hole else 255)
        # The dilated boundary band below covers cells touched by the continuous
        # boundary, including Pillow's truncation and inclusive endpoint rules.
        edges.line(points+[points[0]], fill=255, width=1)
        polygon = Polygon(high).simplify(1e-9, preserve_topology=True)
        geometry = geometry.difference(polygon) if hole else geometry.union(polygon)
    coverage = np.asarray(material, dtype=np.float32)/255
    y, x = np.nonzero(np.asarray(boundary.filter(ImageFilter.MaxFilter(5))))
    # Bound temporary geometry allocation for large sensor images.
    for start in range(0, len(x), 16384):
        xx, yy = x[start:start+16384], y[start:start+16384]
        cells = shapely.box(xx-.5, yy-.5, xx+.5, yy+.5)
        coverage[yy, xx] = shapely.area(shapely.intersection(cells, geometry))
    return coverage

@dataclass
class Camera:
    width: int = 1920
    height: int = 1080
    supersample: int = 2
    band_px: float = 3.0
    rotation: float = 0.
    pixels_per_mm: float = 15.
    translation_x: float = 960.
    translation_y: float = 540.
    k1: float = 0.
    k2: float = 0.
    p1: float = 0.
    p2: float = 0.
    sigma_px: float = 0.8
    dark: float = 12.
    bright: float = 235.
    read_noise_dn: float = 1.5
    electrons_per_dn: float = 40.
    prnu: float = 0.003
    gradient_x: float = 0.
    gradient_y: float = 0.
    vignette: float = 0.

    def project(self, xy):
        c,s=np.cos(self.rotation),np.sin(self.rotation)
        # world +y points up; image +y points down, pixel centers are integers.
        uv=xy @ np.array([[c,-s],[-s,-c]]).T * self.pixels_per_mm
        uv+=np.array([self.translation_x,self.translation_y])
        center=np.array([(self.width-1)/2,(self.height-1)/2])
        scale=max(self.width,self.height)/2
        x,y=((uv-center)/scale).T
        r2=x*x+y*y; radial=1+self.k1*r2+self.k2*r2*r2
        return np.c_[x*radial+2*self.p1*x*y+self.p2*(r2+2*x*x),
                     y*radial+self.p1*(r2+2*y*y)+2*self.p2*x*y]*scale+center


def random_camera(part, rng, width=1920, height=1080, supersample=2):
    angle=float(rng.uniform(-np.pi,np.pi))
    c,s=np.cos(angle),np.sin(angle)
    q=np.concatenate([r.actual for r in part.rings]) @ np.array([[c,-s],[-s,-c]]).T
    extent=np.ptp(q,axis=0);middle=(q.max(axis=0)+q.min(axis=0))/2
    ppm=float(min(width/extent[0],height/extent[1])*rng.uniform(.66,.87))
    return Camera(width,height,supersample,3.,angle,ppm,
        float((width-1)/2-middle[0]*ppm+rng.uniform(-.02,.02)*width),
        float((height-1)/2-middle[1]*ppm+rng.uniform(-.02,.02)*height),
        k1=float(rng.uniform(-.025,.025)),k2=float(rng.uniform(-.008,.008)),
        p1=float(rng.uniform(-.001,.001)),p2=float(rng.uniform(-.001,.001)),
        sigma_px=float(rng.uniform(.45,1.55)),dark=float(rng.uniform(3,30)),
        bright=float(rng.uniform(208,248)),read_noise_dn=float(rng.uniform(.25,2.5)),
        electrons_per_dn=float(rng.uniform(25,150)),prnu=float(rng.uniform(0,.006)),
        gradient_x=float(rng.uniform(-.05,.05)),gradient_y=float(rng.uniform(-.05,.05)),
        vignette=float(rng.uniform(0,.07)))


def feature_bits(part):
    bits=np.zeros(len(part.features)+1,np.uint32)
    for f in part.features:
        bits[f["id"]]=np.uint32(1 << f["semantic_id"]) | bits[f["parent"]]
    return bits


def draw_runs(draw, points, labels, width=1):
    """Draw connected equal-label edge runs, including each run's end vertex."""
    cuts=np.r_[0,np.flatnonzero(labels[1:]!=labels[:-1])+1,len(labels)]
    closed=np.vstack([points,points[0]])
    for a,b in zip(cuts[:-1],cuts[1:]):
        draw.line([tuple(x) for x in closed[a:b+1]],fill=int(labels[a]),width=width)


def render(part, camera, seed):
    start=perf_counter(); w,h,ss=camera.width,camera.height,camera.supersample
    pixels=[camera.project(r.actual) for r in part.rings]
    allpixels=np.concatenate(pixels)
    if (allpixels.min(axis=0)<4).any() or (allpixels.max(axis=0)>[w-5,h-5]).any():
        raise ValueError("Part clipped by camera")
    coverage=rasterize_coverage(pixels, [ring.hole for ring in part.rings], w, h, ss)
    raster_done=perf_counter()
    blurred=gaussian_filter(coverage,camera.sigma_px*ss,mode="nearest")
    # Integrate irradiance across each finite square pixel after the PSF.
    blurred=blurred.reshape(h,ss,w,ss).mean(axis=(1,3))
    occupancy=coverage.reshape(h,ss,w,ss).mean(axis=(1,3))
    blur_done=perf_counter()
    rng=np.random.default_rng(seed)
    xx=np.linspace(-1,1,w,dtype=np.float32)[None,:]; yy=np.linspace(-1,1,h,dtype=np.float32)[:,None]
    background=camera.bright*(1+camera.gradient_x*xx+camera.gradient_y*yy-camera.vignette*(xx*xx+yy*yy))
    expected=(background*(1-blurred)+camera.dark*blurred)
    expected*=rng.normal(1,camera.prnu,(h,w)).astype(np.float32)
    image=rng.poisson(np.maximum(expected,0)*camera.electrons_per_dn).astype(np.float32)/camera.electrons_per_dn
    image+=rng.normal(0,camera.read_noise_dn,(h,w)).astype(np.float32)
    image=np.rint(image).clip(0,255).astype(np.uint8)
    optics_done=perf_counter()
    region=Image.new("I",(w,h),0);d=ImageDraw.Draw(region)
    # A filled object/hole region is useful for closed features; local open
    # features use boundary bands, never an invented filled interior.
    outer_feature=part.features[int(part.rings[0].feature[0])-1]
    while outer_feature["parent"]:outer_feature=part.features[outer_feature["parent"]-1]
    d.polygon([tuple(x) for x in pixels[0]],fill=outer_feature["id"])
    for ring,uv in zip(part.rings[1:],pixels[1:]):
        d.polygon([tuple(x) for x in uv],fill=int(ring.feature[0]))
    region=np.asarray(region,dtype=np.uint16)
    seeds=[]
    for key in ("feature","primitive","defect","anomaly_instance"):
        im=Image.new("I",(w,h),0);d=ImageDraw.Draw(im)
        for ring,uv in zip(part.rings,pixels):draw_runs(d,uv,getattr(ring,key))
        seeds.append(np.asarray(im,dtype=np.int32))
    distance,indices=distance_transform_edt(seeds[0]==0,return_indices=True)
    band=distance<=camera.band_px
    boundary=[np.where(band,x[tuple(indices)],0) for x in seeds]
    lookup=np.array([0]+[f["semantic_id"] for f in part.features],np.uint8)
    bits=feature_bits(part)
    semantic_bits=bits[region] | bits[boundary[0]]
    output={"image":image,"material":(occupancy>=.5).astype(np.uint8)*255,
        "region_instance":region,"region_semantic":lookup[region],
        "boundary_instance":boundary[0].astype(np.uint16),
        "boundary_semantic":lookup[boundary[0]],"boundary_primitive":boundary[1].astype(np.uint16),
        "anomaly":boundary[2].astype(np.uint8),"anomaly_instance":boundary[3].astype(np.uint16),
        "semantic_bits":semantic_bits}
    timing=dict(raster= raster_done-start,blur_integrate=blur_done-raster_done,
                sensor=optics_done-blur_done,labels=perf_counter()-optics_done)
    return output,pixels,timing
