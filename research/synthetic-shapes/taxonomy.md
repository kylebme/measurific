# Implemented taxonomy, version 1

The expanded contour corpus is documented in [expanded-part-taxonomy.md](expanded-part-taxonomy.md).
This document retains the original rendered-dataset contract.

The original starter taxonomy remains the broader roadmap. This implementation
adds an explicit part/view level and keeps form errors, defects and measurement
operators separate from nominal semantic identity.

```text
Part
├── Turned / side projection
│   ├── Stepped shaft
│   ├── Threaded shaft
│   ├── Grooved pin
│   └── Tapered pin
├── Turned or cut / end projection
│   ├── Washer
│   └── Flange (bore plus polar hole pattern)
├── Prismatic / face projection
│   ├── Drilled plate (linear/grid hole pattern)
│   ├── Slotted bracket (optional tabs and notches)
│   └── Polygon spacer
└── Profile cut / face projection
    ├── Toothed disk
    ├── Rack
    └── Cam

Part instance → semantic feature instance → nominal primitive instance → sampled edges
             ↘ relationships / feature patterns
Sampled edges → imperfect contours → optional defect instances
```

The machine-readable registry is `inspection_shapes/taxonomy.py`, copied into each
dataset as `taxonomy.json`. Every feature has a semantic class, compound geometry
class, parent instance, nominal parameters, and candidate measurement operators.
Relationships refer to feature or primitive instances rather than introducing
classes such as "hole-to-edge-distance".

| ID | Semantic class | Typical segmentation support |
|---:|---|---|
| 0 | Background/unlabeled | No feature |
| 1 | External profile | Whole material region and ancestor of boundary features |
| 2 | Hole | Closed bore boundary and empty interior |
| 3 | Slot | Side/end boundary and empty interior |
| 4 | Flat | Surface boundary |
| 5 | Chamfer | Boundary edge |
| 6 | Fillet | Arc boundary |
| 7 | Shoulder | Height-transition boundary |
| 8 | Groove | Consecutive groove surfaces |
| 9 | Taper | Tapered surface boundary |
| 10 | Thread | Complete repeated profile, shared across both visible sides |
| 11 | Tooth profile | Complete linear/angular repeated profile |
| 12 | Polygon | Polygon boundary hierarchy/region |
| 13 | Rounded profile | Rounded plate boundary hierarchy |
| 14 | Cam profile | Cam boundary and region |
| 15 | Tab | Protruding bracket surfaces |
| 16 | Notch | Re-entrant bracket surfaces |

Append IDs when extending the taxonomy; never reinterpret existing IDs. Primitive
kinds are line, circular arc, and sampled freeform curve. Junctions are shared
primitive endpoints. Elliptical arcs and more complex imported curves are future
extensions, not mislabeled supported analytic primitives.

Defects use their own IDs (`none=0`, `burr=1`, `nick=2`). Roughness, lobing, bow,
shear, center offsets, and pitch errors are geometric deviations, not competing
semantic classes. A defective hole remains a hole. Severe cases are topology
checked but are not assigned tolerance pass/fail labels.

Patterns are first-class linear/grid/polar groups of features. Current generators
emit grid and polar patterns. Relationships currently include symmetry, center
offset, parallelism, and perpendicularity in the observed plane. Broader operators
from the starter taxonomy can be added without changing the segmentation classes.

Thread shape style is metadata, not a separate class per standard. Teeth are
labeled generic tooth profiles; none are asserted to be involute gears. The planar
projection does not establish thread lead, helix, cylindrical roundness from side
views, full axis alignment, or hidden surface geometry.
