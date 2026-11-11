# Expanded part taxonomy — contour corpus v2

The [starter taxonomy](starter-shape-taxonomy.md) supplies the organizing principle:
expand part compositions, patterns and relationships while keeping reusable geometric
feature classes small. The contour corpus now has **47 part families**, including
32 additions to the previous 15-family corpus. These are planar silhouette grammars,
not claims of conformity to a mechanical standard or complete 3D inspection.

The implementation is `inspection_shapes/expanded.py`; corpus construction and
sampling are in `contour_model/corpus_v2.py` and `contour_model/sampling.py`.
The original corpus and released model remain available unchanged.

## Part/view level

| Part group | Added grammars | Deliberate coverage |
|---|---|---|
| Perforated tooling, filters and screens | Perforated grid plate, staggered filter plate, honeycomb screen | Up to 24 × 24 individual openings; circular and hexagonal apertures; rectangular and staggered placement |
| Aperture panels | Mixed aperture panel, polygon broach plate, keyhole panel | Mixed circles, obround slots, triangles/squares/hexagons/octagons, D apertures and keyholes; up to 196 mixed/custom openings |
| Ventilation and channel plates | Slotted grille, labyrinth plate | Up to 240 independent slots; serpentine through channels with non-circular end-to-end profiles |
| Polar pattern parts | Multi-bolt-circle flange, index plate, spoke wheel | Several independently clocked bolt circles, central bores, dense radial patterns and elongated radial openings |
| Angular repeated-profile parts | Internal spline ring, ratchet wheel, timing pulley, serrated washer | Internal and external repetition, asymmetric and rounded teeth, independent inner/outer profiles; up to 192 repeats |
| Freeform and keyed sections | Lobed coupling, eccentric cam, D-profile spacer | Three-to-twelve-lobe profiles, eccentric openings, D/double-D external contours and polygon/D bores |
| Links and asymmetric brackets | Clevis link, dogbone link, L bracket, T bracket, C frame, U-channel section | Unequal lobes, thin connecting webs, re-entrant corners, unequal arms and irregular bore placement |
| Repeated open-edge sections | Comb plate, connector strip, heat-sink section | Up to 120 fingers/open channels, individually grouped notches, narrow lands and mounting-hole rows |
| Complex turned silhouettes | Multi-step shaft, groove-stack pin, multi-thread shaft, stepped-taper arbor | Up to 96 axial stations, 80 grooves per observed side, 2–8 separate threaded zones, and many alternating lands/tapers/shoulders |
| Long repeated linear profiles | Sawtooth rack | 20–200 asymmetric repeats; variable pitch, depth and envelope aspect ratio |

The twelve original generator families and the three prior mixed families remain
represented. The v2 corpus widens stepped shafts, grooved pins, drilled plates,
slotted brackets, flanges, toothed disks, racks and cams through the expanded grammar
branches, while also sampling their simpler previous forms. Washers and other
intrinsically simple families remain simple instead of redefining their meaning.

The implementation initially used a misleading `z_bracket` key for a U-shaped
section. This was corrected to `u_channel_section` in the final corpus metadata;
regeneration checks confirmed that no geometric arrays changed. The correction
receipt is retained with the dataset. There is no claim of a separate Z-bracket
grammar in this release.

## Observed coverage in the generated 50,000 parts

| Coverage property | Count / maximum |
|---|---:|
| Parts with at least 100 through openings | 6,615 |
| Parts with at least 300 through openings | 1,206 |
| Maximum through openings on one part | 576 |
| Maximum feature instances on one part | 584 |
| Maximum instances on the outer contour | 388 |
| Training / validation / test / challenge parts | 40,011 / 3,979 / 3,923 / 2,087 |
| Median / 95th percentile / maximum model intervals | 1,600 / 14,912 / 37,632 |

Actual coverage counts matter more than permissive generator bounds: dense examples
occur throughout the corpus rather than as one showcase part. Split assignment is
stable by design ID. The challenge split combines extreme parameter settings and
severe form errors within the implemented grammars; it is not an unseen-family test.

## Feature classes and measurement handoff

The existing 16 semantic classes are retained. Polygon and rounded-profile classes
now also label whole internal polygon/D/keyhole/custom-rounded openings, in addition
to their previous ancestor use. Teeth remain generic repeated tooth profiles:
ratchets, splines and pulley-like shapes are not asserted to be involute or certified
against a standard. Pitch, lead and helix remain distinct.

Every aperture is an individual feature, even when it belongs to a dense pattern.
Thread instances associate the two observed sides of one threaded zone; separate
zones remain separate instances. Grooves are individual observed-side features.
Comb channels are grouped notches rather than three independent wall instances.
All raw points and nominal reference coordinates are retained. Imperfections do not
change nominal semantic identity or replace observations with ideal fits.

Grid, polar and axial-station pattern metadata are retained where the grammar emits
them. Arbitrary pattern inference, datum selection and tolerances are not supplied
by the semantic model. Handwritten modules can derive relationships and fit geometry
from the resulting instance selections. Generic polygon/rounded-profile/gear-like
routes remain extensible; recognition does not imply that a validated measurement
module already exists for every new shape.

## Revised imperfections

The earlier corpus chose severity 2.5 for 20% of parts, and independently attempted
approximately 0.8 defects per ring. Both choices make heavily perforated parts
unrealistically distressed as a default training population.

The v2 ordinary severity mixture is 5% exact nominal, 20% at 0.03, 40% at 0.08,
25% at 0.16 and 10% at 0.30. The deformation multiplier is additionally capped by
80 mm / part extent for large parts. About 3% of training parts use severe stress
settings between 0.8 and 2.0. Defects have a **per-part** Poisson budget: mean 0.35
for ordinary parts and 2 for stress parts. A dense plate therefore does not acquire
hundreds of burrs merely because it has hundreds of rings.

Lobing, bow, independent surface errors, imperfect parallelism, center offsets,
pitch variation and edge roughness remain present. Ordinary training parts have
a median per-part 95th-percentile nominal-to-observed displacement of **0.0113 mm**;
the 95th percentile across those parts is **0.0448 mm**. A 200-part audit of the
previous corpus gave corresponding values of **0.1033 mm** and **0.3623 mm**.
These distributions include different part populations and are synthetic assumptions,
not established manufacturing tolerances. Real captures should ultimately determine
process-specific form-error and noise distributions.

Ordinary validation/test splits contain mild cases. The challenge split separately
contains 1,034 extreme-complexity cases with mild form errors and 1,053 severe-form
stress cases. Extreme parameters are implemented for selected grammars, not every
family, and are not guaranteed absent from the training range. Report performance
by realized feature count and stress status as well as by split.

## Representation and training changes

The original fixed 1,024-interval release should still use its original sampling
contract. V2 uses geometry-only adaptive per-ring resolution: every small opening
retains its own samples and high-turn outer contours receive more intervals. Raw
feature IDs, nominal primitives and part-family labels do not influence sampling.
The `adaptive-turn-v1` control distributes those intervals uniformly within each
ring. The `adaptive-local-v2` experiment uses the same interval counts, with 35%
uniform arc-length mass and 65% smoothed observed turning mass. This preserves
short lands beside long fins without spending more intervals on every part.
Class probabilities transfer back by physical overlap length; boundary events
transfer by maximum overlapping evidence so dense sampling cannot average away
a single corner. Original point density and feature labels are never used as
semantic inputs, although finite geometric probing is tested for density sensitivity.
The initial operating envelope is 65,536 intervals per part; inputs above it are
rejected instead of silently dropping holes or merging surfaces.

Local attention is batched over circular contour windows. Coarse pooling and
interpolation are vectorized; topology checks use spatial indexing instead of
all-pairs ring comparisons. Closed-loop connectivity supplies aperture instances.
Learned semantic/boundary outputs supply outer surface selections; the focused
query experiment learns association of thread/tooth profiles. It does not require
one learned query for every bore, land or shoulder.

GPU smoke tests completed training steps on a 576-hole plate, a 576-hexagon screen,
and a 96-station shaft. The largest observed tensor-memory peak was below 500 MB
for the tested 64-wide model, excluding CUDA context/allocator reservation. Full
training compares joint repeat-association learning against a semantic-only control
initialized from the same adaptive pilot weights.

CPU generation and raster rendering remain independent of CUDA. GPU use is confined
to model training/evaluation. A second augmentation control limits coordinate noise
relative to each ring's sample spacing; it does not change the stored physical
imperfections or their nominal-to-observed displacement distribution.

See `packages/contour-model/VALIDATION_V2.md` for final trained-model results. Source snapshots and dataset configuration hashes distinguish the
frozen generation version from subsequent input-validation fixes.
