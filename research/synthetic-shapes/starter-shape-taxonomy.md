# Recommended 2D Silhouette Feature Taxonomy

## 1. Core design principle

Keep the **geometric feature taxonomy small**.

Do not create a unique low-level feature type for every recognizable mechanical shape. Instead, separate the system into:

1. **Geometric primitives**
2. **Compound geometric features**
3. **Semantic classifications**
4. **Patterns**
5. **Relationships**
6. **Measurement and tolerance operators**
7. **Standard-specific templates**

This allows the system to recognize unfamiliar geometry, measure it generically, and later assign a more specific mechanical interpretation when confidence is sufficient.

---

# 2. Geometric primitives

These should be the lowest-level contour representations.

| Primitive                   | Typical parameters and measurements                                               |
| --------------------------- | --------------------------------------------------------------------------------- |
| **Line segment**            | position, direction, length, straightness/profile residual                        |
| **Circular arc**            | center, radius, start/end angle, arc length, radial/profile residual              |
| **Elliptical arc**          | center, major/minor axes, orientation, angular span, profile residual             |
| **Freeform curve / spline** | control representation, arc length, curvature, normal deviation                   |
| **Junction / corner**       | intersection position, included angle, tangent continuity, virtual-sharp position |

A complete circle can be represented as a closed circular arc.

Fillets, chamfers, slot ends, thread roots, and similar details should usually be represented by these primitives plus a semantic role, rather than by unique geometric classes.

---

# 3. Compound geometric features

These are the main reusable geometric structures that should sit above the primitives.

| Feature class                    | Typical examples                                                           |
| -------------------------------- | -------------------------------------------------------------------------- |
| **Circle**                       | ID hole, OD circle, circular boss, circular opening                        |
| **Width feature**                | two parallel edges, shaft width in side silhouette, slot width, land width |
| **Angle feature**                | intersecting flanks, V-groove, wedge, chamfer angle                        |
| **Closed polygon**               | triangle, rectangle, square, hexagon, octagon                              |
| **Closed rounded profile**       | obround, rounded rectangle, D-profile, double-D                            |
| **Groove / notch**               | rectangular groove, U-groove, V-groove, dovetail, undercut                 |
| **Step / shoulder**              | shaft shoulder, stepped plate edge, height transition                      |
| **Taper**                        | conical side silhouette, wedge, linearly changing width                    |
| **Repeated profile**             | threads, gears, splines, serrations, corrugations                          |
| **Generic closed contour**       | custom through-hole or arbitrary external shape                            |
| **Generic open contour segment** | local spline, cam segment, custom edge                                     |

These classes should be parameterized rather than split into many narrowly defined subclasses.

For example:

```text
ClosedPolygon
    side_count = 6
    regularity = high
    opposite_sides_parallel = true
```

can be semantically reported as a **hexagon**.

---

# 4. Groove and notch model

Use one general `Groove` feature instead of separate geometric engines for rectangular grooves, U-grooves, V-grooves, dovetails, reliefs, and undercuts.

Recommended parameters:

```text
opening_width
depth
bottom_width
left_flank_angle
right_flank_angle
bottom_geometry
left_transition
right_transition
orientation
symmetry
```

Semantic classification can then infer:

* rectangular groove
* U-groove
* V-groove
* dovetail
* asymmetric groove
* undercut
* machining relief

This greatly reduces taxonomy complexity while retaining useful user-facing terminology.

---

# 5. Repeated profiles

Threads, gears, splines, serrations, sprockets, ratchets, and similar features should share a common underlying representation.

## Base feature

```text
RepeatedProfile
    domain = linear | angular
    count
    pitch
    phase / clocking
    mean_profile
    individual_profiles[]
    local_pitch_errors[]
    cumulative_pitch_error
    envelope_geometry
```

## Semantic subclasses

```text
RepeatedProfile
├── Thread
├── GearTeeth
├── Spline
├── Serration
├── Sprocket
├── Ratchet
└── GenericRepeatedProfile
```

This allows the system to measure an unfamiliar repeated profile even when it cannot confidently identify the standard.

For example:

```text
Repeated asymmetric profile
Pitch: 2.000 mm
Left flank angle: 7.1°
Right flank angle: 45.0°
Major diameter: 24.998 mm
Likely buttress thread
Classification confidence: 96%
```

is preferable to either forcing an incorrect classification or returning no result.

---

# 6. Thread classifications

Thread standards should be implemented as templates applied to the generic repeated-profile model.

Recommended initial thread classes:

* ISO metric V thread
* Unified UN / UNR / UNJ
* Acme
* Stub Acme
* ISO trapezoidal
* Buttress
* Square thread
* Round / knuckle thread
* Tapered pipe thread
* Parallel pipe thread
* Custom / unidentified thread

Potential thread measurements include:

* pitch
* TPI
* local pitch
* cumulative pitch error
* major diameter
* minor diameter
* estimated pitch diameter where appropriate
* thread depth
* left flank angle
* right flank angle
* included angle
* crest width
* root width
* crest radius
* root radius
* crest truncation
* root truncation
* flank straightness/profile
* tooth/thread thickness
* space width
* visible thread count
* threaded length
* taper
* lead-in geometry
* runout geometry
* profile deviation from a standard template

Pitch, lead, and helix should remain distinct quantities. A single static silhouette generally supports pitch much more directly than true lead or helix error.

---

# 7. Gear, spline, and tooth-profile classifications

Use the same `RepeatedProfile` engine with application-specific semantic templates.

Recommended classes:

* rack gear
* external spur gear
* internal spur gear
* external involute spline
* internal involute spline
* straight-sided spline
* serration
* sprocket
* synchronous-belt pulley
* ratchet / sawtooth
* generic repeated tooth profile

Typical measurements include:

* count
* pitch
* angular pitch
* module or diametral-pitch inference where applicable
* major/tip diameter
* minor/root diameter
* pressure angle
* flank angles
* tooth thickness
* space width
* tooth height
* root radius
* tip geometry
* individual pitch error
* cumulative pitch error
* indexing error
* radial eccentricity
* profile deviation

A 2D silhouette should be described as **tooth-profile inspection**, not complete gear inspection. Lead, helix, full flank topology, and other 3D gear characteristics require additional measurement capability.

---

# 8. Semantic classifications

Semantic classifications should sit above geometry rather than replace it.

Examples include:

* hole
* external circular feature
* slot
* obround
* rectangle
* square
* hexagon
* D-profile
* double-D
* fillet
* chamfer
* shoulder
* groove
* thread
* gear
* spline
* serration
* lug
* tab
* cam profile

A single geometric feature may receive multiple semantic labels with confidence scores.

For example:

```text
geometry_type = CircularArc
role = Fillet
parent_feature = Shoulder
```

or:

```text
geometry_type = Line
role = ThreadFlank
parent_feature = ExternalThread
```

This keeps geometry stable while allowing higher-level interpretation to evolve.

---

# 9. Patterns

Patterns should be represented as groups of already-detected features rather than as separate feature types.

Recommended pattern classes:

```text
Pattern
├── Linear
├── RectangularGrid
├── Polar
└── ArbitraryRepeated
```

Typical pattern measurements:

* feature count
* nominal pitch
* mean pitch
* local spacing error
* cumulative spacing error
* pattern center
* pattern orientation
* angular spacing
* radial error
* tangential error
* row/column straightness
* grid orthogonality
* member-to-member size variation

Examples include:

* bolt circles
* linear hole arrays
* slot arrays
* rectangular hole grids
* radial slots
* repeated notches

A circle should remain an individual circle feature even when it also belongs to a bolt-circle pattern.

---

# 10. Relationships

Relationships should be first-class objects independent of the underlying feature types.

Recommended relationship operators:

| Relationship                      | Examples                                       |
| --------------------------------- | ---------------------------------------------- |
| **Distance**                      | center-to-center, edge-to-edge, center-to-edge |
| **Angle**                         | line-to-line angle, feature orientation        |
| **Parallelism**                   | two lines, two width-feature sides             |
| **Perpendicularity**              | shoulder to axis, adjacent rectangle sides     |
| **Concentricity / center offset** | bore to OD, nested circular profiles           |
| **Tangency**                      | line-to-arc, arc-to-arc                        |
| **Symmetry**                      | paired geometry relative to a centerline       |
| **Minimum separation**            | wall thickness, clearance                      |
| **Common axis / centerline**      | alignment of multiple features                 |
| **Pattern membership**            | repeated feature relationships                 |

Examples:

```text
CenterDistance(CircleA, CircleB)
Parallelism(LineA, LineB)
MinimumDistance(ContourA, ContourB)
Tangency(LineA, ArcB)
```

This avoids creating highly specific feature names such as `hole_to_edge_distance`.

---

# 11. Measurement and tolerance operators

Measurements and tolerances should be separate from feature classification.

A circle is a circle regardless of whether the user wants least-squares diameter, maximum-inscribed diameter, circularity, or position.

## Size operators

Recommended:

* local two-point size
* minimum local size
* maximum local size
* least-squares associated size
* maximum-inscribed size
* minimum-circumscribed size
* envelope size
* overall extent

Avoid exposing one ambiguous generic `diameter` when multiple size definitions may be relevant.

## Form operators

* straightness
* circularity
* profile of a line
* arc-form deviation
* minimum-zone form error
* least-squares residual
* maximum inward profile deviation
* maximum outward profile deviation

For partial arcs, use **arc form/profile error**, rather than calling it roundness.

## Orientation operators

* parallelism
* perpendicularity
* angularity
* absolute angle
* relative angle

## Location operators

* X coordinate
* Y coordinate
* center position
* center-to-center distance
* edge distance
* pattern position
* radial position
* tangential position
* symmetry error
* datum-relative profile
* 2D true-position evaluation where the required nominal and datum information exists

## Profile operators

* maximum normal deviation
* minimum-zone profile
* inward deviation
* outward deviation
* peak-to-valley error
* RMS deviation
* Hausdorff distance
* deviation as a function of contour position

Not all of these are standardized GD&T characteristics; some are useful engineering and diagnostic metrics.

## Repetition operators

* pitch
* local pitch
* mean pitch
* cumulative pitch error
* angular pitch
* indexing error
* tooth/member-to-member variation

## Derived operators

* minimum wall thickness
* maximum wall thickness
* wall-thickness distribution
* minimum clearance
* eccentricity
* virtual-sharp position
* chord length
* sagitta
* arc length
* area
* perimeter
* centroid

---

# 12. Universal profile comparison

Every feature should support comparison against nominal geometry.

Recommended representation:

```text
MeasuredContour
NominalContour
ProfileDeviation(MeasuredContour, NominalContour)
```

Possible outputs include:

* maximum outward error
* maximum inward error
* peak-to-valley error
* minimum-zone error
* RMS error
* error versus contour parameter
* location of worst deviation

This becomes the universal fallback for:

* splines
* cams
* molded contours
* custom broached profiles
* nonstandard threads
* custom gear-like teeth
* arbitrary CAD curves

It also reduces the need to define specialized tolerance logic for every possible feature.

---

# 13. Feature hierarchy and ownership

Detected primitives should belong to higher-level features where appropriate.

For example:

```text
Straight Slot
├── upper side line
├── lower side line
├── left end arc
└── right end arc
```

The user-facing measurement should normally describe the slot:

```text
Slot width
Overall length
Orientation
Position
Side parallelism
End radius
Profile deviation
```

The constituent lines and arcs should appear only at greater verbosity.

This avoids producing a report containing dozens of low-level primitives instead of meaningful manufactured features.

---

# 14. Recommended verbosity hierarchy

Use four reporting levels.

## Level 1 — major semantic features

Examples:

* external profile
* four through holes
* rectangular slot
* M12 × 1.75 external thread
* 10 mm hex
* annular groove

## Level 2 — primary measurements

Examples:

```text
Slot
    width
    length
    position
    orientation

Thread
    pitch
    major diameter
    minor diameter
    flank angle
```

## Level 3 — detailed feature geometry

Examples:

```text
Slot
    side straightness
    side parallelism
    individual end radii
    end-center spacing
    tangent-transition error
```

## Level 4 — minor details and anomalies

Examples:

* R0.18 corner fillet
* 0.07 mm edge break
* 0.021 mm burr
* small local notch
* local contour waviness
* thread lead-in details

This is preferable to manually assigning arbitrary importance values to hundreds of possible feature names.

---

# 15. Recommended anomaly classes

Manufacturing anomalies should be detected separately from nominal feature geometry.

Useful classes include:

* burr
* flash
* nick
* chipped edge
* missing material
* excess material
* local notch
* cusp
* unexpected radius
* unexpected chamfer
* local profile excursion

Potential measurements include:

* maximum depth or protrusion
* width
* area
* contour length
* position
* inward/outward deviation

These should normally appear at lower verbosity unless they exceed a tolerance.

---

# 16. Characteristics that should not be claimed from one static silhouette

The system should explicitly distinguish between quantities directly supported by the image and characteristics requiring additional dimensional information.

A single 2D backlit projection generally cannot fully establish:

* surface flatness
* cylindricity
* full 3D axis straightness
* out-of-plane parallelism
* out-of-plane perpendicularity
* full bore-axis position
* full coaxiality
* circular runout
* total runout
* thread helix error
* true thread lead in many cases
* gear lead/helix deviation
* full 3D gear flank form
* blind-hole depth
* counterbore depth
* countersink depth
* pocket geometry hidden from the silhouette
* conventional surface roughness such as Ra or Rz

If a rotary stage, multiple cameras, focus-based depth measurement, or other sensors are later added, some of these can become supported.

---

# 17. Recommended internal object model

A detected feature should ideally contain:

```text
Feature
    id
    geometry
    semantic_classification
    classification_confidence
    constituent_primitives[]
    parent_feature
    child_features[]
    nominal_geometry
    measured_geometry
    supported_measurements[]
    relationships[]
    pattern_memberships[]
    standard_match
    standard_match_confidence
    visibility / observability metadata
```

A measurement result should be independent:

```text
Measurement
    operator
    feature_refs[]
    datum_refs[]
    nominal_value
    measured_value
    deviation
    tolerance
    pass_fail
    fitting_method
    uncertainty
```

This distinction is important because the same detected geometry may support many different measurements.

---

# 18. Recommended minimal taxonomy

A practical first implementation can stay very small:

```text
Primitive
├── Line
├── CircularArc
├── EllipticalArc
└── FreeformCurve

CompoundFeature
├── Circle
├── Width
├── Angle
├── ClosedPolygon
├── ClosedRoundedProfile
├── Groove
├── Step
├── Taper
├── RepeatedProfile
└── GenericContour

Pattern
├── Linear
├── Grid
├── Polar
└── ArbitraryRepeated

Relationship
├── Distance
├── Angle
├── Parallel
├── Perpendicular
├── CommonCenter
├── Tangent
├── Symmetric
└── MinimumSeparation
```

Everything more specific can initially be implemented as semantic classification layered onto this model.

---

# Recommendations

## 1. Keep geometry and semantics separate

A line should remain a line even when it is recognized as:

* a chamfer
* a thread flank
* a groove wall
* a datum candidate
* a polygon side

Store the semantic role separately.

## 2. Keep tolerance evaluation separate from feature detection

Feature recognition should answer:

> What geometry appears to exist?

Measurement operators should answer:

> What quantitative characteristics can be extracted from it?

Tolerance evaluation should answer:

> Does this measurement satisfy the specified requirement?

These are separate problems and should remain separate in the architecture.

## 3. Make generic measurement possible before classification

The system should still measure an unidentified repeated profile, unknown polygon, or arbitrary contour.

Recognition should add value, not act as a gate that prevents measurement.

## 4. Use standard templates rather than standard-specific detectors

Recognize the generic geometry first.

Then evaluate possible matches against:

* ISO metric threads
* Unified threads
* Acme
* trapezoidal threads
* buttress threads
* involute splines
* spur gears
* other supported standards

This makes standard support modular.

## 5. Treat profile comparison as a core capability

Any contour should be measurable against nominal CAD geometry.

This provides a universal fallback and makes unusual parts inspectable without building custom feature types.

## 6. Make relationships first-class

A significant fraction of useful inspection results involve two or more features.

Examples include:

* hole position relative to edges
* center distances
* concentricity/eccentricity
* parallelism
* perpendicularity
* minimum wall thickness
* pattern spacing

Do not bury these inside individual feature classes.

## 7. Represent repeated geometry once

Use a common repeated-profile engine for:

* threads
* gears
* splines
* serrations
* sprockets
* ratchets

Then attach domain-specific interpretation.

## 8. Use hierarchical reporting

The inspection report should emphasize semantic features first and expose constituent geometry only when needed.

A typical output should look like:

```text
External thread
    M10 × 1.5
    major diameter
    minor diameter
    pitch
    flank angle

Annular groove
    width
    depth
    root diameter
    position

Through slot
    width
    length
    position
    orientation
```

rather than listing dozens of lines, arcs, and tiny radii at equal priority.

## 9. Attach observability to every measurement

Every measurement operator should carry information such as:

```text
observable_from_single_silhouette = true
requires_rotation = false
requires_multiple_views = false
requires_nominal_geometry = true
requires_datum_definition = true
```

This prevents the system from presenting mathematically computable 2D quantities as equivalent to full 3D GD&T verification.

## 10. Preserve the raw contour

Never reduce the image permanently to fitted primitives.

Retain:

```text
raw calibrated contour
↓
segmented contour
↓
fitted primitives
↓
compound features
↓
semantic interpretation
```

That allows improved fitting, standards, and feature recognition algorithms to be applied later without reacquiring the part.

## 11. Report confidence separately from measurement uncertainty

These are different concepts.

For example:

```text
Classification:
    ISO metric thread
    confidence = 98%

Measurement:
    pitch = 1.5002 mm
    estimated uncertainty = ±0.0010 mm
```

A classification can be uncertain while its geometric measurements are precise, or vice versa.

## 12. Prefer a small extensible ontology over an exhaustive feature list

The system should not require a new geometric feature class every time a customer presents a new mechanical profile.

The preferred flow is:

```text
generic geometry
→ compound structure
→ semantic interpretation
→ standard match
→ applicable measurements
→ tolerance evaluation
```

That architecture should cover common defined features while remaining robust to arbitrary silhouette geometry.
