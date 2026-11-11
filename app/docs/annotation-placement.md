# Mechanical annotation placement

The GUI uses one layout for distance dimensions, included angles, and planar GD&T.
The Python measurement results are authoritative; layout only changes presentation.

## Reference and implementation

This design references Draftwright's
[boundary-labeling research](https://github.com/pzfreo/draftwright/blob/main/docs/research/annotation-placement-boundary-labeling.md)
and its current
[layout implementation](https://github.com/pzfreo/draftwright/blob/main/src/draftwright/layout.py),
reviewed September 12, 2026. The research recommends collecting all annotations
before solving their shared occupancy. The current code uses median PAVA for
minimum-displacement strip placement and bounded assignment of leader alternatives.
Our TypeScript implementation is independent; no Draftwright dependency is added.

`src/annotationLayout.ts` implements three phases:

1. **Collect:** create aligned distance candidates, true-intersection angle arcs,
   and boundary callouts. Read unrotated label dimensions from the rendered DOM.
2. **Solve:** order boundary labels by feature position and use variable-width L1
   PAVA to respect gaps and viewport bounds. Compare complete alternatives with a
   deterministic beam of at most 24 states, including explicit unplaced outcomes.
   Distances are visited from shortest to longest, so enclosing dimensions can
   occupy outer lanes. All annotation types share the same occupied geometry.
3. **Render:** emit exactly the chosen label, witness, extension, arrow and arc
   geometry. No separate annotation renderer subsequently moves a label.

Label overlap, label/line interference, labels over contour edges, viewport
clipping of labels, pinned cards, and crossing fallback leaders are hard conflicts.
Extension-line crossings between actual dimensions are penalized along with
unnecessary line length. Conservative rotated label bounds are used between
annotations; contour clearance uses the actual rotated rectangle.

The boundary strip subproblem minimizes total absolute displacement. The full
candidate assignment is a bounded search, **not** a globally optimal CAD layout.
If its available alternatives cannot fit a label, a notice identifies the count
and lists the affected values in its tooltip. Results remain in the inspector;
hiding other annotations, panning, or resetting the view can free space.

## Geometry rules

- Distances retain the exact selected statistic's witness pair. Extension lines
  are perpendicular to that pair and terminate at the dimension line with a small
  overshoot. Their offset follows contour segments clipped to the measured span,
  avoiding diagonal projections of unrelated empty bounding-box corners.
- Short spans use outward arrows and put the value beyond an endpoint. Coincident
  or subpixel witness pairs use a direct callout instead of an invented axis.
- Angle arcs use the actual intersection and acute sector of the two fitted lines.
  Either equivalent sector and several radii may be considered. The value sits
  outside the arc with a short connector so the text cannot swallow the arc.
- Parallel lines, very small angles, and intersections outside the viewport use an
  angle-marked callout to both fitted line centers. No replacement vertex is drawn.
- Annotation sizes stay constant in screen pixels. Geometry follows the same
  contain/pan/zoom transform as the image. Deformation display can affect contour
  keepouts, but does not alter measured distance witnesses or fitted angle lines.

`MeasurementAnnotations.tsx` owns synchronous DOM measurement and rendering. A
passive resize observer handles the initial parent-ref attachment and later size
changes. Layout input caching avoids solving again on unrelated pointer movement.
Pinned feature cards reserve space; transient hover tooltips remain transient UI.

Labels use intrinsic (`max-content`) widths so their measured size cannot depend
on their solved `left` position. Shrink-to-fit sizing previously oscillated between
117 and 118 pixels near the right edge, repeatedly triggering the layout effect
until React aborted rendering with maximum update depth exceeded.

## Verification

Run `npm test` and `npm run build` in `app`. Tests include a brute-force oracle for
small strip problems, diagonal and reversed witnesses, short/zero distances,
true/distant intersections, mixed occupancy, pinned obstacles, long labels,
viewport exhaustion, and pan/zoom transforms.

For visual checks, start Vite (`npm exec vite -- --host 127.0.0.1`) and open
`/annotation-preview.html`. This development-only entry provides a mechanical
plate, mixed annotations, diagonal/short/zero distances, distant intersections,
and a crowded fixture, plus hide/restore, zoom, pinned-card, and narrow-view
controls. It is not included in the production app entry.

The **Drilled plate angle regression** fixture retains the fitted centers,
directions, and observed support endpoints for `base_00003` and `base_00005` from
the demo worker. It pins both feature cards and fixes the canvas at 1366×951 CSS
pixels, a size that reproduced the crash before the intrinsic-width fix. The page
automatically checks that the angle label settles inside the canvas with readable
text and reports pass/failure. Use a browser viewport at least 1366×1100 to inspect
the whole fixture. The native Add measurement flow was also verified with this
pair and both cards pinned; it displays 87.653°.

## Thread annotation groups

Thread extrema use two ordered horizontal label gutters beside the selected
thread. Each callout draws its original witness segment and a single leader to
its midpoint, instead of two long extension lines in an independent rotated
measurement lane. The large statistics summary gets separate placement proposals.
All proposals still share the same viewport, obstacle, label, and leader collision
checks with ordinary measurements. Equal extrema retain a single min/max label.

Thread count uses endpoint markers and a leader without a full-length dimension
line: it is a count, and spanning the entire thread crossed many unrelated
extrema. Counts and dimensional values remain unchanged in the summary and Results.
The candidate solver caches pairwise conflicts/crossings during each layout pass.

The **Thread image 25** preview fixture contains the actual extracted thread
support and measurements from extras image 25, transformed into the preview image
plane. Its regression test requires the complete annotation set to fit without
label/leader collisions, horizontal text, and insertion-order-independent output.
The **Thread measurements** fixture covers a horizontal thread with equal extrema.

Construction-based distances use `distance_construction` and preserve the full
request definition for editing. Reference order, geometry basis, direction,
branch and explicit area/extension settings form the measurement identity;
changing the displayed statistic replaces the existing result. Preview requests
are debounced and stale responses are discarded on edits, cancellation or image
changes. Ambiguous previews cannot be committed.

The primary toolbar uses manufacturing terms: **Gap between surfaces**, **Gap at
a chosen position**, **Shortest distance**, **Distance between centers**, **Circle
center to line**, and **Circle edge to line**. A/B badges and direction/position
arrows connect those choices to the image. Side selection and bounded vs fitted
geometry remain explicit. Area limits/extensions are under a disclosure.

Normal gap extrema retain the actual section endpoints. A mean has a callout to
its contributing area, with each disconnected band rendered separately. Dashed
leaders distinguish annotation attachment from dimension segments. Dashed gold
supports denote fitted/virtual geometry; thread crest/root envelopes and the
thread axis explain where perpendicular diameters are constructed. Evidence
stays in original image coordinates when deformation display is exaggerated.
