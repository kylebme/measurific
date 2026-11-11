"""Stable IDs: append classes, never renumber an existing dataset schema."""
SEMANTICS = {
    "background": 0, "external_profile": 1, "hole": 2, "slot": 3,
    "flat": 4, "chamfer": 5, "fillet": 6, "shoulder": 7,
    "groove": 8, "taper": 9, "thread": 10, "tooth_profile": 11,
    "polygon": 12, "rounded_profile": 13, "cam_profile": 14,
    "tab": 15, "notch": 16,
}
COMPOUNDS = {
    "external_profile": "generic_closed_contour", "hole": "circle", "slot": "closed_rounded_profile",
    "flat": "width_surface", "chamfer": "angle", "fillet": "circular_arc", "shoulder": "step",
    "groove": "groove", "taper": "taper", "thread": "repeated_profile", "tooth_profile": "repeated_profile",
    "polygon": "closed_polygon", "rounded_profile": "closed_rounded_profile", "cam_profile": "generic_closed_contour",
    "tab": "generic_open_contour", "notch": "groove",
}
MEASUREMENTS = {
    "hole": ["associated_circle_diameter", "circularity_2d", "center_position"],
    "slot": ["width", "length", "orientation", "side_parallelism_2d", "end_arc_profile"],
    "flat": ["straightness_2d", "orientation", "length"],
    "chamfer": ["length", "orientation"], "fillet": ["radius", "arc_profile"],
    "shoulder": ["step_height", "orientation"], "groove": ["opening_width", "depth", "profile"],
    "taper": ["orientation", "profile"], "thread": ["pitch", "local_pitch", "visible_depth", "profile"],
    "tooth_profile": ["count", "pitch", "profile"],
}
PRIMITIVES = {"line": 1, "circular_arc": 2, "freeform": 3}
DEFECTS = {"none": 0, "burr": 1, "nick": 2}
PARTS = {
    "stepped_shaft": {"manufacturing_family": "turned", "view": "side"},
    "threaded_shaft": {"manufacturing_family": "turned", "view": "side"},
    "grooved_pin": {"manufacturing_family": "turned", "view": "side"},
    "tapered_pin": {"manufacturing_family": "turned", "view": "side"},
    "washer": {"manufacturing_family": "turned_or_cut", "view": "end"},
    "flange": {"manufacturing_family": "turned_and_drilled", "view": "end"},
    "drilled_plate": {"manufacturing_family": "prismatic", "view": "face"},
    "slotted_bracket": {"manufacturing_family": "prismatic", "view": "face"},
    "polygon_spacer": {"manufacturing_family": "prismatic", "view": "face"},
    "toothed_disk": {"manufacturing_family": "profile_cut", "view": "face"},
    "rack": {"manufacturing_family": "profile_cut", "view": "face"},
    "cam": {"manufacturing_family": "profile_cut", "view": "face"},
}

def taxonomy():
    from .primitive_labels import LABEL_VERSION,BASE_NAMES,COMPOUND_NAMES
    return {"schema_version": 1, "semantics": SEMANTICS, "primitives": PRIMITIVES,
            "learning_label_version": LABEL_VERSION,
            "base_classes": dict(zip(BASE_NAMES,range(len(BASE_NAMES)))),
            "compound_classes": dict(zip(COMPOUND_NAMES,range(len(COMPOUND_NAMES)))),
            "defects": DEFECTS, "part_families": PARTS, "compound_geometry": COMPOUNDS,
            "supported_2d_measurement_candidates": MEASUREMENTS,
            "scope": "Planar visible silhouettes; no hidden surfaces or 3D GD&T claims",
            "thread_templates": ["v", "trapezoid", "square"],
            "standard_conformance": "None: generic profiles, not certified thread/gear templates"}
