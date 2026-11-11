"""Non-ML contour measurement, in image pixel coordinates."""
from .contours import Contours
from .engine import evaluate
from .adapter import contours_from_image
from .annotation import annotate
from .interaction import (prepare_interactive_report, evaluate_interactive_measurement,
                          redefine_contour_geometry, redefine_compound, redefine_compound_sides,
                          remove_compound, MEASUREMENT_SPECS)
from .corners import corner_exclusions

__all__ = ['Contours', 'evaluate', 'contours_from_image', 'annotate', 'inspect_image',
           'prepare_interactive_report', 'evaluate_interactive_measurement',
           'redefine_contour_geometry', 'redefine_compound', 'redefine_compound_sides', 'remove_compound', 'MEASUREMENT_SPECS', 'corner_exclusions']


def inspect_image(image, request, *, contours=None, extraction=None):
    """Return (report, annotated PIL image, contours). Never runs a model."""
    diagnostics = None
    if contours is None:
        contours, diagnostics = contours_from_image(image, **(extraction or {}))
    elif not isinstance(contours, Contours):
        contours = Contours.from_dict(contours)
    report = evaluate(contours, request)
    if diagnostics is not None:
        report['extraction'] = diagnostics
    return report, annotate(image, report), contours
