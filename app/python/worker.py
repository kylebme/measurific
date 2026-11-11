#!/usr/bin/env python3
"""Persistent JSON-lines bridge between Electron and the inspection pipeline."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from PIL import Image


DEFAULT_MODEL = Path('packages') / 'contour-model' / 'release-v6' / 'onnx'
DEFAULT_EDGE_MODEL = Path('packages') / 'edge-model' / 'models' / 'sdf_v5-onnx'
EDGE_METHODS = ('pae', 'deep-learning')


def configure_imports(root: Path | None) -> None:
    """Development checkouts import the pipeline from source; packaged builds install it."""
    if root is None:
        return
    # edge-model keeps its package (edgesr) at the project root.
    for source in (root / "packages" / "edge-model", root / "packages" / "contour-model" / "src", root / "packages" / "measurement" / "src",
                   root / "packages" / "pae" / "src"):
        if source.is_dir():
            sys.path.insert(0, str(source))


class InspectionWorker:
    def __init__(self, root: Path | None = None, backend: str | None = None, *, model: Path | str | None = None,
                 edge_model: Path | str | None = None, device: str = 'cpu', edge_method: str = 'pae'):
        """The app always uses the learned model; ``backend='geometry'`` only serves tests.

        ``edge_method='pae'`` seeds PAE with the coarse model of ``edge_model``;
        ``'deep-learning'`` runs the full coarse + fine edge model.
        """
        configure_imports(root)
        self.root = root
        from inspection_measurement.adapter import load_edge_model, load_pae_edge_model
        if edge_model is None:
            if root is None:
                raise ValueError('An edge model path is required')
            edge_model = root / DEFAULT_EDGE_MODEL
        if edge_method not in EDGE_METHODS:
            raise ValueError(f'Edge method must be one of {", ".join(EDGE_METHODS)}')
        self.edge_method = edge_method
        if edge_method == 'pae':
            self.edge_model = load_pae_edge_model(Path(edge_model), device=device)
            # Compile PAE's kernels now rather than during the first analysis.
            self.edge_model.warmup()
        else:
            self.edge_model = load_edge_model(Path(edge_model), device=device)
        self.backend = backend or 'learned'
        if self.backend not in ('geometry', 'learned'):
            raise ValueError('Recognition must be geometry or learned')
        self.device = device
        self.model, self.checkpoint, self.model_path = None, {}, None
        if self.backend == 'learned':
            from contour_model.onnx_runtime import load_recognition_model
            if model is None:
                if root is None:
                    raise ValueError('A recognition model path is required')
                model = root / DEFAULT_MODEL
            self.model_path = Path(model)
            self.model, self.checkpoint = load_recognition_model(self.model_path, device=device)
            if self.model.config.label_schema != "primitives-compounds-v2":
                raise ValueError("The GUI requires the primitives-compounds-v2 release checkpoint")
        from contour_model.geometric import GeometricConfig
        self.geometric_config = GeometricConfig()
        self.current_contours = None
        self.current_prediction = None
        self.current_report = None

    def health(self):
        """Runtime identity for diagnostics and packaged-build smoke tests."""
        import importlib.metadata
        import platform

        versions = {}
        for name in ('numpy', 'scipy', 'shapely', 'pillow', 'opencv-python-headless', 'onnxruntime',
                     'onnxruntime-directml', 'numba', 'torch', 'inspection-measurement', 'inspection-contour-model',
                     'inspection-edge-model', 'inspection-pae'):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                pass
        try:
            import onnxruntime
            available = onnxruntime.get_available_providers()
        except ImportError:
            available = []
        model = self.model.describe() if hasattr(self.model, 'describe') else (
            {'format': 'torch', 'path': str(self.model_path)} if self.model is not None else None)
        return {
            'python': sys.version.split()[0], 'platform': platform.platform(), 'executable': sys.executable,
            'recognition': self.backend, 'device': self.device, 'edgeMethod': self.edge_method,
            'availableProviders': available,
            'model': model, 'edgeModel': self.edge_model.describe(), 'packages': versions,
        }

    @staticmethod
    def grayscale_pixels(image: Image.Image) -> np.ndarray:
        """Preserve native grayscale depth; ordinary color files become L8."""
        if image.mode in ('I;16', 'I;16L', 'I;16B'):
            values = np.asarray(image)
            if values.dtype.byteorder == '>':
                values = values.astype(np.uint16)
            return np.ascontiguousarray(values, dtype=np.uint16)
        if image.mode == 'L':
            return np.ascontiguousarray(np.asarray(image), dtype=np.uint8)
        return np.ascontiguousarray(np.asarray(image.convert('L')), dtype=np.uint8)

    def contours_for_image(self, path: Path, gray: np.ndarray, valid_mask=None):
        from inspection_measurement import Contours, contours_from_image

        cached = path.with_suffix(".contours.npz")
        if cached.exists() and valid_mask is None:
            with np.load(cached, allow_pickle=False) as values:
                xy = values["actual_xy_px"].astype(np.float64)
                offsets = values["ring_offsets"].astype(np.int64)
            return Contours(xy, offsets), {
                "method": "synthetic_generator_contours",
                "source": str(cached),
                "interpolated_point_count": 0,
            }
        contours, diagnostics = contours_from_image(gray, valid_mask=valid_mask, model=self.edge_model)
        return contours, diagnostics

    @staticmethod
    def analysis_mask(shape, spec):
        if spec is None:
            return None
        if not isinstance(spec, dict) or spec.get("shape") != "circle":
            raise ValueError("Unsupported analysis mask")
        cx, cy, radius = (float(spec.get(key)) for key in ("cx", "cy", "r"))
        if not np.isfinite([cx, cy, radius]).all() or radius <= 0:
            raise ValueError("Invalid circular analysis mask")
        yy, xx = np.ogrid[:shape[0], :shape[1]]
        return (xx + .5 - cx) ** 2 + (yy + .5 - cy) ** 2 <= radius ** 2

    def analyze(self, path_value: str, valid_mask_spec=None):
        from inspection_measurement import prepare_interactive_report

        started = time.perf_counter()
        path = Path(path_value).expanduser().resolve()
        # A failed load must not leave relationship requests bound to an older part.
        self.current_contours = None
        self.current_prediction = None
        self.current_report = None
        with Image.open(path) as image:
            width, height = image.size
            gray = self.grayscale_pixels(image)
        valid_mask = self.analysis_mask(gray.shape, valid_mask_spec)
        contours, extraction = self.contours_for_image(path, gray, valid_mask)
        extracted = time.perf_counter()
        # The recognizers take one part (outline + holes); each object is recognized alone.
        from contour_model.objects import predict_objects
        if self.backend == 'geometry':
            from contour_model.geometric import predict
            prediction, _ = predict_objects(
                lambda part: predict(part, units='px', config=self.geometric_config), contours.model_arrays())
        else:
            from contour_model.inference_primitives import predict
            threshold = float(self.checkpoint.get("input_contract", {}).get("default_threshold", 0.7))
            instances = bool(self.checkpoint.get("instances_trained", False))
            prediction, _ = predict_objects(
                lambda part: predict(self.model, part, threshold=threshold, instances=instances, units="px"),
                contours.model_arrays())
        predicted = time.perf_counter()
        report, measurement_request = prepare_interactive_report(contours, prediction)
        measured = time.perf_counter()
        report["extraction"] = extraction
        self.current_contours = contours
        self.current_prediction = measurement_request
        self.current_report = report
        return {
            "path": str(path),
            "name": path.name,
            "width": width,
            "height": height,
            "prediction": prediction,
            "report": report,
            "timingsMs": {
                "edges": round((extracted - started) * 1000),
                "recognition": round((predicted - extracted) * 1000),
                "measurement": round((measured - predicted) * 1000),
            },
        }

    def analyze_response(self, path_value: str, valid_mask_spec=None):
        """Return extraction failures as displayable results for the GUI."""
        path = Path(path_value).expanduser().resolve()
        try:
            return self.analyze(path_value, valid_mask_spec)
        except (ValueError, RuntimeError) as error:
            diagnostic = getattr(error, 'diagnostic', None)
            if diagnostic is None:
                raise
            with Image.open(path) as image:
                width, height = image.size
            return {
                "path": str(path), "name": path.name,
                "width": width, "height": height,
                "error": str(error), "diagnostic": diagnostic,
            }

    def relationship(self, request: dict):
        from inspection_measurement import evaluate_interactive_measurement

        if self.current_contours is None or self.current_prediction is None or self.current_report is None:
            raise ValueError("Analyze an image before creating a relative measurement")
        normalized = dict(request)
        # Retain compatibility with older bridge clients while all validation
        # and geometry-dependent applicability stays in inspection-measurement.
        if "featureRefs" not in normalized and "featureA" in normalized:
            normalized["featureRefs"] = [normalized.get("featureA"), normalized.get("featureB")]
        result = evaluate_interactive_measurement(
            self.current_contours, self.current_prediction, self.current_report, normalized)
        result["feature_anchors"] = self.feature_anchors(self.current_report, self.measurement_refs(normalized))
        return result

    @staticmethod
    def measurement_refs(request: dict):
        params = request.get("params")
        datum_ref = params.get("datum_feature_ref") if isinstance(params, dict) else None
        return [*(request.get("featureRefs") or []), *([datum_ref] if datum_ref else [])]

    @staticmethod
    def feature_anchors(report: dict, refs):
        """Describe derived junction vertices so their references survive reindexing.

        Vertex IDs are assigned by enumeration order and are regenerated whenever
        retained support changes; recognized features keep stable IDs.
        """
        records = {feature["id"]: feature for feature in report.get("features", [])}
        anchors = {}
        for ref in refs:
            feature = records.get(ref)
            if feature and feature.get("type") == "point" and feature.get("semantic") == "vertex":
                anchors[ref] = {
                    "adjacent_feature_ids": sorted(feature.get("adjacent_feature_ids") or []),
                    "point": (feature.get("geometry") or {}).get("point"),
                }
        return anchors

    @staticmethod
    def resolve_anchor(report: dict, anchor: dict, max_distance_px=6.0):
        """Find the regenerated vertex nearest the anchored one; exclusion radii are <= 5 px."""
        adjacent = set(anchor.get("adjacent_feature_ids") or [])
        point = anchor.get("point")
        best = None
        for feature in report.get("features", []):
            if feature.get("type") != "point" or feature.get("semantic") != "vertex":
                continue
            candidate_adjacent = set(feature.get("adjacent_feature_ids") or [])
            shared = len(adjacent & candidate_adjacent)
            candidate = (feature.get("geometry") or {}).get("point")
            if point is None or candidate is None:
                if candidate_adjacent != adjacent:
                    continue
                distance = 0.0
            else:
                distance = float(np.hypot(candidate[0] - point[0], candidate[1] - point[1]))
                if distance > max_distance_px:
                    continue
            key = (-shared, distance)
            if best is None or key < best[0]:
                best = (key, feature["id"])
        return None if best is None else best[1]

    def remeasure(self, measurement: dict):
        """Re-evaluate one saved measurement against the current report.

        Returns an ``unavailable`` outcome instead of raising when the retained
        support no longer yields the referenced features.
        """
        from inspection_measurement import evaluate_interactive_measurement

        request = deepcopy(measurement)
        anchors = request.pop("featureAnchors", None) or {}
        mapping = {}
        for ref, anchor in anchors.items():
            resolved = self.resolve_anchor(self.current_report, anchor)
            if resolved is None:
                return {"id": request.get("id"), "status": "unavailable",
                        "reason": "A referenced corner vertex is not present at this corner exclusion"}
            mapping[ref] = resolved
        request["featureRefs"] = [mapping.get(ref, ref) for ref in request.get("featureRefs") or []]
        params = request.get("params")
        if isinstance(params, dict) and params.get("datum_feature_ref"):
            request["params"] = {**params, "datum_feature_ref": mapping.get(params["datum_feature_ref"], params["datum_feature_ref"])}
        try:
            result = evaluate_interactive_measurement(
                self.current_contours, self.current_prediction, self.current_report, request)
        except ValueError as error:
            return {"id": request.get("id"), "status": "unavailable", "reason": str(error)}
        result["feature_anchors"] = self.feature_anchors(self.current_report, self.measurement_refs(request))
        return {"id": request.get("id"), "status": "measured", "result": result, "feature_map": mapping}

    def set_corner_exclusion(self, radius, measurements=None):
        from inspection_measurement import prepare_interactive_report

        if self.current_contours is None or self.current_prediction is None or self.current_report is None:
            raise ValueError("Analyze an image before changing measurement settings")
        request = deepcopy(self.current_prediction)
        request["corner_exclusion_px"] = radius
        report, prepared_request = prepare_interactive_report(self.current_contours, request)
        extraction = self.current_report.get("extraction")
        if extraction is not None:
            report["extraction"] = extraction
        self.current_prediction = prepared_request
        self.current_report = report
        return {"prediction": prepared_request, "report": report,
                "measurements": [self.remeasure(measurement) for measurement in measurements or []]}

    def redefine_geometry(self, request: dict):
        from inspection_measurement import redefine_contour_geometry

        if self.current_contours is None or self.current_prediction is None or self.current_report is None:
            raise ValueError("Analyze an image before redefining geometry")
        result = redefine_contour_geometry(
            self.current_contours, self.current_prediction,
            request.get("type"), request.get("points"), request.get("cursorPath"),
        )
        if result["complete"]:
            extraction = self.current_report.get("extraction")
            if extraction is not None:
                result["report"]["extraction"] = extraction
            self.current_prediction = result.pop("request")
            self.current_report = result["report"]
            result["prediction"] = self.current_prediction
        return result

    def redefine_compound(self, request: dict):
        from inspection_measurement import redefine_compound, redefine_compound_sides

        if self.current_contours is None or self.current_prediction is None or self.current_report is None:
            raise ValueError("Analyze an image before redefining geometry")
        if "points" in request:
            # Two-sided compounds are picked as start/end clicks on each side.
            result = redefine_compound_sides(self.current_contours, self.current_prediction,
                                             request.get("semantic"), request.get("points"),
                                             request.get("cursorPaths"))
            if not result["complete"]:
                return result
        else:
            result = redefine_compound(self.current_contours, self.current_prediction,
                                       request.get("semantic"), request.get("featureIds"))
        extraction = self.current_report.get("extraction")
        if extraction is not None:
            result["report"]["extraction"] = extraction
        self.current_prediction = result.pop("request")
        self.current_report = result["report"]
        result["prediction"] = self.current_prediction
        return result

    def remove_compound(self, request: dict):
        from inspection_measurement import remove_compound

        if self.current_contours is None or self.current_prediction is None or self.current_report is None:
            raise ValueError("Analyze an image before redefining geometry")
        result = remove_compound(self.current_contours, self.current_prediction, request.get("featureId"))
        extraction = self.current_report.get("extraction")
        if extraction is not None:
            result["report"]["extraction"] = extraction
        self.current_prediction = result.pop("request")
        self.current_report = result["report"]
        result["prediction"] = self.current_prediction
        return result

    def handle(self, message: dict):
        command = message.get("command")
        if command == "analyze":
            return self.analyze_response(message["path"], message.get("validMask"))
        if command == "relationship":
            return self.relationship(message["request"])
        if command == "set_corner_exclusion":
            return self.set_corner_exclusion(message["radius"], message.get("measurements"))
        if command == "redefine_geometry":
            return self.redefine_geometry(message["request"])
        if command == "redefine_compound":
            return self.redefine_compound(message["request"])
        if command == "remove_compound":
            return self.remove_compound(message["request"])
        if command == "health":
            return self.health()
        raise ValueError(f"Unknown command: {command}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, help="development checkout to import the pipeline from")
    parser.add_argument("--model", type=Path, help="exported ONNX directory or torch checkpoint")
    parser.add_argument("--edge-model", type=Path, help="exported edge-model ONNX directory or SDF torch checkpoint")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--edge-method", choices=EDGE_METHODS, default="pae",
                        help="PAE seeded by the edge model's coarse model, or the full deep learning edge model")
    args = parser.parse_args()
    worker = InspectionWorker(args.root.resolve() if args.root else None, model=args.model,
                              edge_model=args.edge_model, device=args.device, edge_method=args.edge_method)
    for line in sys.stdin:
        try:
            message = json.loads(line)
            response = {"id": message.get("id"), "result": worker.handle(message)}
        except Exception as error:  # errors must cross the process boundary cleanly
            traceback.print_exc(file=sys.stderr)
            response = {"id": message.get("id") if "message" in locals() else None, "error": str(error)}
        print(json.dumps(response, allow_nan=False, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
