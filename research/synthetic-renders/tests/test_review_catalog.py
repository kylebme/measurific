import collections
import json
import math
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from synth.catalog import LOCAL_FEATURE_LIMIT_PX, catalog_sha256, sample_catalog, sample_catalog_scene, sample_part


class ReviewCatalogTests(unittest.TestCase):
    def test_each_part_gets_distinct_imaging_classes_without_changing_its_mask(self):
        parts = sample_catalog(count=300, resolution=2048)
        digest = catalog_sha256(parts)
        for base_index in (0, 1, 2, 3, 4, 137, 299):
            variants = [sample_catalog_scene(20260923, base_index + 300 * variant,
                                             parts, digest) for variant in range(6)]
            self.assertTrue(all(m["geometry"] == variants[0]["geometry"] and
                                m["projection"] == variants[0]["projection"]
                                for m in variants))
            self.assertEqual(len({m["camera"]["focus_class"] for m in variants}), 5)
            self.assertEqual(len({m["illumination"]["class"] for m in variants}), 4)
            self.assertEqual(len({m["material"]["family"] for m in variants}), 6)

    def test_catalog_is_reproducible_and_has_local_diversity(self):
        parts = sample_catalog(count=300, resolution=2048)
        self.assertEqual(catalog_sha256(parts), catalog_sha256(sample_catalog(count=300, resolution=2048)))
        self.assertEqual(collections.Counter(p["geometry"]["family"] for p in parts),
                         {"plate": 60, "turned": 60, "machined": 60, "sphere": 60, "freeform": 60})
        self.assertEqual(len({p["geometry_seed"] for p in parts}), 300)
        for part in parts:
            g = part["geometry"]
            features = (g.get("features", []) + g.get("edge_segments", []) +
                        g.get("thread_regions", []) + g.get("axial_features", []) +
                        g.get("cuts", []) + g.get("rim_features", []))
            self.assertGreaterEqual(len(features), 16, part["part_id"])
            for feature in features:
                self.assertLessEqual(feature.get("span_px", 0), LOCAL_FEATURE_LIMIT_PX)
                self.assertLessEqual(feature.get("length_px", 0), LOCAL_FEATURE_LIMIT_PX)
                self.assertLessEqual(feature.get("width_px", 0), LOCAL_FEATURE_LIMIT_PX)

    def test_invalid_catalog_inputs(self):
        with self.assertRaises(ValueError):
            sample_part(1, -1)
        with self.assertRaises(ValueError):
            sample_part(1, 0, 128)
        with self.assertRaises(ValueError):
            sample_catalog(count=0)

    def test_turned_bodies_cover_one_to_hundreds_of_pixels(self):
        turned = [p["geometry"] for p in sample_catalog() if p["geometry"]["family"] == "turned"]
        diameters = [g["minimum_diameter_px"] for g in turned]
        self.assertEqual(len(turned), 60)
        self.assertEqual(diameters[0], 1.0)
        self.assertEqual(diameters[-1], 500.0)
        self.assertEqual(diameters, sorted(diameters))
        self.assertGreater(sum(d < 10 for d in diameters), 10)
        self.assertGreater(sum(d > 100 for d in diameters), 20)
        for g in turned:
            realized_minimum = 2 * g["base_radius_px"] * min(g["steps"])
            self.assertAlmostEqual(realized_minimum, g["minimum_diameter_px"], places=3)

    def test_tiny_tooth_hole_does_not_break_plate_fill(self):
        blender = os.environ.get("SYNTH_BLENDER", "/snap/blender/current/blender")
        if not os.path.isfile(blender):
            self.skipTest("pinned Blender binary is unavailable")
        part = sample_part(731029, 85, 2048)
        with tempfile.TemporaryDirectory(prefix="synth-plate-fill-") as directory:
            catalog = Path(directory) / "catalog.json"
            catalog.write_text(json.dumps({"parts": [part]}))
            worker = Path(__file__).resolve().parents[1] / "synth" / "review_worker.py"
            subprocess.run([blender, "-b", "-t", "2", "--python", str(worker),
                            "--", str(catalog), directory],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            with Image.open(Path(directory) / "masks" / (part["part_id"] + ".png")) as mask:
                self.assertEqual(mask.mode, "1")
                self.assertEqual(mask.getpixel((1024, 1500)), 255)
                self.assertGreater(mask.histogram()[255] / (2048 * 2048), 0.45)

    def test_one_pixel_turned_body_is_visible_across_the_part(self):
        blender = os.environ.get("SYNTH_BLENDER", "/snap/blender/current/blender")
        if not os.path.isfile(blender):
            self.skipTest("pinned Blender binary is unavailable")
        part = sample_part(731029, 1, 2048)
        g = part["geometry"]
        with tempfile.TemporaryDirectory(prefix="synth-thin-turned-") as directory:
            catalog = Path(directory) / "catalog.json"
            catalog.write_text(json.dumps({"parts": [part]}))
            worker = Path(__file__).resolve().parents[1] / "synth" / "review_worker.py"
            subprocess.run([blender, "-b", "-t", "2", "--python", str(worker),
                            "--", str(catalog), directory],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            with Image.open(Path(directory) / "masks" / (part["part_id"] + ".png")) as mask:
                self.assertEqual(mask.mode, "1")
                self.assertLess(mask.histogram()[255], 5000)
                hits = 0
                for i in range(128):
                    x_world = g["length_px"] * (-0.45 + 0.9 * i / 127)
                    x = round(1024 + x_world * math.cos(g["pose_angle_rad"]))
                    y = round(1024 - x_world * math.sin(g["pose_angle_rad"]))
                    hits += any(mask.getpixel((xx, yy)) for xx in range(x - 1, x + 2)
                                for yy in range(y - 1, y + 2))
                self.assertGreaterEqual(hits, 115)


if __name__ == "__main__":
    unittest.main()
