import collections
import json
import math
import os
import tempfile
import unittest

from PIL import Image

from synth import host_v3
from synth.appearance_v3 import sample_v3_scene
from synth.catalog import catalog_sha256, sample_catalog


class PhysicalV3SamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parts = sample_catalog(count=300, resolution=2048)
        cls.digest = catalog_sha256(cls.parts)
        cls.scenes = [sample_v3_scene(20260927, i, cls.parts, cls.digest) for i in range(1500)]

    def test_deterministic_and_geometry_preserved(self):
        again = sample_v3_scene(20260927, 17, self.parts, self.digest)
        self.assertEqual(again, self.scenes[17])
        part = next(p for p in self.parts if p["part_id"] == again["catalog"]["part_id"])
        self.assertEqual(again["geometry"], part["geometry"])
        self.assertNotEqual(self.scenes[0]["scene_id"],
                            sample_v3_scene(1, 0, self.parts, self.digest)["scene_id"])

    def test_class_mix(self):
        n = len(self.scenes)
        tilt = sum(m["pose"]["tilt_rad"] > 0 for m in self.scenes) / n
        levels = collections.Counter(m["contamination"]["level"] for m in self.scenes)
        blur = collections.Counter(m["optics"]["blur_class"] for m in self.scenes)
        point = sum(m["render"]["pixel_filter"] == "point" for m in self.scenes) / n
        self.assertAlmostEqual(tilt, 0.30, delta=0.04)
        self.assertAlmostEqual(levels["clean"] / n, 0.60, delta=0.04)
        self.assertAlmostEqual(levels["heavy"] / n, 0.10, delta=0.03)
        self.assertAlmostEqual(point, 0.10, delta=0.03)
        self.assertEqual(set(blur), {"in_focus", "mild", "medium", "heavy"})
        self.assertTrue(all(m["pose"]["tilt_rad"] <= math.radians(20) for m in self.scenes))

    def test_mask_plan_follows_tilt_and_contamination(self):
        for m in self.scenes:
            expected = ("full" if m["pose"]["tilt_rad"] > 0 else
                        "base_plus_contaminants" if m["contamination"]["items"] else "base")
            self.assertEqual(m["mask_plan"], expected)

    def test_optics_are_physically_coupled(self):
        for m in self.scenes:
            o = m["optics"]
            # Diffraction core stays below ~1.2 px sigma for the sampled NA / pixel size.
            self.assertLessEqual(o["lens_psf"]["diffraction_sigma_px"], 1.2 + 1e-6)
            ang = m["backlight"]["angular"]
            if ang["type"] == "collimated":
                self.assertGreaterEqual(ang["half_angle_deg"],
                                        1.5 * math.degrees(math.asin(o["na"])) - 1e-6)
            for item in m["contamination"]["items"]:
                if item["kind"] in ("fiber", "hair", "particle"):
                    self.assertGreater(item["diameter_px"], 0)


class HostMergeTests(unittest.TestCase):
    def test_contaminant_tiles_are_ored_into_shared_base(self):
        with tempfile.TemporaryDirectory() as root:
            base = os.path.join(root, "base_masks", "part_x")
            os.makedirs(base)
            tiles = []
            for x in (0, 16):
                im = Image.new("1", (16, 16), 0)
                if x == 0:
                    im.paste(1, (0, 0, 8, 16))
                im.save(os.path.join(base, f"x{x:05d}_y00000.tif"), format="TIFF", compression="group4")
                tiles.append({"x": x, "y": 0, "width": 16, "height": 16,
                              "file": f"x{x:05d}_y00000.tif",
                              "foreground_pixels": 128 if x == 0 else 0})
            with open(os.path.join(base, "index.json"), "w") as f:
                json.dump({"width": 32, "height": 16, "tile_size": 16, "tiles": tiles,
                           "foreground_pixels": 128}, f)
            scene = os.path.join(root, "scene_1")
            extra = os.path.join(scene, "mask_extra")
            os.makedirs(extra)
            e = Image.new("1", (16, 16), 0)
            e.paste(1, (4, 0, 12, 2))           # overlaps the base by 8 px, adds 8
            e.save(os.path.join(extra, "x00000_y00000.png"))
            with open(os.path.join(extra, "index.json"), "w") as f:
                json.dump({"tiles": [{"x": 0, "y": 0, "width": 16, "height": 16,
                                      "file": "x00000_y00000.png", "foreground_pixels": 16}]}, f)
            index, changed = host_v3._merge_extra(scene, base, "part_x")
            self.assertEqual(changed, 1)
            self.assertEqual(index["foreground_pixels"], 136)
            by_x = {t["x"]: t for t in index["tiles"]}
            self.assertEqual(by_x[0]["file"], "x00000_y00000.tif")
            self.assertEqual(by_x[16]["file"], "../../base_masks/part_x/x00016_y00000.tif")
            self.assertTrue(os.path.isfile(os.path.join(scene, "mask", by_x[16]["file"])))
            with Image.open(os.path.join(scene, "mask", "x00000_y00000.tif")) as im:
                self.assertEqual(im.histogram()[255], 136)
            self.assertFalse(os.path.exists(extra))


if __name__ == "__main__":
    unittest.main()
