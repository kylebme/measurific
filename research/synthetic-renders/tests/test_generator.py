import json
import os
import tempfile
import unittest

from PIL import Image

from synth.cli import _run_one
from synth.projection import pixel_center_world, tile_camera
from synth.reader import SceneReader
from synth.sampling import sample_scene
from synth.validation import require_complete_coverage, validate_scene


class SamplingTests(unittest.TestCase):
    def test_deterministic_pair_and_coverage(self):
        first = sample_scene(41, 0, 192, 192)
        second = sample_scene(41, 1, 192, 192)
        self.assertEqual(first, sample_scene(41, 0, 192, 192))
        self.assertEqual(first["geometry"], second["geometry"])
        self.assertNotEqual(first["backlight"], second["backlight"])
        self.assertNotEqual(first["scene_id"], sample_scene(42, 0, 192, 192)["scene_id"])
        require_complete_coverage([sample_scene(41, i, 192, 192) for i in range(60)])

    def test_tile_projection_matches_global_pixel_centers(self):
        width, height, scale = 8192, 8192, 512
        for x, y, tw, th in ((0, 0, 1024, 1024), (7168, 6144, 1024, 1024),
                             (1024, 2048, 1024, 1024), (0, 0, 511, 1000)):
            cx, cy, local_scale = tile_camera(width, height, scale, x, y, tw, th)
            for u, v in ((0, 0), (tw - 1, th - 1), (tw // 2, th // 2)):
                actual_x = cx + ((u + 0.5) / tw - 0.5) * local_scale * tw / th
                actual_y = cy + (0.5 - (v + 0.5) / th) * local_scale
                expected_x, expected_y = pixel_center_world(width, height, scale, x + u, y + v)
                self.assertAlmostEqual(actual_x, expected_x, places=10)
                self.assertAlmostEqual(actual_y, expected_y, places=10)


class BlenderRegistrationTest(unittest.TestCase):
    def test_binary_mask_and_cycles_silhouette_are_registered(self):
        blender = os.environ.get("SYNTH_BLENDER", "/snap/blender/current/blender")
        if not os.path.isfile(blender):
            self.skipTest("pinned Blender binary is unavailable")
        manifest = sample_scene(707, 0, 192, 192)
        g = manifest["geometry"]
        g["outline"] = "round"
        g["edge_damage"] = False
        g["burrs"] = []
        g["features"] = [{"kind": "round", "center_px": [0, 0], "width_px": 24,
                          "length_px": 24, "angle_rad": 0, "scale_class": "patch",
                          "sides": 32, "phase_rad": 0}]
        manifest["backlight"]["dust"] = []
        manifest["backlight"]["gradient"] = 0
        manifest["backlight"]["texture_strength"] = 0
        manifest["illumination"]["spill_strength"] = 0
        manifest["illumination"]["ambient_strength"] = 0
        manifest["material"]["albedo"] = 0.01
        manifest["camera"]["focus_distance_world"] = 700
        manifest["camera"]["fstop"] = 64
        manifest["render"]["cycles_samples"] = 16
        with tempfile.TemporaryDirectory(prefix="synth-registration-") as directory:
            _run_one(manifest, directory, blender, threads=2)
            path = os.path.join(directory, manifest["scene_id"])
            validate_scene(path)
            with Image.open(os.path.join(path, "image.png")) as source:
                source.load()
                background = source.getpixel((0, 0))
                object_level = source.getpixel((96 + 30, 96))
                self.assertLess(object_level, background * 0.4)
                self.assertGreater(source.getpixel((96, 96)), background * 0.7)
                threshold = (background + object_level) / 2
                source_dark = Image.new("L", source.size)
                source_dark.putdata([255 if value < threshold else 0 for value in source.getdata()])
                source_box = source_dark.getbbox()
            with open(os.path.join(path, "mask", "index.json")) as stream:
                index = json.load(stream)
            mask = Image.new("1", (index["width"], index["height"]))
            for tile in index["tiles"]:
                with Image.open(os.path.join(path, "mask", tile["file"])) as image:
                    mask.paste(image, (tile["x"], tile["y"]))
            self.assertEqual(mask.getpixel((96 * 16, 96 * 16)), 0)
            mask_box = mask.getbbox()
            projected_box = tuple(round(value / 16) for value in mask_box)
            for actual, expected in zip(source_box, projected_box):
                self.assertLessEqual(abs(actual - expected), 2, (source_box, projected_box))
            reader = SceneReader(path)
            box = (980, 980, 1090, 1090)  # crosses four independent mask tiles
            self.assertEqual(list(reader.mask_region(box).getdata()), list(mask.crop(box).getdata()))
            self.assertEqual(reader.source_region((0, 0, 12, 12)).size, (12, 12))

    def test_partial_edge_tiles_keep_the_same_projection(self):
        blender = os.environ.get("SYNTH_BLENDER", "/snap/blender/current/blender")
        if not os.path.isfile(blender):
            self.skipTest("pinned Blender binary is unavailable")
        manifest = sample_scene(1001, 0, 200, 200)
        g = manifest["geometry"]
        g.update({"radius_px": 52, "outline": "round", "outline_amplitude_px": 0,
                  "pose_angle_rad": 0, "features": [], "burrs": []})
        manifest["render"]["mask_tile_size"] = 1200
        manifest["render"]["cycles_samples"] = 4
        with tempfile.TemporaryDirectory(prefix="synth-partial-tiles-") as directory:
            _run_one(manifest, directory, blender, threads=2)
            path = os.path.join(directory, manifest["scene_id"])
            result = validate_scene(path)
            self.assertEqual(result["mask_bounds"], (768, 768, 2431, 2431))
            with open(os.path.join(path, "mask", "index.json")) as stream:
                index = json.load(stream)
            partial = {(tile["x"], tile["y"]): tile["foreground_pixels"]
                       for tile in index["tiles"] if tile["width"] < 1200 or tile["height"] < 1200}
            self.assertGreater(partial[(2400, 1200)], 0)
            self.assertEqual(partial[(2400, 1200)], partial[(1200, 2400)])


if __name__ == "__main__":
    unittest.main()
