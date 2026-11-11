"""Blender-free, region-readable output consumer."""

import json
import os

from PIL import Image


class SceneReader:
    def __init__(self, scene_dir):
        self.scene_dir = scene_dir
        with open(os.path.join(scene_dir, "metadata.json")) as stream:
            self.metadata = json.load(stream)
        with open(os.path.join(scene_dir, self.metadata["files"]["mask_index"])) as stream:
            self.mask_index = json.load(stream)

    def source_region(self, box):
        """Return a source-image crop; box is (left, top, right, bottom)."""
        with Image.open(os.path.join(self.scene_dir, self.metadata["files"]["image"])) as image:
            return image.crop(box).copy()

    def mask_region(self, box):
        """Read only mask tiles intersecting box, in full-mask pixel coordinates."""
        x0, y0, x1, y1 = box
        index = self.mask_index
        if not (0 <= x0 < x1 <= index["width"] and 0 <= y0 < y1 <= index["height"]):
            raise ValueError("mask region must lie within full mask dimensions")
        output = Image.new("1", (x1 - x0, y1 - y0))
        for tile in index["tiles"]:
            tx0, ty0 = tile["x"], tile["y"]
            tx1, ty1 = tx0 + tile["width"], ty0 + tile["height"]
            left, top = max(x0, tx0), max(y0, ty0)
            right, bottom = min(x1, tx1), min(y1, ty1)
            if left >= right or top >= bottom:
                continue
            path = os.path.join(self.scene_dir, self.metadata["files"]["mask_dir"], tile["file"])
            with Image.open(path) as image:
                crop = image.crop((left - tx0, top - ty0, right - tx0, bottom - ty0))
                output.paste(crop, (left - x0, top - y0))
        return output
