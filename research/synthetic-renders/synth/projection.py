"""Orthographic mapping shared by the tiled renderer and registration tests."""


def pixel_center_world(full_width, full_height, ortho_scale, x, y):
    world_width = ortho_scale * full_width / full_height
    return (((x + 0.5) / full_width - 0.5) * world_width,
            (0.5 - (y + 0.5) / full_height) * ortho_scale)


def tile_camera(full_width, full_height, ortho_scale, x, y, tile_width, tile_height):
    """Return world center and vertical scale for a top-left-origin image tile."""
    world_width = ortho_scale * full_width / full_height
    return (((x + tile_width / 2) / full_width - 0.5) * world_width,
            (0.5 - (y + tile_height / 2) / full_height) * ortho_scale,
            ortho_scale * tile_height / full_height)
