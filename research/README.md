# Research tools

Synthetic data generators and renderers used to train and validate the models
in [`packages/`](../packages). The desktop app does not depend on them, except
that [`synthetic-shapes`](synthetic-shapes/README.md) regenerates the app's demo
images.

| Folder | What it does |
|---|---|
| [`synthetic-shapes`](synthetic-shapes/README.md) | Procedural 2-D part geometry with exact primitive labels (the contour-model training corpus) and demo images. |
| [`synthetic-esf`](synthetic-esf/README.md) | Physically based telecentric silhouette renderer for 0.01 px accuracy studies (edge spread, blur, distortion, sensor). |
| [`synthetic-renders`](synthetic-renders/README.md) | Blender Cycles part renders with 16× geometric masks (the edge-model training source). |
| [`synthetic-macro`](synthetic-macro/README.md) | 512 × 512 multi-part scenes for training the coarse segmentation model. |

Generated datasets are large (tens of GB) and are not in this repository. Paths
in these docs and scripts such as `/media/kyle/T7 Shield/...` refer to the
author's external data drive; pass your own `--output`, or the matching
environment variable where a script supports one.
