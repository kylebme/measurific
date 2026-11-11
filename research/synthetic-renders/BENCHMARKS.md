# Local baseline measurements

Measurements were made with Blender 5.2.2 LTS on the installed CPU runtime. They are a starting point for a larger production run, not a throughput guarantee for other hardware or geometry.

## Resolution and mask cost

The same plate geometry seed was rendered at two square source sizes with 4 Cycles samples to isolate render and mask overhead. Both used 16× masks and 1024×1024 Workbench tiles.

| Source | Mask | Blender render duration | Source PNG | Mask tile bytes |
| --- | --- | ---: | ---: | ---: |
| 512×512 | 8192×8192 | 8.16 s | 226 KB | about 32 KB after Group 4 conversion |
| 1024×1024 | 16384×16384 | 29.95 s | 734 KB | 108 KB Group 4 |

The 512×512 image retains a target width above 400 source pixels, including geometry much larger than the downstream 48-pixel patch. It was selected as the starting production resolution because the 1024×1024 mask pass cost about 3.7 times as much on this machine. The generator keeps dimensions configurable.

At 512×512 and the chosen 128 Cycles samples, the final pilot scene took 12.14 seconds in Blender and produced a 147 KB source PNG plus 29 KB of Group 4 mask tiles; all files together were 189 KB. Actual corpus size depends strongly on geometry, material, sampling noise, and compression. A 100 GB corpus requires many scenes and a sustained production run.

## Lossless format comparison

One 1024×1024 rendered source and one mixed-occupancy 1024×1024 mask tile were used for this comparison. Decode times are five-run medians from Pillow on the local host.

| Data | Format | Bytes | Decode time |
| --- | --- | ---: | ---: |
| Source | 16-bit PNG | 733,578 | 8.46 ms |
| Source | 16-bit TIFF Deflate | 998,962 | 4.02 ms |
| Source | 16-bit TIFF LZW | 1,643,334 | 9.81 ms |
| Mask tile | 1-bit PNG | 2,768 | 0.64 ms |
| Mask tile | 1-bit TIFF Group 4 | 860 | 0.62 ms |
| Mask tile | 1-bit TIFF Deflate | 2,098 | 0.62 ms |

The source uses PNG for its smaller size. Mask tiles use Group 4 TIFF for substantially smaller files with similar tile decode cost. Each tile remains independently readable.

## Cycles convergence

Two 256×256 cases were compared with a 512-sample reference: a defocused sphere and a defocused reflective machined cylinder. Denoising was off and the pixel filter was fixed at 0.01.

| Case | Samples | Mean absolute difference to 512 | 99th percentile difference | Source render duration |
| --- | ---: | ---: | ---: | ---: |
| Sphere | 64 | 45.76 DN | 866 DN | 1.33 s |
| Sphere | 128 | 30.02 DN | 560 DN | 1.81 s |
| Machined cylinder | 64 | 20.91 DN | 674 DN | 1.38 s |
| Machined cylinder | 128 | 14.34 DN | 441 DN | 1.68 s |

The default is 128 samples. It reduces the measured error at modest additional source-render cost while the mask pass remains the larger fixed cost. Stronger convergence checks should precede any final large production run.

## Geometry and registration checks

A 60-scene 192×192 stress batch completed and passed the coverage gate: 12 scenes per part family, 10 per feature-scale class, all six material families, all four focus classes, all three illumination classes, all thread profiles, and all plate opening classes. A separate 12-scene turned and machined-cylinder batch exercised every thread profile and the post-Boolean mesh checks. The integration tests also cover source/mask registration for a plate with a hole and the partial-tile projection at a 200×200 source size.

## GPU rendering and physical_v3 calibration (2026-09-27)

Measured on the RTX 3070 (OptiX) with Blender 5.2.2 LTS at 2048².

| Measurement | Result |
| --- | --- |
| Scene 2295 source render, CPU run (2 threads, 128 spp) vs OptiX | 258 s vs 5.2 s; images agree to 0.007 DN mean absolute difference |
| Wall time per scene incl. Blender start and geometry, OptiX 128 spp | 5–9 s |
| 16× mask pass, 256 Workbench tiles (old path) | 117 s, of which ~0.3 s/tile is PNG write and read-back and 0.12 s is rendering |
| physical_v3 pilot, 6 workers, 3 GPU slots | 6.7 s per scene effective; median 11 s render; full-mask scenes 20 s of mask time for ~30–40 rendered tiles |

**Pixel filter.** Blender 5.2 applies `scene.render.filter_size` to Cycles. The old 0.01 value point-samples: sharp renders have ~0.01 intermediate pixels per edge crossing. A box filter of width 1 gives 0.84, as expected for area integration.

**Sample count.** Edge-pixel RMS error against a 2048-spp reference, as a fraction of edge contrast:

| Setting | Edge RMS error | Render time |
| --- | ---: | ---: |
| 128 spp | 1.45% | 4.6 s |
| 512 spp | 0.64% | 15.4 s |
| 512 max, adaptive threshold 0.003 | 0.64% | 4.4 s |

**Orthographic depth of field.** A knife edge at z = 0 with a camera at z = 4096 was rendered for several f-stops and defocus distances. The blur-circle diameter from the line-spread variance matches `dz / (fstop · D)` to within −8% to +4% (for example, measured 9.02 px vs 9.77 predicted, and 11.67 vs 12.21). The aperture radius is therefore 1 / (2 fstop) world units, with NA = 1 / (2 fstop D). The 50% crossing stays at the geometric edge under defocus (0.293–0.298 px vs 0.300).

**Registration of the 104-scene pilot.** `fit_shift` against exact 1× coverage, median outward shift per scene by class: in focus −0.002 px, mild +0.018, medium −0.007, heavy −0.013. Effective Gaussian blur σ (median) was 0.7 / 1.0 / 1.4 / 2.8 px for the same classes. Larger per-scene offsets are physical, not registration errors. Examples: backlight wrapping onto light-matte side walls (−0.1 px), and deep walls outside the depth of field, which spread darkness outward.
