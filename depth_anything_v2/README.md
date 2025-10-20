# candle-DepthAnythingV2

[Depth Anything V2] is a model for Monocular Depth Estimation (MDE, i.e. just using a single image) which
builds on the [DINOv2](https://github.com/facebookresearch/dinov2) vision transformer.

This example first instantiates the DINOv2 model and then proceeds to create DepthAnythingV2 and run it.

## Running an example with color map

```bash
# CPU
$ cargo run -- \
  --color-map \
  --image ../assets/bike.jpg

# CUDA
$ cargo run --features cuda -- \
  --color-map \
  --image ../assets/bike.jpg

# Metal
$ cargo run --features metal -- \
  --color-map \
  --image ../assets/bike.jpg 
```

