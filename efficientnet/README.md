# candle-efficientnet

Demonstrates a Candle implementation of EfficientNet for image classification based on ImageNet classes.

## Running an example

```bash
# CPU
$ cargo run --release -- \
  --image ../assets/bike.jpg \
  --which b1

# CUDA
$ cargo run --features cuda --release -- \
  --image ../assets/bike.jpg \
  --which b1

# Metal
$ cargo run --features metal --release -- \
  --image ../assets/bike.jpg \
  --which b1

> bicycle-built-for-two, tandem bicycle, tandem: 45.85%
> mountain bike, all-terrain bike, off-roader: 30.45%
> crash helmet            : 2.58%
> unicycle, monocycle     : 2.21%
> tricycle, trike, velocipede: 1.53%
```
