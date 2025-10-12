# candle-blip

The
[blip-image-captioning](https://huggingface.co/Salesforce/blip-image-captioning-base)
model can generate captions for an input image.

## Running on an examples

```bash
# Running on CPU
$ cargo run --release -- --image ../assets/bike.jpg

# Running on CUDA
$ cargo run --release --features cuda -- --image ../assets/bike.jpg

# Running on Metal
$ cargo run --release --features metal -- --image ../assets/bike.jpg

> loaded image Tensor[dims 3, 384, 384; f32, metal:4294968307]
> several cyclists are riding down a road with cars in the background
```

![Leading group, Giro d'Italia 2021](../assets/bike.jpg)
