# candle-dinov2-reg4

[DINOv2-reg4](https://arxiv.org/abs/2309.16588) is the latest version of DINOv2 with registers.
In this example, it is used as an plant species classifier: the model returns the
probability for the image to belong to each of the 7806 PlantCLEF2024 categories.

## Running some examples

```bash
# Download classes names and a plant picture to identify
curl https://huggingface.co/vincent-espitalier/dino-v2-reg4-with-plantclef2024-weights/raw/main/species_id_mapping.txt --output ../assets/species_id_mapping.txt
curl https://bs.plantnet.org/image/o/bd2d3830ac3270218ba82fd24e2290becd01317c --output ../assets/bd2d3830ac3270218ba82fd24e2290becd01317c.jpg

# -- Perform inference
# CPU
$ cargo run --release -- \
  --image ../assets/bd2d3830ac3270218ba82fd24e2290becd01317c.jpg

# CUDA
$ cargo run --features cuda --release -- \
  --image ../assets/bd2d3830ac3270218ba82fd24e2290becd01317c.jpg

# Metal
$ cargo run --features metal --release -- \
  --image ../assets/bd2d3830ac3270218ba82fd24e2290becd01317c.jpg

> Orchis simia Lam.       : 45.55%
> Orchis × bergonii Nanteuil: 9.80%
> Orchis italica Poir.    : 9.66%
> Orchis × angusticruris Franch.: 2.76%
> Orchis × bivonae Tod.   : 2.54%

```

![Orchis Simia](https://bs.plantnet.org/image/o/bd2d3830ac3270218ba82fd24e2290becd01317c)
