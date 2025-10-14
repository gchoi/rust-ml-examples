# candle-clip

Contrastive Language-Image Pre-Training (CLIP) is an architecture trained on
pairs of images with related texts.

https://github.com/openai/CLIP

https://github.com/huggingface/transformers/tree/f6fa0f0bf0796ac66f201f23bdb8585de1609add/src/transformers/models/clip

## Running on an example on cpu

```bash
$ cargo run --release -- \
    --images "../assets/stable-diffusion-xl.jpg","../assets/bike.jpg" \
    --cpu \
    --sequences  "a cycling race","a photo of two cats","a robot holding a candle"

> softmax_image_vec: [2.1265574e-9, 4.0209294e-9, 1.0, 0.9999994, 5.752247e-7, 1.6381116e-9]
> 
> 
> Results for image: ../assets/stable-diffusion-xl.jpg
> 
> Probability: 0.0000% Text: a cycling race 
> Probability: 0.0000% Text: a photo of two cats 
> Probability: 100.0000% Text: a robot holding a candle 
> 
> 
> Results for image: ../assets/bike.jpg
> 
> Probability: 99.9999% Text: a cycling race 
> Probability: 0.0001% Text: a photo of two cats 
> Probability: 0.0000% Text: a robot holding a candle 
```

## Running on an example with cuda feature

```bash
$ cargo run --features cuda --release -- \
  --images "../assets/stable-diffusion-xl.jpg","../assets/bike.jpg" \
  --cpu \
  --sequences "a cycling race","a photo of two cats","a robot holding a candle"

> softmax_image_vec: [2.126675e-9, 4.0211976e-9, 1.0, 0.9999994, 5.752093e-7, 1.6380555e-9]
> 
> 
> Results for image: ../assets/stable-diffusion-xl.jpg
> 
> Probability: 0.0000% Text: a cycling race 
> Probability: 0.0000% Text: a photo of two cats 
> Probability: 100.0000% Text: a robot holding a candle 
> 
> 
> Results for image: ../assets/bike.jpg
> 
> Probability: 99.9999% Text: a cycling race 
> Probability: 0.0001% Text: a photo of two cats 
> Probability: 0.0000% Text: a robot holding a candle 
```

## Running on an example with metal feature (mac)

```bash
$ cargo run --features metal --release -- \
  --images "../assets/stable-diffusion-xl.jpg","../assets/bike.jpg" \
  --cpu \
  --sequences "a cycling race","a photo of two cats","a robot holding a candle"

> softmax_image_vec: [2.1265574e-9, 4.0209294e-9, 1.0, 0.9999994, 5.752247e-7, 1.6381116e-9]
> 
> 
> Results for image: ../assets/stable-diffusion-xl.jpg
> 
> Probability: 0.0000% Text: a cycling race 
> Probability: 0.0000% Text: a photo of two cats 
> Probability: 100.0000% Text: a robot holding a candle 
> 
> 
> Results for image: ../assets/bike.jpg
> 
> Probability: 99.9999% Text: a cycling race 
> Probability: 0.0001% Text: a photo of two cats 
> Probability: 0.0000% Text: a robot holding a candle 
```
