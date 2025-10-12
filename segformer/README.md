# candle-segformer

- [HuggingFace Segformer Model Card][segformer]
- [`mit-b0` - An encoder only pretrained model][encoder]
- [`segformer-b0-finetuned-ade-512-512` - A fine tuned model for segmentation][ade512]

## How to run the example

If you want you can use the example images from this [pull request][pr], download them and supply the path to the image as an argument to the example.

```bash
# -- run the image classification task
# CPU
$ cargo run --release -- \
  classify ../assets/bike.jpg

# CUDA
$ cargo run --release --features cuda -- \
  classify ../assets/bike.jpg

# Metal
$ cargo run --release --features metal -- \
  classify ../assets/bike.jpg

# -- run the segmentation task
# CPU
$ cargo run --release -- \
  segment ../assets/bike.jpg --output-path ./output/output.jpg

# CUDA
$ cargo run --release --features cuda -- \
  segment ../assets/bike.jpg --output-path ./output/output.jpg

# Metal
$ cargo run --release --features metal -- \
  segment ../assets/bike.jpg --output-path ./output/output.jpg
```

Example output for classification:

```text
classification logits [3.275261e-5, 0.0008562019, 0.0008868563, 0.9977506, 0.0002465068, 0.0002241473, 2.846596e-6]
label: hamburger
```

[pr]: https://github.com/huggingface/candle/pull/1617
[segformer]: https://huggingface.co/docs/transformers/model_doc/segformer
[encoder]: https://huggingface.co/nvidia/mit-b0
[ade512]: https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512
