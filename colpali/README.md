# Colpali

[HuggingFace Model Card](https://huggingface.co/vidore/colpali-v1.2-merged)

```bash
# CPU
$ cargo run --release -- \
  --prompt "What is Positional Encoding" \
  --pdf "../assets/1706.03762.pdf"

# CUDA
$ cargo run --release --features cuda -- \
  --prompt "What is Positional Encoding" \
  --pdf "../assets/1706.03762.pdf"

# Metal
$ cargo run --release --features metal -- \
  --prompt "What is Positional Encoding" \
  --pdf "../assets/1706.03762.pdf"

> Prompt: what is position encoding?
> top 3 page numbers that contain similarity to the prompt
> -----------------------------------
> Page: 6
> Page: 3
> Page: 2
> -----------------------------------
```