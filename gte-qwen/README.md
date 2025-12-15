# gte-Qwen1.5-7B-instruct

gte-Qwen1.5-7B-instruct is a variant of the GTE embedding model family.

- [Model card](https://huggingface.co/Alibaba-NLP/gte-Qwen1.5-7B-instruct) on the HuggingFace Hub.
- [Technical report](https://arxiv.org/abs/2308.03281) *Towards General Text Embeddings with Multi-stage Contrastive Learning*


## Running the example

Automatically download the model from the HuggingFace hub:
```bash
# CUDA
$ cargo run --release --features cuda

# Metal
$ cargo run --release --features metal
```

or, load the model from a local directory:
```bash
# CUDA
cargo run --release --features cuda -- --local-repo /path/to/gte_Qwen1.5-7B-instruct/

# Metal
cargo run --release --features metal -- --local-repo /path/to/gte_Qwen1.5-7B-instruct/
```
