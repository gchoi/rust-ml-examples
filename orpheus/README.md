# Orpheus

Orpheus is a 3B text-to-speech model based on Llama.

- Weights on HuggingFace
  [canopylabs/orpheus-3b-0.1-ft](https://huggingface.co/canopylabs/orpheus-3b-0.1-ft).
- Code on GitHub [canopyai/Orpheus-TTS](https://github.com/canopyai/Orpheus-TTS).


```bash
# CUDA
$ cargo run --release --features cuda

# Metal
$ cargo run --release --features metal

cargo run --release --features accelerate,metal
```