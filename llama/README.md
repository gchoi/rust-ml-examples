# candle-llama

Candle implementations of various Llama based architectures.

## Running an example

```bash
# CUDA
$ cargo run --release --features cuda -- \
  --prompt "Machine learning is " \
  --which v32-3b-instruct

# Metal
$ cargo run --release --features metal -- \
  --prompt "Machine learning is " \
  --which v32-3b-instruct

> Machine learning is  the part of computer science which deals with the development of algorithms and
```