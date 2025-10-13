## Using ONNX LLM models in Candle

This example demonstrates how to run [ONNX](https://github.com/onnx/onnx) based LLM models in Candle.

This script only implements SmolLM-135M right now.

You can run the examples with the following commands:

```bash
# CPU
$ cargo run --release -- --prompt "My favorite theorem is "

# CUDA
$ cargo run --release --features cuda -- --prompt "My favorite theorem is "
```