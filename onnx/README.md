## Using ONNX models in Candle

This example demonstrates how to run [ONNX](https://github.com/onnx/onnx) based models in Candle.

It contains small variants of two models, [SqueezeNet](https://arxiv.org/pdf/1602.07360.pdf) (default) and [EfficientNet](https://arxiv.org/pdf/1905.11946.pdf).

You can run the examples with following commands:

```bash
# CPU
$ cargo run --release \
  -- --image ../assets/bike.jpg

# CUDA
$ cargo run --release --features cuda \
  -- --image ../assets/bike.jpg

# Metal
$ cargo run --release --features metal \
  -- --image ../assets/bike.jpg
```

Use the `--which` flag to specify explicitly which network to use, i.e.

```bash
# CPU
$ cargo run --release -- \
  --which squeeze-net --image ../assets/bike.jpg

# CUDA
$ cargo run --release --features cuda -- \
  --which squeeze-net --image ../assets/bike.jpg

# Metal
$ cargo run --release --features metal -- \
  --which squeeze-net --image ../assets/bike.jpg

    Finished release [optimized] target(s) in 0.21s
     Running `target/release/examples/onnx --which squeeze-net --image candle-examples/examples/yolo-v8/assets/bike.jpg`
loaded image Tensor[dims 3, 224, 224; f32]
unicycle, monocycle                               : 83.23%
ballplayer, baseball player                       : 3.68%
bearskin, busby, shako                            : 1.54%
military uniform                                  : 0.78%
cowboy hat, ten-gallon hat                        : 0.76%
```

```bash
# CPU
$ cargo run --release -- \
  --which efficient-net --image ../assets/bike.jpg

# CUDA
$ cargo run --release --features cuda -- \
  --which efficient-net --image ../assets/bike.jpg

# Metal
$ cargo run --release --features metal -- \
  --which efficient-net --image ../assets/bike.jpg

    Finished release [optimized] target(s) in 0.20s
     Running `target/release/examples/onnx --which efficient-net --image candle-examples/examples/yolo-v8/assets/bike.jpg`
loaded image Tensor[dims 224, 224, 3; f32]
bicycle-built-for-two, tandem bicycle, tandem     : 99.16%
mountain bike, all-terrain bike, off-roader       : 0.60%
unicycle, monocycle                               : 0.17%
crash helmet                                      : 0.02%
alp                                               : 0.02%
```

## Link to ONNX models

* [candle-onnx](https://huggingface.co/lmz/candle-onnx)
* [onnx/EfficientNet-Lite4](https://huggingface.co/onnx/EfficientNet-Lite4)
* [qualcomm/Real-ESRGAN-x4plus](https://huggingface.co/qualcomm/Real-ESRGAN-x4plus)