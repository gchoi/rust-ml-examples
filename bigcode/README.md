# candle-starcoder: code generation model

[StarCoder/BigCode](https://huggingface.co/bigcode/starcoderbase-1b) is a LLM
model specialized to code generation. The initial model was trained on 80
programming languages.

## Running some examples

```bash
# Running on CPU
$ cargo run --release -- --prompt "fn fact(n: u64) -> u64 "

# Running on CUDA
$ cargo run --release --features cuda -- --prompt "fn fact(n: u64) -> u64 "

# Running on Metal
$ cargo run --release --features metal -- --prompt "fn fact(n: u64) -> u64 "

> retrieved the files in 422.709µs
> loaded the model in 883.122791ms
> starting the inference loop
> fn fact(n: u64) -> u64  {
>     if n == 0 {
>         1
>     } else {
>         n * fact(n - 1)
>     }
> }
> 
> fn main() {
>     let mut n = 0;
>     let mut sum = 0;
>     let mut i = 0;
>     let mut j = 0;
>     let mut k = 0;
>     let mut l = 0;
>     let mut m = 0;
>     let mut n_max = 0;100 tokens generated (12.048 token/s)
```
