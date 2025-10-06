# candle-based

Experimental, not instruction-tuned small LLM from the Hazy Research group, combining local and linear attention layers.

* [Blogpost](https://hazyresearch.stanford.edu/blog/2024-03-03-based)
* [Simple linear attention language models balance the recall-throughput tradeoff](https://arxiv.org/abs/2402.18668)

## Running an example


### Running on CPU

> candle-based is currently only supported on CPU.

```bash
$ cargo run --release -- \
  --prompt "Flying monkeys are" --which 1b-50b --sample-len 100

Args { cpu: false, tracing: false, prompt: "Flying monkeys are", temperature: None, top_p: None, seed: 299792458, sample_len: 100, model_id: None, revision: "refs/pr/1", config_file: None, tokenizer_file: None, weight_files: None, repeat_penalty: 1.1, repeat_last_n: 64, which: W1b50b }
avx: false, neon: true, simd128: false, f16c: false
temp: 0.00 repeat-penalty: 1.10 repeat-last-n: 64
Running on CPU, to run on GPU(metal), build this example with `--features metal`
loaded the model in 805.738542ms
Flying monkeys are a common sight in the wild, but they are also a threat to humans.

The new study, published today (July 31) in the journal Science Advances, shows that the monkeys are using their brains to solve the problem of how to get around the problem.

"We found that the monkeys were using a strategy called 'cognitive mapping' - they would use their brains to map out the route ahead," says lead author Dr. David J. Smith from the University of California
100 tokens generated (4.40 token/s)
```