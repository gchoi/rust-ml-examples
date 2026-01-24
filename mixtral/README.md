# candle-mixtral: 8x7b LLM using a sparse mixture of experts.

Mixtral-8x7B-v0.1 is a pretrained generative LLM with 56 billion parameters.

- [Blog post](https://mistral.ai/news/mixtral-of-experts/) from Mistral announcing the model release.
- [Model card](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1) on the HuggingFace Hub.

## Running the example

```bash
# CUDA
$ cargo run --release --features cuda  -- --prompt "def print_prime(n): "

# Metal
$ cargo run --release --features metal  -- --prompt "def print_prime(n): "
```

```python
def print_prime(n):  # n is the number of prime numbers to be printed
    i = 2
    count = 0
    while (count < n):
        if (isPrime(i)):
            print(i)
            count += 1
        i += 1

def isPrime(n):
    for x in range(2, int(n**0.5)+1):
        if (n % x == 0):
            ...
```
