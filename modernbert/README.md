# candle-modernbert

ModernBERT is a bidirectional encoder-only language model. In this example it is used for the fill-mask task:

## Usage

```bash
# CUDA
cargo run --release --features cuda -- \
  --model modern-bert-large \
  --prompt 'The capital of Korea is [MASK].'

# Metal
cargo run --release --features metal -- \
  --model modern-bert-large \
  --prompt 'The capital of Korea is [MASK].'
```

```markdown
Sentence: 1 : The capital of Korea is Seoul.
```
