# candle-metavoice

MetaVoice-1B is a text-to-speech model trained on 100K hours of speech, more
details on the [model
card](https://huggingface.co/metavoiceio/metavoice-1B-v0.1).

Note that the current candle implementation suffers from some limitations as of
2024-03-02:
- The speaker embeddings are hardcoded.
- The generated audio file quality is weaker than the Python implementation,
  probably because of some implementation discrepancies.

## Run an example

```bash
# CUDA
$ cargo run --release --features cuda -- \
  --prompt "This is a demo of text to speech by MetaVoice-1B, an open-source foundational audio model."

# Metal
$ cargo run --release --features metal -- \
  --prompt "This is a demo of text to speech by MetaVoice-1B, an open-source foundational audio model."

$ cargo run --release --features metal -- \
  --prompt "Alex Choi is a very excellent research scientist. He is old, but remains young thinking."
```
