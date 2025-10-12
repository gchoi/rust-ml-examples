# Using Hugging Face with Rust (ft. Axum Serving)

## Reference

---

[Using Hugging Face with Rust](https://www.shuttle.dev/blog/2024/05/01/using-huggingface-rust)

## Requirements

---

Create `.env` file:

```bash
$ cp example.env .env

# or use Just command runner
$ just create-env
```

Provide your Hugging Face API key in `.env` file.

```ini
HF_TOKEN={YOUR_HF_API_KEY}
```

## How to run

---

```bash
# CPU
$ cargo run --release

# CUDA
$ cargo run --release --features cuda

# Metal
$ cargo run --release --features metal
```

## Test for Axum Serving

---

```bash
# Simple test
curl http://127.0.0.1:8000/

# Prompt test
curl -X POST http://127.0.0.1:8000/prompt \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Where is Seoul?"}'
```