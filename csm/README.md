# Conversational Speech Model (CSM)

CSM is a speech generation model from Sesame,
[SesameAILabs/csm](https://github.com/SesameAILabs/csm).

It can generate a conversational speech between two different speakers.
The speakers turn are delimited by the `|` character in the prompt.

Hugging Face Login:

```bash
$ hf auth login

> Enter your HuggingFace token
```

```bash
# CPU
$ cargo run --release -- \
  --voices ../assets/voices.safetensors  \
  --prompt "Hey how are you doing?|Pretty good, pretty good. How about you?"

# CUDA
$ cargo run --features cuda --release -- \
  --voices ../assets/voices.safetensors  \
  --prompt "Hey how are you doing?|Pretty good, pretty good. How about you?"

# Metal
$ cargo run --features metal --release -- \
  --voices ../assets/voices.safetensors  \
  --prompt "Hey how are you doing?|Pretty good, pretty good. How about you?"
```
