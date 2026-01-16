# LLllama2.c

Have you ever wanted to inference a baby Llama 2 model in pure C? No? Well, now you can!

Train the Llama 2 LLM architecture in PyTorch then inference it with one simple 700-line C file (run.c). You might think that you need many billion parameter LLMs to do anything useful, but in fact very small LLMs can have surprisingly strong performance if you make the domain narrow enough (ref: TinyStories paper). This repo is a "fullstack" train + inference solution for Llama 2 LLM, with focus on minimalism and simplicity.

As the architecture is identical, you can also load and inference Meta's Llama 2 models. However, the current code only inferences models in fp32, so you will most likely not be able to productively load models larger than 7B. Work on model quantization is currently ongoing.

Please note that this repo started recently as a fun weekend project: I took my earlier nanoGPT, tuned it to implement the Llama-2 architecture instead of GPT-2, and the meat of it was writing the C inference engine in run.c. So the project is young and moving quickly. Hat tip to the awesome llama.cpp for inspiring this project. Compared to llama.cpp, I wanted something super simple, minimal, and educational so I chose to hard-code the Llama 2 architecture and just roll one inference file of pure C with no dependencies.

[https://github.com/karpathy/llama2.c](https://github.com/karpathy/llama2.c)

## Run

```bash
# CUDA
$ cargo run --release --features cuda

# Metal
$ cargo run --release --features metal

loading the model weights from karpathy/tinyllamas
Config { dim: 288, hidden_dim: 768, n_layers: 6, n_heads: 6, n_kv_heads: 6, vocab_size: 32000, seq_len: 256, norm_eps: 1e-5 }
starting the inference loop
Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine and pick flowers. One day, she found a shiny rock that sparkled in the light. It was so pretty!
Lily showed her mommy the rock and said, "Look what I found! It's so pretty!" Her mommy smiled and said, "That's a special rock, Lily. It's called a diamond."
Later that day, Lily went to the park with her friends. They played on the swings and the slide. But then, Lily saw something shiny in the grass. She picked it up and showed it to her friends. "Look what I found!" she said.
Her friend said, "Wow, that's so cool! Can we keep it?" Lily thought for a moment and said, "I don't know if we should. It might belong to someone else." Her friend agreed and they decided to ask around. Finally, they found the owner of the diamond and returned it to him. He was very happy and thanked them for being honest. From that day on, Lily and her friends always looked out
256 tokens generated (58.59 token/s)
```
