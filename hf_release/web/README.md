---
title: 124M chat model, built from scratch
emoji: 💬
colorFrom: green
colorTo: gray
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: A from-scratch 124M chat model running in your browser
models:
- V1jey04/gpt2-124m-from-scratch
custom_headers:
  cross-origin-embedder-policy: credentialless
  cross-origin-opener-policy: same-origin
---

Chat with a 124M-parameter model whose tokenizer, transformer and training loop were written from
scratch. It runs in your browser with ONNX Runtime Web: an int8 copy of the weights (126 MB) is
downloaded once and cached, and nothing you type leaves your device.

Code: https://github.com/V1jey04/gpt2-124m-from-scratch
