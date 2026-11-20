---
license: mit
language:
- en
library_name: pytorch
pipeline_tag: text-generation
tags:
- from-scratch
- gpt2
- chat
- small-language-model
datasets:
- HuggingFaceFW/fineweb-edu
- HuggingFaceTB/smoltalk
- databricks/databricks-dolly-15k
- OpenAssistant/oasst1
model-index:
- name: gpt2-124m-from-scratch
  results:
  - task: {type: multiple-choice}
    dataset: {name: ARC-Easy, type: allenai/ai2_arc, config: ARC-Easy, split: test}
    metrics: [{type: accuracy, value: 40.9, name: zero-shot accuracy}]
  - task: {type: multiple-choice}
    dataset: {name: HellaSwag, type: Rowan/hellaswag, split: validation}
    metrics: [{type: accuracy, value: 28.4, name: zero-shot accuracy}]
  - task: {type: text-generation}
    dataset: {name: LAMBADA, type: EleutherAI/lambada_openai, split: test}
    metrics: [{type: accuracy, value: 20.7, name: zero-shot last-word accuracy}]
---

# gpt2-124m-from-scratch

A 124M-parameter chat model whose tokenizer, architecture, training loop and chat code were all
written from scratch, then trained on Kaggle's free T4 GPUs.

- **Code:** [github.com/V1jey04/gpt2-124m-from-scratch](https://github.com/V1jey04/gpt2-124m-from-scratch)

This is a custom PyTorch model, not a `transformers` model. Load it with the repo's own code (below).

## Model

| | |
|---|---|
| Parameters | 124,475,904 |
| Shape | 12 layers, 12 heads, 768 dimensions, 1,024-token context (GPT-2 small) |
| Embeddings | learned positions; input and output embeddings tied |
| Tokenizer | byte-level BPE, 50,000 pieces + 4 chat tokens, vocabulary padded to 50,304 |
| Chat format | `<|bos|><|user|>{question}<|end|><|assistant|>{answer}<|end|>` |
| Files | `model.safetensors` (fp32, tied weight stored once), `config.json`, `tokenizer.json` |

## Training

| Stage | Data | Steps | Compute | Validation loss |
|---|---|---|---|---|
| Pretraining | 2.42B tokens of FineWeb-Edu | 9,236 × 262,144 tokens | 2× T4, 22.6 h | 10.96 → 3.297 |
| Chat fine-tuning | 276M tokens of SmolTalk, Dolly, OpenAssistant | 4,217 × 65,536 tokens | 2× T4, 2 h 40 min | 2.74 → 1.445 |

Training used DDP, fp16 autocast with a gradient scaler, and `torch.compile`, at about 33k
tokens/s. Fine-tuning computes the loss on assistant tokens only.

## Evaluation

These are zero-shot results, with both models scored by the same script (`benchmark.py` in the
code repo). Multiple choice is scored by loss per byte. Note that this is the chat-tuned model
compared with GPT-2's base model.

| Test | This model | OpenAI GPT-2 124M |
|---|---|---|
| ARC-Easy (2,376 questions) | **40.9%** | 40.5% |
| HellaSwag (10,042) | 28.4% | **29.9%** |
| LAMBADA (5,153) | 20.7% | **33.7%** |
| WikiText-2 bits per byte (lower is better) | 1.324 | **1.020** |

## Use

```bash
git clone https://github.com/V1jey04/gpt2-124m-from-scratch && cd gpt2-124m-from-scratch
pip install torch safetensors huggingface_hub
hf download V1jey04/gpt2-124m-from-scratch --local-dir weights
python chat.py --ckpt weights/model.safetensors --tokenizer weights/tokenizer.json
```

## Limitations

A 124M model trained on 2.4B tokens is a learning project, not an assistant to rely on. It writes
fluent English and follows the chat format, but it often states wrong facts with confidence. In
one test it explained photosynthesis correctly, then claimed that glucose and oxygen are needed
"for the production of red blood cells". It has had no safety tuning, and it only knows English.

## Licences

Code and weights: MIT. Training data:
- FineWeb-Edu (ODC-By)
- SmolTalk (Apache 2.0)
- Databricks Dolly 15k (CC BY-SA 3.0)
- OpenAssistant oasst1 (Apache 2.0)

Please follow the dataset licences if you build on this model.
