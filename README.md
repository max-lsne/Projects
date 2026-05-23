# Learning neural networks from first principles

I'm working through Andrej Karpathy's "Zero to Hero" series to understand how neural networks actually work, from the math up. Karpathy recently joined Anthropic, but his YouTube series remains the best free resource for building this understanding from scratch.

This repo is my personal learning space. Everything here runs on my own machines and my own AWS account, separate from work.

## Why

I use AI tools daily at work (LLMs, dashboards, agents) but I wanted to understand what's happening underneath. How does backpropagation work? What is a gradient? Why do transformers work? The Karpathy series answers these questions by having you build everything yourself in Python and PyTorch.

## What's here

### Karpathy series

Working through the videos in order, coding along in my own notebooks:

1. **micrograd** (done) — built an autograd engine from scratch. Forward pass, backward pass, chain rule, gradient descent. ~100 lines of Python that implement backpropagation.
2. **makemore part 1** — bigram character-level language model (next)
3. **makemore part 2** — multilayer perceptron (MLP)
4. **makemore part 3** — activations, gradients, batchnorm
5. makemore parts 4-5, nanoGPT, tokenizer, GPT-2 reproduction (later)

### Wine quality project

After learning the MLP architecture from makemore part 2, I applied it to a real dataset: predicting wine quality from chemical properties. Trained on AWS SageMaker using a from-scratch PyTorch model (no sklearn, no frameworks). The model predicts quality within ±1 point 89% of the time.

## Setup

- Python 3.11 via `uv`
- PyTorch 2.12.0 (CPU)
- Local development on Windows with Kiro IDE
- AWS training on SageMaker notebook instance (personal account, `eu-north-1`)

## Resources

- [Karpathy Zero to Hero playlist](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
- [Karpathy's GitHub repo](https://github.com/karpathy/nn-zero-to-hero)
- [Wine quality dataset](https://huggingface.co/datasets/codesignal/wine-quality) (CC-BY 4.0)
