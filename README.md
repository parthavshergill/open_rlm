# Open RLM

This repository experiments with the RLM (Recursive Language Model) harness on long context benchmarks, evaluating how models perform when they can recursively query sub-LLMs to process and reason over extensive documents. The implementation supports multiple model backends including local models, Google Gemini, and OpenAI, enabling comparative analysis of different approaches to handling long-context tasks through recursive decomposition and sub-task delegation.

Based on the [Recursive Language Models blog post](https://alexzhang13.github.io/blog/2025/rlm/).

