# Snip-Cache

[![Paper](https://img.shields.io/badge/DOI-10.1016%2Fj.iot.2025.101852-0b7285)](https://doi.org/10.1016/j.iot.2025.101852)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)](https://www.python.org/)

**Snip-Cache is a code snippet caching system for LLM-based command-driven IoT systems.** It treats a natural-language command as a function call and caches reusable command-processing logic instead of a static prompt-response pair. This repository contains the research prototype and the Fluent Speech Commands (FSC) evaluation demo described in the associated paper.

> **Paper:** Chiwon Song and Sooyong Kang, “Snip-Cache: A code snippet caching system for LLM-based command-driven IoT systems,” *Internet of Things*, vol. 36, article 101852, 2026. [DOI](https://doi.org/10.1016/j.iot.2025.101852) · [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S254266052500366X)

## Why Snip-Cache?

Conventional prompt caches store an `(input, output)` pair. This works for repeated prompts but has limited reuse when semantically equivalent commands contain different expressions or argument values. Embedding-based semantic caches improve reuse, but a small wording difference such as “turn on” versus “turn off” can be critical in a device-control system.

Snip-Cache instead represents a command as an action plus arguments. A cache entry is created per action type and contains:

- an **Action Trigger Dictionary (ATD)** for matching expressions to an action;
- an **Action Argument Dictionary (AAD)** for normalizing parameter expressions;
- an **argument extractor snippet** for producing function arguments; and
- a **response generator snippet** for constructing the user-facing response.

This structure lets commands such as “Turn on the light in the bedroom” and “Switch on the radio in the kitchen” reuse the same action-level cache entry while preserving their different arguments.

## Processing flow

```text
Natural-language command
        |
        v
Action expression matching (ATD)
        |
        v
Argument extraction (AAD + generated snippet)
        |
        v
Native function execution
        |
        v
User response generation (generated snippet)
```

On a cache miss, the main LLM agent processes the command. Valid interactions are collected as samples, and after the configured sample threshold is reached, the snippet generator creates and validates an action-level cache entry. Later matching commands can be handled locally.

## Results reported in the paper

The paper evaluates Snip-Cache against GPTCache and vCache using 1,200 distinct command expressions derived from the FSC dataset. Each experiment was repeated five times.

| Metric | Reported result |
| --- | --- |
| Response accuracy | 99.92%–100% across sample sizes of 3, 6, and 9 |
| Cache-hit response time | 2.5 ms on average |
| End-to-end response-time reduction | 43%–47% compared with no cache |
| Cache hit ratio | 41.4% with 3 samples; up to 49.7% with 15 samples |
| Total token usage | 737K with 3 samples, compared with approximately 1.17M without caching |
| Cache footprint | Approximately 50–53 KB for six action types |

These values describe the paper's experimental setup and are not universal performance guarantees. Results depend on the workload, model version, API behavior, hardware, generated snippets, and policy configuration.

## Repository layout

```text
snip-cache/
|-- client/          # LLM client abstraction and OpenAI implementation
|-- core/            # Snip-Cache, generation logic, and cache policy
|-- fsc_demo/        # FSC-based dataset loader and mock IoT environment
|-- main_adapter/    # Interfaces for the main agent and function handler
|-- prompt/          # Prompts used to generate and validate cache entries
|-- utils/           # Text comparison and resource monitoring utilities
|-- demo.py          # End-to-end evaluation entry point
`-- requirements.txt
```

## Requirements

- Python 3.10 or later
- An OpenAI API key
- Internet access for OpenAI API calls and initial model downloads

The demo currently uses `gpt-4o` in `demo.py`. Running it incurs API usage and may produce different results as hosted models change.

## Quick start

1. Clone the repository and enter it:

   ```bash
   git clone https://github.com/hyu-splab/snip-cache.git
   cd snip-cache
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   ```

   On macOS or Linux:

   ```bash
   source .venv/bin/activate
   ```

   On Windows PowerShell:

   ```powershell
   .venv\Scripts\Activate.ps1
   ```

3. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Create a `.env` file in the repository root:

   ```dotenv
   OPENAI_API_KEY=YOUR_API_KEY
   ```

5. Run the demo:

   ```bash
   python demo.py
   ```

The default run evaluates all 1,200 commands with a sample threshold of three. Because commands are shuffled and snippets are generated by an LLM, runtime and results can vary between runs.

## Outputs

The default `python demo.py` run writes:

- `logs/session/test/test.log`: command-level results and the final summary;
- `logs/session/test/resource_usage.csv`: monitored resource usage; and
- `cache_store/test_cache.json`: generated Snip-Cache entries.

These runtime outputs are excluded from version control.

## Main integration points

| Component | Purpose |
| --- | --- |
| `core.snip_cache.SnipCache` | Cache lookup, learning, snippet execution, persistence |
| `core.policy.Policy` | Sample threshold, validation, retry, ambiguity, and background-generation policy |
| `core.snip_generator.SnippetGenerator` | LLM-assisted cache-entry generation and validation |
| `client.base.LLMClientBase` | Interface for a snippet-generation LLM client |
| `main_adapter.base_agent.BaseMainAgent` | Interface for the fallback command-processing agent |
| `main_adapter.base_function_handler.BaseFunctionHandler` | Interface for action specifications and argument validation |

`SnipCache.lookup(command, main_agent)` is the high-level path for cache-first processing with fallback learning. See `demo.py` for a complete assembly of functions, specifications, the main agent, policy, and snippet generator.

## Research prototype and security notice

This repository is a research prototype, not a production-ready IoT control system. Generated Python snippets are compiled and executed locally. Do not run cache entries or model-generated code from an untrusted source. A production deployment should add strong isolation, restricted execution, auditing, deterministic validation, and device-level authorization before executing generated logic or physical-device commands.

The current prototype focuses on single-step commands. Multi-step commands, context-dependent references, cache-population cost optimization, and continuously corrected dictionaries remain areas for future work, as discussed in the paper.

## Dataset

The included `fsc_demo/fluent_speech_commands_extend.csv` contains 1,200 commands across six action types: `activate`, `deactivate`, `increase`, `decrease`, `bring`, and `change language`. It is based on the [Fluent Speech Commands dataset](https://fluent.ai/fluent-speech-commands-a-dataset-for-spoken-language-understanding-research/) and includes expression-level enrichment prepared for the paper's evaluation.

Please review the terms of the upstream dataset separately. The repository's Apache-2.0 license applies to the software in this repository and does not replace third-party dataset terms.

## Citation

If you use Snip-Cache in research, please cite the paper:

```bibtex
@article{Song2026SnipCache,
  title   = {Snip-Cache: A code snippet caching system for LLM-based command-driven IoT systems},
  author  = {Song, Chiwon and Kang, Sooyong},
  journal = {Internet of Things},
  volume  = {36},
  pages   = {101852},
  year    = {2026},
  issn    = {2542-6605},
  doi     = {10.1016/j.iot.2025.101852},
  url     = {https://doi.org/10.1016/j.iot.2025.101852}
}
```

GitHub-compatible citation metadata is also available in [`CITATION.cff`](CITATION.cff), and machine-readable research-software metadata is provided in [`codemeta.json`](codemeta.json).

## License

Copyright 2025 Chiwon Song.

Licensed under the [Apache License 2.0](LICENSE).

## Keywords

LLM prompt caching, code snippet caching, snippet caching, semantic caching, command-driven IoT, natural-language device control, function calling, cache-assisted LLM inference, edge AI, intelligent prompt cache.
