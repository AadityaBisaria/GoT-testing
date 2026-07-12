# GoT-testing

A minimal, from-scratch implementation of Graph of Thoughts (GoT, [Besta et al. 2023](https://arxiv.org/abs/2308.09687)), extended with a **hybrid neuro-symbolic mode** (LLM for fuzzy steps, deterministic code for checkable ones) and an **LLM router with a fail-loud fallback**, benchmarked across 8 tasks and 5 prompting methods on a local LLM served via [LM Studio](https://lmstudio.ai/).

**Docs:** [Architecture](docs/architecture.md) · [Benchmarks & findings](docs/benchmarks.md)

## Headline result

On sorting, the hybrid method solved **5/5 instances at 4.1k tokens — cheaper than a single CoT call (9.4k tokens, 0/5 solved)** and 30× cheaper than paper-faithful GoT (40% solved). Three regimes emerged across the 8 tasks:

| Task regime | Winner |
|---|---|
| Model solves it in one shot | **io** (cheapest) |
| Decomposable, sub-steps checkable by code | **hybrid** |
| Fuzzy end-to-end, interdependent pieces | **got** (doc_merge: 100% vs ≤80% all others) |

Full tables and analysis in [docs/benchmarks.md](docs/benchmarks.md).

## Setup

1. `pip install -r requirements.txt`
2. Pick an LLM backend:

**LM Studio (default, local, free)** — start LM Studio's server (default `http://localhost:1234/v1`) with a model loaded; no flags needed. Override with `LMSTUDIO_BASE_URL` / `LMSTUDIO_MODEL`.

**OpenAI (hosted)**
```bash
export OPENAI_API_KEY=sk-...
python run.py --provider openai --model gpt-4o-mini --task sorting --method hybrid
```

## Usage

```bash
python run.py --task sorting --method hybrid --samples 5 --size 32
python run.py --all --samples 5                    # 8 tasks x 5 methods (hours locally)
python run.py --task doc_merge --method got --verbose
```

Tasks: `sorting`, `set_intersection`, `keyword_count`, `extract_compute`, `constrained_writing`, `doc_qa`, `doc_merge`, `doc_summary` (doc tasks download real Project Gutenberg texts to `data/` on first use).
Methods: `io`, `cot`, `tot`, `got`, `hybrid`.

Results (summary + per-instance thought graphs, scores, routing decisions) are written to `results/<task>_<method>_<size>.json`.

## Layout

```
got/          framework: thought graph, operations, controller, LLM adapter, router
tasks/        8 task definitions (prompts, scoring, operation graphs)
docs/         architecture and benchmark documentation
data/         cached Gutenberg texts
results/      benchmark output JSON
run.py        benchmark CLI
```

See [docs/architecture.md](docs/architecture.md) for how the pieces fit together — including the scoring philosophy (programmatic, no solving fallbacks) and the router's misclassification-asymmetry design.
