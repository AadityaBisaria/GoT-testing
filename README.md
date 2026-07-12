# GoT-testing

A minimal, from-scratch implementation of Graph of Thoughts (GoT, [Besta et al. 2023](https://arxiv.org/abs/2308.09687)), benchmarked against plain IO prompting and Chain-of-Thought (CoT) on a local LLM served via [LM Studio](https://lmstudio.ai/).

## Setup

1. `pip install -r requirements.txt`
2. Pick an LLM backend (see below).

## LLM backend

`got/llm.py` provides a single `LLM` adapter that talks to either a local LM Studio server or the hosted OpenAI API — both via the same `openai` SDK, since LM Studio exposes an OpenAI-compatible endpoint.

**LM Studio (default, local, free)**
1. Start LM Studio's local server (default `http://localhost:1234/v1`) with a model loaded.
2. Run as normal — no flags needed. Override the base URL / model with `LMSTUDIO_BASE_URL` / `LMSTUDIO_MODEL` env vars if needed.

**OpenAI (hosted, needs an API key)**
```bash
export OPENAI_API_KEY=sk-...
python run.py --provider openai --model gpt-4o-mini --task sorting --method got --samples 5
```

Provider selection precedence: `--provider` CLI flag > `LLM_PROVIDER` env var > `lmstudio` (or `openai` if `OPENAI_API_KEY` is set and nothing else was specified). Model selection: `--model` flag > `LMSTUDIO_MODEL`/`OPENAI_MODEL` env var > provider default (LM Studio's currently loaded model, or `gpt-4o-mini`).

## Usage

```bash
python run.py --task sorting --method got --samples 5      # single task/method, default provider
python run.py --all --samples 5                             # all tasks x all methods
python run.py --all --provider openai --model gpt-4o-mini    # run against OpenAI instead
```

Results (mean error, solve rate, LLM calls, tokens, per-instance thought graphs) are written to `results/<task>_<method>.json`.

## Architecture

```
got/
  llm.py          provider-agnostic chat client (LM Studio / OpenAI)
  graph.py        Thought + ThoughtGraph (DAG of reasoning states)
  operations.py   Split, Generate(k, lazy), Aggregate(tries), Refine,
                  ConditionalRefine, Execute, JudgeEnsemble, Score,
                  KeepBest, KeepBestPerParent
  controller.py   executes a graph-of-operations; early-exits on a
                  verified-complete answer
  router.py       LLM router + whitelisted deterministic tool registry
tasks/
  sorting.py             sort a list of integers
  set_intersection.py    intersect two sets
  keyword_count.py       count keyword mentions in synthetic text
  extract_compute.py     extract values from text, compute arithmetic
  constrained_writing.py write a paragraph meeting code-checkable rules
  doc_qa.py              verifiable QA over real book excerpts (router demo)
  doc_merge.py           merge overlapping documents (GoT paper task 4)
  doc_summary.py         constraint-checked summarization of real text
run.py            benchmark CLI: 8 tasks x 5 methods (io/cot/tot/got/hybrid)
docs/             cached Project Gutenberg texts (Alice, Sherlock, Pride & Prejudice)
```

## Methods

- **io** — single direct prompt.
- **cot** — single chain-of-thought prompt.
- **tot** — Tree of Thoughts baseline: beam search (generate k → keep best b → refine).
- **got** — faithful Graph of Thoughts (Besta et al. 2023): split → best-of-N generate per chunk → pairwise LLM merge (best-of-N) → refine → keep best. Element-preservation scoring (dropped/duplicated elements are penalized, computed from parent thoughts without ground truth); unparseable outputs retry then fall back to a deliberately *non-solving* naive merge that scoring punishes.
- **hybrid** — neuro-symbolic GoT: the LLM does only the genuinely fuzzy steps; deterministic sub-steps run as code (`Execute`, 0 LLM calls). Refinement is conditional (fires only when the free programmatic check fails, with the exact violations fed back), candidate generation is lazy (stops at the first verified-correct candidate), and the controller early-exits once a complete answer verifies. Fuzzy-only quality (writing/merging) is scored by an odd-N LLM judge ensemble (median vote).

### The router (hybrid routing with a safety net)

`got/router.py` lets the LLM classify a sub-question as `deterministic(tool, args)` or `fuzzy` against a **whitelisted registry** (sort, sum, charset-guarded arithmetic, set ops, occurrence/word/sentence counts). Misclassification is asymmetric by design:

- a wrong deterministic route (unknown tool, invalid args, execution error) fails **loudly** and falls back to the fuzzy LLM path — it can waste a call, never produce a silent wrong answer;
- a wrong fuzzy route only costs tokens, never correctness.

The LLM never writes code and never relays large texts (code injects the reference text into tool args). Routing decisions and fallback reasons are recorded in `results/*.json`.

**Routing rule of thumb:** delegate to code anything that is cheaper to *verify* than to generate — even when code can't produce it (constrained writing: LLM drafts, code checks the constraints for free and drives refinement).

## Results

Model: `google/gemma-4-e4b` (LM Studio, Vulkan on AMD RX 9070 XT), 5 samples per task/method, size 32. Calls and tokens are totals across the 5 samples.

### Synthetic tasks

| Task | Method | Mean err | Solved | Calls | Tokens | Calls/solved |
|---|---|---|---|---|---|---|
| sorting | io | 0.40 | 60% | 5 | 4.8k | 1.7 |
| sorting | cot | 13.00 | 0% | 5 | 9.4k | – |
| sorting | tot | 0.20 | 80% | 22 | 31.5k | 5.5 |
| sorting | got | 1.40 | 40% | 120 | 121.9k | 60.0 |
| sorting | **hybrid** | **0.00** | **100%** | **20** | **4.1k** | **4.0** |
| set_intersection | io / cot | 0.00 | 100% | 5 | 7–9k | 1.0 |
| set_intersection | tot / hybrid | 0.00 | 100% | 20 | 13–29k | 4.0 |
| set_intersection | got | 0.00 | 100% | 105 | 50.8k | 21.0 |
| keyword_count | io | 1.60 | 80% | 5 | 7.3k | 1.2 |
| keyword_count | cot | 7.40 | 60% | 5 | 9.4k | 1.7 |
| keyword_count | tot | 0.00 | 100% | 20 | 29.6k | 4.0 |
| keyword_count | got | 0.00 | 100% | 105 | 72.0k | 21.0 |
| keyword_count | **hybrid** | **0.00** | **100%** | **20** | **14.0k** | **4.0** |
| extract_compute | io / cot / hybrid | 0.00 | 100% | 5 | 1.9–3k | 1.0 |
| extract_compute | tot / got | 0.00 | 100% | 20 | 8.4–8.7k | 4.0 |
| constrained_writing | io / cot / hybrid | 0.00 | 100% | 5 | 1.7–2.2k | 1.0 |
| constrained_writing | tot / got | 0.00 | 100% | 15–20 | 5–7k | 3–4 |

### Document tasks (real Gutenberg texts)

| Task | Method | Mean err | Solved | Calls | Tokens |
|---|---|---|---|---|---|
| doc_qa | io | 0.0 | 100% | 5 | 3.3k |
| doc_qa | cot / got | 0.2 | 80% | 5 / 15 | 3.8k / 9.2k |
| doc_qa | **hybrid (router)** | 0.2 | 80% | **6** | **3.7k** |
| doc_summary | io / cot / hybrid | 0.0 | 100% | 5 | ~4k |
| doc_summary | got | 0.0 | 100% | 15 | 12.5k |
| doc_merge | io | 3.75 | 20% | 5 | 13.3k |
| doc_merge | cot | 1.33 | 40% | 5 | 13.7k |
| doc_merge | got | **0.00** | **100%** | 22 | 45.6k |
| doc_merge | hybrid | 0.40 | 80% | 25 | 49.3k |

### Takeaways

- **Sorting is the headline.** Hybrid solved 5/5 at **4.1k tokens — cheaper than a single-call CoT (9.4k)** — because the LLM only sorts small chunks (terse prompts, lazy generation stops at the first verified sort) while code merges. Faithful GoT managed 40% at 30× the tokens: its pairwise LLM merges keep dropping elements even with best-of-3. Answer to "can hybrid beat pure LLM thinking on cost": yes, on tokens, when decomposition shrinks what the LLM must touch.
- **GoT's cost explosion is real**: 105–120 calls per 5 samples on split-aggregate tasks (best-of-3 at every node). ToT is the better pure-LLM structure here — 80–100% solved at ~20 calls. GoT-with-honest-scoring only won outright on **doc_merge** (100% vs everyone else ≤80%), the task closest to the paper's motivating use case: many interdependent fuzzy pieces.
- **The pattern by task difficulty**: model solves it in one shot → io wins (extract_compute, constrained_writing, doc_summary — all methods 100%, so cheapest wins). Model fails in one shot but the task decomposes with checkable sub-steps → **hybrid wins** (sorting, keyword_count). Task is fuzzy end-to-end with cross-piece dependencies → **got wins** (doc_merge).
- **CoT was the worst method on decomposable structured tasks** (0% on sorting — long reasoning chains degrade into inconsistent final lists; 60% on keyword_count). Step-by-step prose is not the same as verified decomposition.
- **Verification asymmetry is the highest-leverage trick**: code that can't produce the answer still checks it for free, which is what lets hybrid prune candidates (lazy generate), skip refinement (conditional refine), and exit pipelines early.
- **Router safety held**: no silent wrong answers from misrouting in doc_qa; a failed deterministic route falls back loudly to the LLM path (fallback reasons logged in `results/*.json`).

Full per-instance records (thought graphs, routing decisions, scores) are in `results/*.json`.
