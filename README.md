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

Model: `google/gemma-4-e4b` (via LM Studio), 5 samples per task/method, size 32. Doc-task results below are final; the synthetic-task grid is being re-run after the fidelity fixes (earlier 3-sample results shown as v1 where noted).

### Document tasks (real Gutenberg texts)

| Task | Method | Mean err | Solved | Calls | Tokens |
|---|---|---|---|---|---|
| doc_qa | io | 0.0 | 100% | 5 | 3.3k |
| doc_qa | cot | 0.2 | 80% | 5 | 3.8k |
| doc_qa | got | 0.2 | 80% | 15 | 9.2k |
| doc_qa | **hybrid (router)** | 0.2 | 80% | **6** | **3.7k** |
| doc_summary | io | 0.0 | 100% | 5 | 4.2k |
| doc_summary | cot | 0.0 | 100% | 5 | 3.8k |
| doc_summary | got | 0.0 | 100% | 15 | 12.5k |
| doc_summary | **hybrid** | 0.0 | 100% | **5** | **4.2k** |
| doc_merge | io | 3.75 | 20% | 5 | 13.3k |

### v1 results (naive GoT, before fidelity fixes, 3 samples)

| Task | Method | Mean error | Solved | Calls |
|---|---|---|---|---|
| sorting | io | 1.33 | 0% | 3 |
| sorting | cot | 30.67 | 0% | 3 |
| sorting | got | 1.33 | 33% | 27 |
| set_intersection | io/cot/got | 0.00 | 100% | 3/3/24 |
| keyword_count | io | 1.67 | 0% | 3 |
| keyword_count | cot/got | 0.00 | 100% | 3/24 |

### Takeaways so far

- **GoT structure pays only on decomposable tasks that exceed single-shot capacity** (sorting): it was the only method to fully solve instances there. On tasks the model handles in one shot, GoT's 3–8x call overhead buys nothing — matching the paper.
- **The hybrid pattern collapses GoT's cost to ~CoT levels** while keeping its structure: 5–6 calls / ~4k tokens vs faithful GoT's 15 calls / 9–12.5k tokens on doc tasks. The savings come from three sources: deterministic merges cost 0 calls, refinement fires only on failed checks, and generation stops at the first verified candidate.
- **Cheating fallbacks flatter GoT.** v1's merge fallback silently called `sorted()` — code was solving the task for the LLM. The faithful version uses non-solving fallbacks penalized by scoring; the honest question then becomes *which sub-steps should code own openly* — which is exactly the hybrid method.
- **Verification asymmetry is the highest-leverage trick**: code that cannot write a constrained paragraph or summary still checks it perfectly for free, turning refinement from a blind extra pass into targeted feedback.

Full per-instance records (thought graphs, routing decisions, scores) are in `results/*.json`.
