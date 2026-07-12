# Architecture

A minimal, from-scratch Graph of Thoughts (GoT) framework ([Besta et al. 2023](https://arxiv.org/abs/2308.09687)) extended with a hybrid neuro-symbolic execution mode and an LLM router.

## Core concepts

**Thought** (`got/graph.py`) — one node of the reasoning DAG. Holds task-specific `content` (a list, a dict, a paragraph…), links to `parents`, the `operation` that produced it, and a `score` (an error estimate; lower is better, 0 = verified correct).

**ThoughtGraph** — the DAG plus a `frontier`: the set of thoughts the next operation acts on.

**Operation** (`got/operations.py`) — transforms the frontier. A reasoning method is just an ordered list of operations (the Graph-of-Operations).

**Controller** (`got/controller.py`) — creates the root thought from the problem input, runs each operation in order, and returns the best-scoring frontier thought. If any frontier thought is both *complete* (a full-problem answer, per `task.is_complete`) and *verified* (score 0), remaining operations are skipped (early exit).

**Task** (`tasks/base.py`) — bundles everything problem-specific: instance generation, prompts (io / cot / generate / aggregate / refine / judge), parsing (tolerant JSON extraction with fallback), programmatic scoring, validation, split/merge structure, and the operation graphs for each method (`got_operations`, `tot_operations`, `hybrid_operations`).

**LLM** (`got/llm.py`) — provider-agnostic chat client (LM Studio local server or OpenAI API via the same `openai` SDK) with retries and call/token counters.

## Operations

| Operation | LLM calls | Purpose |
|---|---|---|
| `Split(n)` | 0 | chop a thought into sub-problems (programmatic) |
| `Generate(k, lazy)` | ≤k per thought | sample k candidate solutions; `lazy=True` stops at the first candidate scoring 0 |
| `Aggregate(tries)` | ≤tries per merge | merge frontier thoughts pairwise; best-of-N candidates per merge; falls back to a deliberately *non-solving* naive merge (penalized by scoring) if nothing parses |
| `Refine(attempts)` | ≤attempts | ask the LLM to improve a thought; keep only if not worse |
| `ConditionalRefine(attempts)` | 0 if verified | refine **only** thoughts whose free programmatic check fails, feeding back the exact violations; stops as soon as the fix verifies |
| `Execute(fn, merge)` | 0 | deterministic code step (the hybrid pattern's workhorse) |
| `Score` | 0 | programmatic scoring of the frontier |
| `JudgeEnsemble(n, weight)` | n per thought | odd-N LLM judges with style-varied prompts; median vote converted to an error term (for fuzzy quality with no programmatic score) |
| `KeepBest(n)` / `KeepBestPerParent` | 0 | pruning |

## Scoring philosophy

Scores are **programmatic wherever possible** — small local models are unreliable judges of their own output. Two principles:

1. **Ground-truth-free but honest.** Sorting's score counts inversions *plus* the multiset difference against the thought's parents' elements — so a merge that drops elements is penalized even though the result "looks sorted". (v1 lacked this and let broken merges score 0.)
2. **No solving fallbacks.** When all LLM attempts are unparseable, the fallback does bookkeeping only (concatenation, count summing) — never `sorted()` or set math that would silently do the LLM's job and inflate results.

## Methods

- **io** — one direct prompt. **cot** — one chain-of-thought prompt.
- **tot** — Tree of Thoughts baseline: `Generate(4) → KeepBest(2) → Refine(2) → KeepBest(1)` beam search on the whole problem.
- **got** — paper-faithful GoT: split → best-of-3 generate per chunk → keep best per parent → pairwise best-of-3 LLM aggregate → refine → keep best.
- **hybrid** — same decomposition, but every deterministic sub-step is code:
  - `Execute` replaces LLM merges (heapq-merge sorted chunks, union partial intersections, sum count dicts) — 0 calls, 0 hallucination;
  - `Generate(lazy=True)` stops paying for candidates once one verifies;
  - `ConditionalRefine` only fires on failed checks, with the violations in the prompt;
  - controller early-exits on a verified complete answer.

## The router (`got/router.py`)

For tasks where sub-steps aren't known in advance (doc_qa), the LLM itself classifies each sub-question as `deterministic(tool, args)` or `fuzzy` — against a **whitelisted registry** (sort, sum, charset-guarded arithmetic, set ops, occurrence/word/sentence counting). The LLM never writes code.

Misclassification is **asymmetric by design**:

- *det-misroute* (unknown tool, invalid args, execution error, failed check) fails **loudly** → caught → falls back to the fuzzy LLM path. Costs one wasted call; can never produce a silent wrong answer.
- *fuzzy-misroute* (code could have done it) only costs tokens, never correctness.

Two hardening details found by testing: the arithmetic tool only accepts `[\d\s+\-*/().]` (a code-injection attempt falls back safely), and reference texts are **injected by code** into tool args — the LLM proved unreliable at copying text verbatim, which caused silent wrong counts until fixed.

**Routing rule of thumb:** delegate to code anything that is cheaper to *verify* than to generate — even when code can't produce it (constrained writing: LLM drafts, code checks constraints for free and drives refinement).

## Repository layout

```
got/          framework (graph, operations, controller, llm, router)
tasks/        8 tasks, each defining prompts, scoring, and operation graphs
data/         cached Project Gutenberg texts (downloaded on first use)
docs/         this documentation
results/      per-run JSON: summary + per-instance records incl. thought graphs
run.py        benchmark CLI
```
