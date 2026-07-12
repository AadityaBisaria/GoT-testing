# Benchmarks

Setup: `google/gemma-4-e4b` via LM Studio (Vulkan, AMD RX 9070 XT), 5 samples per task/method, `--size 32`, fixed seed 42. Calls/tokens are totals over the 5 samples. "Solved" = exact/zero-violation answer. Raw per-instance records (thought graphs, routing decisions, scores) live in `results/*.json`.

## Tasks

| Task | Problem (size 32) | Ground truth | Free (score) check |
|---|---|---|---|
| sorting | sort 32 ints | exact | inversions + element preservation vs parents |
| set_intersection | intersect two 32-elem sets | exact | result vs set math on carried A/B |
| keyword_count | count 8 country mentions in 32 sentences | exact | regex recount of carried text |
| extract_compute | extract 4 values from a report, arithmetic over 4 of them | exact | round-trip: value literally in text |
| constrained_writing | 3 sentences, ≤64 words, 3 required words | violations | code constraint checker |
| doc_qa | count/word questions on real book excerpts | computed | well-formedness only (why hybrid uses the router) |
| doc_merge | merge 3 overlapping doc variants | programmatic | sentence coverage + redundancy |
| doc_summary | ≤64-word summary naming key characters | violations | code constraint checker |

## Methods

io (1 call) · cot (1 call) · tot (beam search, ~4 calls/sample) · got (paper-faithful split/merge, 4–24 calls/sample) · hybrid (LLM for fuzzy steps, code for deterministic ones, conditional refinement; 1–5 calls/sample).

## Results — synthetic tasks

| Task | Method | Mean err | Solved | Calls | Tokens | Calls/solved |
|---|---|---|---|---|---|---|
| sorting | io | 0.40 | 60% | 5 | 4.8k | 1.7 |
| sorting | cot | 13.00 | 0% | 5 | 9.4k | – |
| sorting | tot | 0.20 | 80% | 22 | 31.5k | 5.5 |
| sorting | got | 1.40 | 40% | 120 | 121.9k | 60.0 |
| sorting | **hybrid** | **0.00** | **100%** | 20 | **4.1k** | 4.0 |
| set_intersection | io / cot | 0.00 | 100% | 5 | 7–9k | 1.0 |
| set_intersection | tot / hybrid | 0.00 | 100% | 20 | 13–29k | 4.0 |
| set_intersection | got | 0.00 | 100% | 105 | 50.8k | 21.0 |
| keyword_count | io | 1.60 | 80% | 5 | 7.3k | 1.2 |
| keyword_count | cot | 7.40 | 60% | 5 | 9.4k | 1.7 |
| keyword_count | tot | 0.00 | 100% | 20 | 29.6k | 4.0 |
| keyword_count | got | 0.00 | 100% | 105 | 72.0k | 21.0 |
| keyword_count | **hybrid** | **0.00** | **100%** | 20 | **14.0k** | 4.0 |
| extract_compute | io / cot / hybrid | 0.00 | 100% | 5 | 1.9–3k | 1.0 |
| extract_compute | tot / got | 0.00 | 100% | 20 | 8.4–8.7k | 4.0 |
| constrained_writing | io / cot / hybrid | 0.00 | 100% | 5 | 1.7–2.2k | 1.0 |
| constrained_writing | tot / got | 0.00 | 100% | 15–20 | 5–7k | 3–4 |

## Results — document tasks (real Gutenberg texts)

| Task | Method | Mean err | Solved | Calls | Tokens |
|---|---|---|---|---|---|
| doc_qa | io | 0.0 | 100% | 5 | 3.3k |
| doc_qa | cot / got | 0.2 | 80% | 5 / 15 | 3.8k / 9.2k |
| doc_qa | **hybrid (router)** | 0.2 | 80% | 6 | 3.7k |
| doc_summary | io / cot / hybrid | 0.0 | 100% | 5 | ~4k |
| doc_summary | got | 0.0 | 100% | 15 | 12.5k |
| doc_merge | io | 3.75 | 20% | 5 | 13.3k |
| doc_merge | cot | 1.33 | 40% | 5 | 13.7k |
| doc_merge | **got** | **0.00** | **100%** | 22 | 45.6k |
| doc_merge | hybrid | 0.40 | 80% | 25 | 49.3k |

## Findings

1. **Hybrid beat CoT on tokens for sorting** — 4.1k tokens for 5/5 solved vs CoT's 9.4k for 0/5. Decomposition shrank what the LLM touches (short chunk prompts, no reasoning prose) and lazy generation stopped paying once a chunk verified. This inverts the usual "structured methods cost more" assumption when sub-steps are code-checkable.
2. **Faithful GoT's cost is dominated by best-of-N LLM merges** (105–120 calls per 5 samples) and merges are also its failure mode (dropped elements → 40% on sorting despite 30x hybrid's tokens). Replacing exactly that step with code is the single biggest win.
3. **Three regimes, by task difficulty:**
   - one-shot-solvable → **io** wins (extract_compute, constrained_writing, doc_summary, set_intersection);
   - decomposable with checkable sub-steps → **hybrid** wins (sorting, keyword_count);
   - fuzzy end-to-end, interdependent pieces → **got** wins (doc_merge: 100% vs ≤80% for all others — the task shaped like the paper's motivating use case).
4. **CoT was the worst method on structured tasks** (0% sorting, 60% keyword_count): long reasoning chains degrade into inconsistent final answers. ToT is the stronger pure-LLM baseline (80–100% at ~20 calls).
5. **Router safety held**: deterministic misroutes fail loudly and fall back to the LLM; no silent wrong answers observed. One real bug class was found and fixed during testing: the LLM cannot be trusted to relay reference text into tool args — code must inject it.

## Caveats

- 5 samples per cell — directional, not statistically tight.
- Single small local model; ranking may shift with a stronger model (in particular, GoT's merge failures may vanish, narrowing hybrid's accuracy edge while its cost edge remains).
- doc_merge scoring is programmatic (sentence coverage/redundancy) + judge ensemble; it rewards verbatim-preserving merges, which is the task definition here but not the only reasonable one.
- Size 32 only for the final grid; GoT's structural advantage should grow at 64/128 (`--size` supports this; run overnight on a local model).

## Reproducing

```bash
python run.py --task sorting --method hybrid --samples 5 --size 32
python run.py --all --samples 5 --size 32          # full grid (hours on a local model)
python run.py --task doc_merge --method got --samples 5 --verbose
```
