"""Document merging (the GoT paper's 4th task) over real book paragraphs.

Instance: 3 overlapping document variants built from a pool of paragraphs.
Merge them into one document that keeps every distinct paragraph once.
Scoring is programmatic: coverage (source sentences preserved) + redundancy
(duplicated sentences), plus a judge ensemble for fluency tie-breaks.
"""
import re

from got.operations import (Aggregate, ConditionalRefine, JudgeEnsemble,
                            KeepBest, Refine, Score, Split)

from .base import Task
from .doc_utils import paragraphs


def _norm_sentences(text):
    sents = [re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()
             for s in re.split(r"[.!?]+", text)]
    return [s for s in sents if len(s.split()) >= 4]


def _key(s, n=6):
    return " ".join(s.split()[:n])


class DocMergeTask(Task):
    name = "doc_merge"

    def __init__(self, pool=6):
        self.pool = max(4, pool)  # paragraphs in the shared pool

    def make_instance(self, rng):
        paras = paragraphs(rng.choice(["alice", "sherlock", "pride"]), max_words=80)
        start = rng.randrange(0, len(paras) - self.pool)
        pool = paras[start : start + self.pool]
        docs = []
        for _ in range(3):
            take = rng.sample(range(self.pool), rng.randint(self.pool - 2, self.pool - 1))
            docs.append(" ".join(pool[i] for i in sorted(take)))
        return {"docs": docs, "text": ""}, None  # scored programmatically, no exact GT

    # --- programmatic checker ---
    def _coverage_redundancy(self, content):
        merged = content.get("text", "")
        if not merged.strip():
            return None
        merged_keys = [_key(s) for s in _norm_sentences(merged)]
        merged_set = set(merged_keys)
        source_keys = set()
        for d in content.get("docs", []):
            source_keys |= {_key(s) for s in _norm_sentences(d)}
        missing = len(source_keys - merged_set)
        dup = len(merged_keys) - len(merged_set)
        return missing, dup

    def score(self, content):
        if not isinstance(content, dict):
            return float("inf")
        cr = self._coverage_redundancy(content)
        if cr is None:
            return float("inf") if content.get("docs") else 0
        missing, dup = cr
        return missing + dup

    def evaluate(self, content, ground_truth):
        return self.score(content) if isinstance(content, dict) else 99

    def validate(self, content):
        return isinstance(content, dict) and bool(content.get("text", "").strip())

    def is_complete(self, content, root_content):
        # complete once it merges ALL source docs (root carries the full list)
        return (self.validate(content)
                and len(content.get("docs", [])) >= len(root_content.get("docs", [])))

    # --- prompts ---
    def _instr(self):
        return ("Merge the documents into ONE document that contains every distinct "
                "paragraph exactly once, preserving the original wording. Remove "
                "duplicated paragraphs. Put the merged document between <answer> and </answer> tags.")

    def io_prompt(self, content):
        docs = "\n\n".join(f"Document {i+1}:\n{d}" for i, d in enumerate(content["docs"]))
        return f"{docs}\n\n{self._instr()}"

    def cot_prompt(self, content):
        return (f"{self.io_prompt(content)}\n"
                "First list which paragraphs appear in which documents, then merge.")

    def generate_prompt(self, content):
        return self.io_prompt(content)

    def aggregate_prompt(self, a, b):
        return (f"Document 1:\n{a.get('text') or ' '.join(a.get('docs', []))}\n\n"
                f"Document 2:\n{b.get('text') or ' '.join(b.get('docs', []))}\n\n"
                f"{self._instr()}")

    def refine_prompt(self, content):
        cr = self._coverage_redundancy(content)
        missing, dup = cr if cr else (0, 0)
        return (f"Merged document:\n{content.get('text', '')}\n\n"
                f"Source documents:\n" +
                "\n\n".join(content.get("docs", [])) +
                f"\n\nProblems: {missing} source sentences are missing; "
                f"{dup} sentences are duplicated. Fix both: include every distinct "
                "paragraph exactly once, original wording. "
                "Put the corrected document between <answer> and </answer> tags.")

    def judge_prompt(self, content, style=0):
        styles = ["You are a strict editor. Rate the coherence and ordering",
                  "You are a reader. Rate the readability",
                  "You are a proofreader. Rate the internal consistency"]
        return (f"{styles[style % len(styles)]} of this merged document on a scale of "
                f"0-10. Reply with only the number.\n\n{content.get('text', '')}")

    # --- parsing / structure ---
    def parse(self, text, fallback=None):
        base = fallback if isinstance(fallback, dict) else {}
        m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
        para = m.group(1).strip() if m else text.strip()
        if not para:
            return fallback
        return {**base, "text": para}

    def split(self, content, n):
        return [{"docs": [d], "text": d} for d in content["docs"]]

    def merge_naive(self, a, b):
        # bookkeeping: concatenate texts and union source-doc lists; duplicated
        # sentences remain and are penalized by the redundancy score
        return {"docs": a.get("docs", []) + b.get("docs", []),
                "text": (a.get("text", "") + " " + b.get("text", "")).strip()}

    # --- operation graphs ---
    def hybrid_operations(self):
        return [
            Split(n=3),
            Aggregate(tries=2),
            Score(),
            ConditionalRefine(attempts=2),  # fires only when code finds missing/dup
            JudgeEnsemble(n=3, weight=0.05),
            KeepBest(1),
        ]

    def got_operations(self):
        return [
            Split(n=3),
            Aggregate(tries=2),
            Score(),
            Refine(attempts=2),  # unconditional, no code feedback
            Score(),
            JudgeEnsemble(n=3, weight=0.05),
            KeepBest(1),
        ]
