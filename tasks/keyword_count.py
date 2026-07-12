"""Keyword counting task: count country mentions in a passage."""
import json
import re

from got.operations import (Aggregate, ConditionalRefine, Execute, Generate,
                            KeepBest, KeepBestPerParent, Refine, Score, Split)

from .base import Task

COUNTRIES = ["France", "Japan", "Brazil", "Canada", "Egypt", "India", "Norway", "Chile"]

TEMPLATES = [
    "Travelers from {c} often visit the coast.",
    "The delegation from {c} arrived on Tuesday.",
    "Exports to {c} rose sharply this year.",
    "A film festival in {c} drew large crowds.",
    "Scientists in {c} published a new study.",
    "The team from {c} won the qualifier.",
    "Cuisine from {c} is popular in the city.",
    "Students traveled to {c} for the exchange program.",
]


def extract_json_obj(text):
    matches = re.findall(r"\{[^{}]*\}", text)
    for m in reversed(matches):
        try:
            obj = json.loads(m)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


class KeywordCountTask(Task):
    name = "keyword_count"

    def __init__(self, sentences=24):
        self.sentences = sentences

    def make_instance(self, rng):
        parts = []
        counts = {c: 0 for c in COUNTRIES}
        for _ in range(self.sentences):
            c = rng.choice(COUNTRIES)
            t = rng.choice(TEMPLATES)
            parts.append(t.format(c=c))
            counts[c] += 1
        text = " ".join(parts)
        return {"text": text}, counts

    def evaluate(self, content, ground_truth):
        ans = content.get("counts") if isinstance(content, dict) else None
        if not isinstance(ans, dict):
            return sum(ground_truth.values())
        return sum(abs(ground_truth.get(c, 0) - int(ans.get(c, 0) or 0)) for c in COUNTRIES)

    def _count_instruction(self):
        return (
            f"Count how many times each of these countries is mentioned: {COUNTRIES}. "
            'End your answer with a JSON object mapping each country to its count, e.g. {"France": 2, ...}. '
            "Include every country, using 0 if absent."
        )

    def io_prompt(self, content):
        return f"Text: {content['text']}\n\n{self._count_instruction()}"

    def cot_prompt(self, content):
        return (
            f"Text: {content['text']}\n\nGo through the text sentence by sentence, "
            f"noting each country mention. Then total them. {self._count_instruction()}"
        )

    def aggregate_prompt(self, a, b):
        ca = a.get("counts", {}) if isinstance(a, dict) else {}
        cb = b.get("counts", {}) if isinstance(b, dict) else {}
        return (
            f"Add these two count dictionaries together key by key:\n"
            f"Counts 1: {json.dumps(ca)}\nCounts 2: {json.dumps(cb)}\n"
            "End your answer with the summed JSON object."
        )

    def refine_prompt(self, content):
        return (
            f"Text: {content['text']}\n"
            f"Proposed counts: {json.dumps(content.get('counts', {}))}\n"
            f"Recheck the counts against the text and correct any mistakes. "
            f"{self._count_instruction()}"
        )

    def parse(self, text, fallback=None):
        obj = extract_json_obj(text)
        if obj is not None:
            counts = {}
            for c in COUNTRIES:
                try:
                    counts[c] = int(obj.get(c, 0) or 0)
                except (TypeError, ValueError):
                    counts[c] = 0
            base = fallback if isinstance(fallback, dict) else {}
            return {**base, "counts": counts}
        return fallback

    def score(self, content):
        # ground-truth-free: compare claimed counts against a regex count of the text
        if not isinstance(content, dict) or "counts" not in content:
            return float("inf")
        text = content.get("text", "")
        if not text:
            return 0
        err = 0
        for c in COUNTRIES:
            actual = len(re.findall(re.escape(c), text))
            err += abs(actual - int(content["counts"].get(c, 0) or 0))
        return err

    def split(self, content, n):
        sents = re.split(r"(?<=\.)\s+", content["text"])
        k = max(1, len(sents) // n)
        chunks = [" ".join(sents[i : i + k]) for i in range(0, len(sents), k)][:n]
        return [{"text": ch} for ch in chunks if ch]

    def validate(self, content):
        return isinstance(content, dict) and "counts" in content

    def is_complete(self, content, root_content):
        return (isinstance(content, dict) and "counts" in content
                and content.get("text", "") == root_content.get("text", ""))

    def merge_naive(self, a, b):
        # bookkeeping only: summing per-chunk counts; the counting itself was LLM work
        # and the regex-recount score still penalizes wrong per-chunk counts
        ca = a.get("counts", {}) if isinstance(a, dict) else {}
        cb = b.get("counts", {}) if isinstance(b, dict) else {}
        text = " ".join(filter(None, [a.get("text", "") if isinstance(a, dict) else "",
                                      b.get("text", "") if isinstance(b, dict) else ""]))
        return {"text": text, "counts": {c: int(ca.get(c, 0) or 0) + int(cb.get(c, 0) or 0) for c in COUNTRIES}}

    @staticmethod
    def _code_sum(parts):
        text = " ".join(p.get("text", "") for p in parts)
        counts = {c: sum(int(p.get("counts", {}).get(c, 0) or 0) for p in parts) for c in COUNTRIES}
        return {"text": text, "counts": counts}

    def hybrid_operations(self):
        """LLM counts keywords per chunk (fuzzy work); code sums the dicts."""
        return [
            Split(n=4),
            Generate(k=3, lazy=True),
            KeepBestPerParent(),
            Execute(self._code_sum, name="code-sum", merge=True),
            Score(),
            ConditionalRefine(attempts=2),
            KeepBest(1),
        ]

    def got_operations(self):
        return [
            Split(n=4),
            Generate(k=3),
            KeepBestPerParent(),
            Aggregate(tries=3),
            Score(),
            Refine(attempts=2),
            Score(),
            KeepBest(1),
        ]
