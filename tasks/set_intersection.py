"""Set intersection task: find common elements of two lists."""
import json

from got.operations import (Aggregate, ConditionalRefine, Execute, Generate,
                            KeepBest, KeepBestPerParent, Refine, Score, Split)

from .base import Task, extract_json_list


class SetIntersectionTask(Task):
    name = "set_intersection"

    def __init__(self, size=32):
        self.size = size

    def make_instance(self, rng):
        pool = list(range(0, self.size * 4))
        rng.shuffle(pool)
        a = sorted(pool[: self.size])
        rng.shuffle(pool)
        b = sorted(pool[: self.size])
        content = {"A": a, "B": b}
        return content, sorted(set(a) & set(b))

    def evaluate(self, content, ground_truth):
        ans = content.get("result") if isinstance(content, dict) else content
        if not isinstance(ans, list):
            return len(ground_truth) + 1
        return len(set(ans) ^ set(ground_truth))

    def io_prompt(self, content):
        return (
            f"Find the intersection (common elements) of these two sets:\n"
            f"Set A: {json.dumps(content['A'])}\nSet B: {json.dumps(content['B'])}\n"
            "Output only the intersection as a sorted JSON array, nothing else."
        )

    def cot_prompt(self, content):
        return (
            f"Find the intersection of these two sets:\n"
            f"Set A: {json.dumps(content['A'])}\nSet B: {json.dumps(content['B'])}\n"
            "Go through Set A element by element and check whether it appears in Set B. "
            "End your answer with the intersection as a sorted JSON array."
        )

    def aggregate_prompt(self, a, b):
        ra = a.get("result", []) if isinstance(a, dict) else a
        rb = b.get("result", []) if isinstance(b, dict) else b
        return (
            f"Combine these two partial intersection results into one sorted list with "
            f"no duplicates:\nPart 1: {json.dumps(ra)}\nPart 2: {json.dumps(rb)}\n"
            "End your answer with the combined list as a JSON array."
        )

    def refine_prompt(self, content):
        return (
            f"Set A: {json.dumps(content['A'])}\nSet B: {json.dumps(content['B'])}\n"
            f"Proposed intersection: {json.dumps(content.get('result', []))}\n"
            "Check the proposed intersection: remove any element not present in BOTH "
            "sets, and add any missing common elements. "
            "End your answer with the corrected intersection as a sorted JSON array."
        )

    def parse(self, text, fallback=None):
        lst = extract_json_list(text)
        if lst is not None and all(isinstance(x, (int, float)) for x in lst):
            result = sorted({int(x) for x in lst})
            base = fallback if isinstance(fallback, dict) else {}
            return {**base, "result": result}
        return fallback

    def score(self, content):
        # error estimate: proposed elements not actually in both sets, plus missed ones
        if not isinstance(content, dict) or "result" not in content:
            return float("inf")
        a, b = set(content.get("A", [])), set(content.get("B", []))
        res = set(content["result"])
        if not a or not b:
            return 0  # partial thought without full context; can't judge
        true = a & b
        return len(res ^ true)

    def split(self, content, n):
        # split Set A into chunks; each sub-problem intersects a chunk with full B
        a, b = content["A"], content["B"]
        k = max(1, len(a) // n)
        chunks = [a[i : i + k] for i in range(0, len(a), k)][:n] or [a]
        return [{"A": c, "B": b} for c in chunks]

    def validate(self, content):
        return isinstance(content, dict) and "result" in content

    def is_complete(self, content, root_content):
        return (isinstance(content, dict) and "result" in content
                and sorted(content.get("A", [])) == sorted(root_content.get("A", [])))

    def merge_naive(self, a, b):
        # bookkeeping only: union of partial results (the per-chunk intersection
        # work was done by the LLM; combining disjoint partial results is trivial)
        ra = a.get("result", []) if isinstance(a, dict) else (a or [])
        rb = b.get("result", []) if isinstance(b, dict) else (b or [])
        merged_a = (a.get("A", []) if isinstance(a, dict) else []) + (
            b.get("A", []) if isinstance(b, dict) else []
        )
        base_b = a.get("B", []) if isinstance(a, dict) else []
        return {"A": merged_a, "B": base_b, "result": sorted(set(ra) | set(rb))}

    @staticmethod
    def _code_union(parts):
        merged_a, result = [], set()
        b = parts[0].get("B", []) if parts else []
        for p in parts:
            merged_a += p.get("A", [])
            result |= set(p.get("result", []))
        return {"A": merged_a, "B": b, "result": sorted(result)}

    def hybrid_operations(self):
        """LLM intersects each chunk of A with B (fuzzy work); code unions the
        partial results (bookkeeping)."""
        return [
            Split(n=4),
            Generate(k=3, lazy=True),
            KeepBestPerParent(),
            Execute(self._code_union, name="code-union", merge=True),
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
