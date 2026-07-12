"""Sorting task: sort a list of integers (GoT paper's canonical benchmark)."""
import json

from got.operations import (Aggregate, ConditionalRefine, Execute, Generate,
                            KeepBest, KeepBestPerParent, Refine, Score, Split)

from .base import Task, extract_json_list


class SortingTask(Task):
    name = "sorting"

    def __init__(self, length=32):
        self.length = length

    def make_instance(self, rng):
        nums = [rng.randint(0, 9) for _ in range(self.length)]
        return nums, sorted(nums)

    def evaluate(self, content, ground_truth):
        if not isinstance(content, list):
            return len(ground_truth)
        # misplaced elements + multiset mismatch
        errors = sum(1 for i in range(1, len(content)) if content[i - 1] > content[i])
        from collections import Counter
        diff = Counter(ground_truth) - Counter(content)
        diff2 = Counter(content) - Counter(ground_truth)
        return errors + sum(diff.values()) + sum(diff2.values())

    def io_prompt(self, content):
        return (
            f"Sort this list of numbers in ascending order: {json.dumps(content)}\n"
            "Output only the sorted list as a JSON array, nothing else."
        )

    def cot_prompt(self, content):
        return (
            f"Sort this list of numbers in ascending order: {json.dumps(content)}\n"
            "Think step by step: count how many of each digit there are, then build "
            "the sorted list. End your answer with the sorted list as a JSON array."
        )

    def aggregate_prompt(self, a, b):
        return (
            f"Merge these two sorted lists of numbers into one sorted list:\n"
            f"List A: {json.dumps(a)}\nList B: {json.dumps(b)}\n"
            "The result must contain every element of both lists exactly once, in "
            "ascending order. End your answer with the merged list as a JSON array."
        )

    def refine_prompt(self, content):
        return (
            f"This list should be sorted in ascending order but may contain mistakes: "
            f"{json.dumps(content)}\n"
            "Fix any out-of-order elements. Do not add or remove elements. "
            "End your answer with the corrected list as a JSON array."
        )

    def parse(self, text, fallback=None):
        lst = extract_json_list(text)
        if lst is not None and all(isinstance(x, (int, float)) for x in lst):
            return [int(x) for x in lst]
        return fallback

    def score(self, content):
        if not isinstance(content, list):
            return float("inf")
        return sum(1 for i in range(1, len(content)) if content[i - 1] > content[i])

    def score_thought(self, thought):
        """Inversions + multiset difference vs the elements this thought should contain.

        The expected multiset is recoverable without ground truth: it is the union
        of the thought's parents' elements (paper's element-preservation term).
        """
        if not isinstance(thought.content, list):
            return float("inf")
        err = self.score(thought.content)
        if thought.parents:
            from collections import Counter

            expected = Counter()
            for p in thought.parents:
                if isinstance(p.content, list):
                    expected += Counter(p.content)
            got = Counter(thought.content)
            diff = (expected - got) + (got - expected)
            err += sum(diff.values())
        return err

    def validate(self, content):
        return isinstance(content, list) and len(content) > 0

    def is_complete(self, content, root_content):
        from collections import Counter
        return isinstance(content, list) and Counter(content) == Counter(root_content)

    def split(self, content, n):
        k = max(1, len(content) // n)
        return [content[i : i + k] for i in range(0, len(content), k)][:n] or [content]

    def merge_naive(self, a, b):
        # non-solving: plain concatenation; inversions in the score penalize it
        return (a or []) + (b or [])

    def hybrid_operations(self):
        """LLM sorts chunks (the fuzzy work in this benchmark); code merges the
        already-sorted chunks (deterministic bookkeeping) and refinement only
        fires if the free correctness check fails."""
        import heapq

        return [
            Split(n=4),
            Generate(k=3, lazy=True),
            KeepBestPerParent(),
            Execute(lambda chunks: list(heapq.merge(*chunks)), name="code-merge", merge=True),
            Score(),
            ConditionalRefine(attempts=2),
            KeepBest(1),
        ]

    def got_operations(self):
        return [
            Split(n=4),
            Generate(k=3),        # 3 candidate sorts per chunk
            KeepBestPerParent(),  # keep the best sort of each chunk
            Aggregate(tries=3),   # merge pairwise, best-of-3 per merge
            Score(),
            Refine(attempts=3),
            Score(),
            KeepBest(1),
        ]
