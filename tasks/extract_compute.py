"""Extract-then-compute task: LLM extracts values from a passage (fuzzy),
arithmetic over them is computed by code in the hybrid method vs by the LLM
in io/cot/got. Measures the hallucination drop from delegating computation.
"""
import json
import re

from got.operations import ConditionalRefine, Execute, Generate, KeepBest, Score

from .base import Task

NAMES = ["Acme", "Bolt", "Crux", "Delta", "Ember", "Flint", "Gale", "Hollow",
         "Iris", "Juno", "Krait", "Lumen"]

FACT_TEMPLATES = [
    "{n} reported revenue of {v} thousand dollars this quarter.",
    "The {n} division brought in {v} thousand dollars.",
    "Analysts noted {n} earned {v} thousand dollars.",
    "{n} posted {v} thousand dollars in sales.",
]

FILLER = [
    "Markets were volatile throughout the period.",
    "Several executives commented on supply chain issues.",
    "The overall outlook remains uncertain according to observers.",
    "Regional trends varied considerably across segments.",
]


class ExtractComputeTask(Task):
    name = "extract_compute"

    def __init__(self, facts=6):
        self.facts = max(4, facts)

    def make_instance(self, rng):
        names = rng.sample(NAMES, self.facts)
        values = {n: rng.randint(11, 999) for n in names}
        sentences = [rng.choice(FACT_TEMPLATES).format(n=n, v=values[n]) for n in names]
        sentences += [rng.choice(FILLER) for _ in range(self.facts // 2)]
        rng.shuffle(sentences)
        # expression: sum of 2-3 entities minus 1
        plus = rng.sample(names, min(3, len(names) - 1))
        minus = rng.choice([n for n in names if n not in plus])
        expr = [["+", n] for n in plus] + [["-", minus]]
        gt = sum(values[n] for n in plus) - values[minus]
        return {"text": " ".join(sentences), "expr": expr}, gt

    # --- helpers ---
    def _question(self, expr):
        plus = [n for s, n in expr if s == "+"]
        minus = [n for s, n in expr if s == "-"]
        q = " plus ".join(plus)
        for m in minus:
            q += f" minus {m}"
        return f"What is the revenue of {q} (in thousand dollars)?"

    @staticmethod
    def compute(content):
        values = content.get("values", {})
        try:
            ans = sum((1 if s == "+" else -1) * int(values[n]) for s, n in content["expr"])
        except (KeyError, TypeError, ValueError):
            ans = None
        return {**content, "answer": ans}

    # --- prompts ---
    def io_prompt(self, content):
        return (
            f"Text: {content['text']}\n\n{self._question(content['expr'])}\n"
            'Output only a JSON object like {"answer": 123}.'
        )

    def cot_prompt(self, content):
        return (
            f"Text: {content['text']}\n\n{self._question(content['expr'])}\n"
            "First find each company's revenue in the text, then do the arithmetic "
            'step by step. End your answer with a JSON object like {"answer": 123}.'
        )

    def generate_prompt(self, content):
        if "values" in content:
            # compute stage (pure-LLM variant): arithmetic over extracted values
            return (
                f"Values: {json.dumps(content['values'])}\n"
                f"Compute: {self._question(content['expr'])}\n"
                'End your answer with a JSON object like {"answer": 123}.'
            )
        entities = [n for _, n in content["expr"]]
        return (
            f"Text: {content['text']}\n\n"
            f"Extract the revenue (in thousand dollars) of each of these companies: {entities}.\n"
            'End your answer with a JSON object mapping company name to number, e.g. {"Acme": 120}.'
        )

    def refine_prompt(self, content):
        return (
            f"Text: {content['text']}\n"
            f"Extracted values: {json.dumps(content.get('values', {}))}\n"
            "Some of these values may be wrong. Recheck each against the text. "
            'End your answer with the corrected JSON object mapping company name to number.'
        )

    # --- parsing / scoring ---
    def parse(self, text, fallback=None):
        matches = re.findall(r"\{[^{}]*\}", text)
        base = fallback if isinstance(fallback, dict) else {}
        for m in reversed(matches):
            try:
                obj = json.loads(m)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            if "answer" in obj:
                try:
                    return {**base, "answer": int(obj["answer"])}
                except (TypeError, ValueError):
                    continue
            nums = {k: int(v) for k, v in obj.items()
                    if isinstance(v, (int, float)) or (isinstance(v, str) and v.strip().isdigit())}
            if nums:
                return {**base, "values": nums}
        return fallback

    def validate(self, content):
        return isinstance(content, dict) and ("values" in content or "answer" in content)

    def is_complete(self, content, root_content):
        return isinstance(content, dict) and content.get("answer") is not None

    def score(self, content):
        # round-trip check: each extracted value must literally appear in the text
        if not isinstance(content, dict):
            return float("inf")
        if "values" not in content:
            return float("inf") if "answer" not in content else 0
        text = content.get("text", "")
        entities = [n for _, n in content.get("expr", [])]
        err = 0
        for n in entities:
            v = content["values"].get(n)
            if v is None or not re.search(rf"\b{v}\b", text):
                err += 1
        return err

    def evaluate(self, content, ground_truth):
        ans = content.get("answer") if isinstance(content, dict) else None
        return 0 if ans == ground_truth else 1

    # --- operation graphs ---
    def hybrid_operations(self):
        """LLM extracts (round-trip-validated), code does the arithmetic."""
        return [
            Generate(k=3, lazy=True),
            KeepBest(1),
            ConditionalRefine(attempts=2),
            Execute(self.compute, name="code-arith"),
        ]

    def tot_operations(self):
        # the default beam search never reaches a compute stage; reuse got's graph
        return self.got_operations()

    def got_operations(self):
        """Same graph but the LLM also does the arithmetic (compute stage)."""
        return [
            Generate(k=3),
            KeepBest(1),
            ConditionalRefine(attempts=2),
            Generate(k=1),  # compute stage: content now has "values"
            KeepBest(1),
        ]
