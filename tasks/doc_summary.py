"""Summarize a real book excerpt under code-checkable constraints.

Verification asymmetry on real text: code cannot write the summary but checks
the constraints (length cap, must mention the excerpt's key names) for free;
refinement fires only on violations, with the exact violations fed back.
"""
import re

from got.operations import (ConditionalRefine, Generate, JudgeEnsemble,
                            KeepBest, Refine, Score)

from .base import Task
from .doc_utils import paragraphs, proper_nouns


class DocSummaryTask(Task):
    name = "doc_summary"

    def __init__(self, n_paras=4, max_words=80):
        self.n_paras = max(2, n_paras)
        self.max_words = max_words

    def make_instance(self, rng):
        paras = paragraphs(rng.choice(["alice", "sherlock", "pride"]))
        start = rng.randrange(0, len(paras) - self.n_paras)
        text = " ".join(paras[start : start + self.n_paras])
        required = proper_nouns(text, top=4)
        content = {"text": text, "required": required,
                   "max_words": self.max_words, "summary": ""}
        return content, None  # evaluated by constraint violations

    # --- constraint checker ---
    def violations(self, content):
        s = content.get("summary", "")
        if not s.strip():
            return ["empty summary"]
        v = []
        low = s.lower()
        for w in content.get("required", []):
            if w.lower() not in low:
                v.append(f"missing required name '{w}'")
        n_words = len(s.split())
        if n_words > content["max_words"]:
            v.append(f"has {n_words} words, max {content['max_words']}")
        return v

    def evaluate(self, content, ground_truth):
        return len(self.violations(content)) if isinstance(content, dict) else 99

    def score(self, content):
        if not isinstance(content, dict):
            return float("inf")
        return len(self.violations(content))

    def validate(self, content):
        return isinstance(content, dict) and bool(content.get("summary", "").strip())

    def is_complete(self, content, root_content):
        return self.validate(content)

    # --- prompts ---
    def _rules(self, content):
        return (f"Summarize the text in at most {content['max_words']} words. "
                f"The summary must mention: {', '.join(content['required'])}. "
                "Put the summary between <answer> and </answer> tags.")

    def io_prompt(self, content):
        return f"Text: {content['text']}\n\n{self._rules(content)}"

    def cot_prompt(self, content):
        return (f"Text: {content['text']}\n\nIdentify the key events and the "
                f"required names, then write the summary. {self._rules(content)}")

    def generate_prompt(self, content):
        return self.io_prompt(content)

    def refine_prompt(self, content):
        return (f"Text: {content['text']}\n"
                f"Summary: {content.get('summary', '')}\n"
                f"Violations: {'; '.join(self.violations(content))}\n"
                f"{self._rules(content)}\nRewrite the summary fixing every violation.")

    def judge_prompt(self, content, style=0):
        styles = ["You are a strict editor. Rate how faithfully this summary reflects the text",
                  "You are a reader who knows the text. Rate the completeness of this summary",
                  "You are a writing teacher. Rate the clarity of this summary"]
        return (f"{styles[style % len(styles)]} on a scale of 0-10. Reply with only the "
                f"number.\n\nText: {content.get('text', '')[:1500]}\n\n"
                f"Summary: {content.get('summary', '')}")

    # --- parsing ---
    def parse(self, text, fallback=None):
        base = fallback if isinstance(fallback, dict) else {}
        m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
        s = m.group(1).strip() if m else text.strip()
        if not s:
            return fallback
        return {**base, "summary": s}

    # --- operation graphs ---
    def hybrid_operations(self):
        return [
            Generate(k=3, lazy=True),
            Score(),
            KeepBest(1),
            ConditionalRefine(attempts=2),
            JudgeEnsemble(n=3, weight=0.1),
            KeepBest(1),
        ]

    def got_operations(self):
        return [
            Generate(k=3),
            Score(),
            KeepBest(2),
            Refine(attempts=2),
            Score(),
            JudgeEnsemble(n=3, weight=0.1),
            KeepBest(1),
        ]
