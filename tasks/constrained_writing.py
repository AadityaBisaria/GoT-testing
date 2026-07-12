"""Constrained writing: LLM writes a paragraph, code checks the constraints.

Verification asymmetry: code cannot write the paragraph, but it can check the
constraints perfectly and for free — so refinement fires only on violations,
with the exact violations fed back. A judge ensemble breaks quality ties.
"""
import re

from got.operations import (ConditionalRefine, Generate, JudgeEnsemble,
                            KeepBest, Refine, Score)

from .base import Task

WORD_POOL = ["lantern", "harbor", "velvet", "compass", "ember", "meadow",
             "granite", "whisper", "orchard", "tide", "sparrow", "amber"]

TOPICS = ["an early morning walk", "a small coastal town", "an old workshop",
          "a train journey", "a night market", "a library in winter"]


class ConstrainedWritingTask(Task):
    name = "constrained_writing"

    def __init__(self, max_words=60):
        self.max_words = max_words
        self.n_sentences = 3

    def make_instance(self, rng):
        required = rng.sample(WORD_POOL, 3)
        topic = rng.choice(TOPICS)
        content = {"required": required, "sentences": self.n_sentences,
                   "max_words": self.max_words, "topic": topic, "text": ""}
        return content, None  # no ground truth: evaluate = constraint violations

    # --- constraint checker (the free verifier) ---
    def violations(self, content):
        text = content.get("text", "") if isinstance(content, dict) else ""
        if not text.strip():
            return ["empty text"]
        v = []
        low = text.lower()
        for w in content["required"]:
            if w.lower() not in low:
                v.append(f"missing required word '{w}'")
        n_sent = len([s for s in re.split(r"[.!?]+", text) if s.strip()])
        if n_sent != content["sentences"]:
            v.append(f"has {n_sent} sentences, needs exactly {content['sentences']}")
        n_words = len(text.split())
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
        return isinstance(content, dict) and bool(content.get("text", "").strip())

    def is_complete(self, content, root_content):
        # any non-empty paragraph is a full-problem answer. Note: a paragraph with
        # zero constraint violations early-exits BEFORE the judge ensemble — the
        # deliberate cost/quality trade of the hybrid pattern (constraints are the
        # solve criterion; the judge only breaks ties among violating candidates).
        return self.validate(content)

    # --- prompts ---
    def _rules(self, content):
        return (
            f"Write about {content['topic']}. Rules:\n"
            f"- exactly {content['sentences']} sentences\n"
            f"- at most {content['max_words']} words total\n"
            f"- must include the words: {', '.join(content['required'])}\n"
            "Put the final paragraph between <answer> and </answer> tags."
        )

    def io_prompt(self, content):
        return self._rules(content)

    def cot_prompt(self, content):
        return (
            f"{self._rules(content)}\n"
            "First plan how to satisfy every rule, then write the paragraph."
        )

    def generate_prompt(self, content):
        return self.io_prompt(content)

    def refine_prompt(self, content):
        # code-generated feedback: the exact violations
        return (
            f"This paragraph violates some rules.\n"
            f"Paragraph: {content.get('text', '')}\n"
            f"Violations: {'; '.join(self.violations(content))}\n"
            f"{self._rules(content)}\nRewrite it fixing every violation."
        )

    def judge_prompt(self, content, style=0):
        styles = [
            "You are a strict literary editor. Rate the prose quality",
            "You are an average reader. Rate how enjoyable this paragraph is",
            "You are a writing teacher. Rate the clarity and flow",
        ]
        return (
            f"{styles[style % len(styles)]} of the following paragraph on a scale "
            f"of 0-10. Reply with only the number.\n\nParagraph: {content.get('text', '')}"
        )

    # --- parsing ---
    def parse(self, text, fallback=None):
        base = fallback if isinstance(fallback, dict) else {}
        m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
        para = m.group(1).strip() if m else text.strip()
        if not para:
            return fallback
        return {**base, "text": para}

    # --- operation graphs ---
    def hybrid_operations(self):
        return [
            Generate(k=3, lazy=True),
            Score(),
            KeepBest(1),
            ConditionalRefine(attempts=2),  # only fires on violations, with exact feedback
            JudgeEnsemble(n=3, weight=0.1),
            KeepBest(1),
        ]

    def got_operations(self):
        return [
            Generate(k=3),
            Score(),
            KeepBest(2),
            Refine(attempts=2),  # unconditional refine, no code feedback loop
            Score(),
            JudgeEnsemble(n=3, weight=0.1),
            KeepBest(1),
        ]
