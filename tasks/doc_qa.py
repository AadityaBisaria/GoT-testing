"""Verifiable QA over real book excerpts. Ground truth is computed
programmatically at instance creation, so hallucination is directly measurable.
The hybrid method uses the LLM Router: counting/arithmetic sub-questions go to
whitelisted code tools, everything else falls back to the LLM.
"""
import json
import re

from got.graph import Thought
from got.operations import Operation
from got.router import route_and_execute

from .base import Task
from .doc_utils import paragraphs, proper_nouns

QUESTION_KINDS = ["count_word", "word_count_para", "sum_counts"]


class RouteAnswer(Operation):
    """One routed step: LLM classifies the question, code or LLM answers it."""

    def run(self, graph, task, llm):
        new = []
        for t in graph.frontier:
            content = t.content
            ans, routing = route_and_execute(
                llm, task, content, content["question"], context_text=content["text"])
            th = graph.add(Thought({**content, "answer": ans, "routing": routing},
                                   parents=[t], operation=f"route-{routing['route']}"))
            th.score = task.score_thought(th)
            new.append(th)
        graph.frontier = new


class DocQATask(Task):
    name = "doc_qa"

    def __init__(self, n_paras=3):
        self.n_paras = max(2, n_paras)

    def make_instance(self, rng):
        paras = paragraphs(rng.choice(["alice", "sherlock", "pride"]))
        start = rng.randrange(0, len(paras) - self.n_paras)
        text = " ".join(paras[start : start + self.n_paras])
        nouns = proper_nouns(text, top=4) or ["the"]
        kind = rng.choice(QUESTION_KINDS)
        if kind == "count_word" or (kind == "sum_counts" and len(nouns) < 2):
            w = rng.choice(nouns)
            q = f'How many times does the word "{w}" appear in the text (case-insensitive)?'
            gt = len(re.findall(re.escape(w), text, re.IGNORECASE))
        elif kind == "word_count_para":
            q = "How many words does the text contain (split on whitespace)?"
            gt = len(text.split())
        else:  # sum_counts
            w1, w2 = rng.sample(nouns, 2)
            q = (f'How many times do the words "{w1}" and "{w2}" appear in the text '
                 f"in total (case-insensitive, sum of both counts)?")
            gt = (len(re.findall(re.escape(w1), text, re.IGNORECASE))
                  + len(re.findall(re.escape(w2), text, re.IGNORECASE)))
        return {"text": text, "question": q}, gt

    def evaluate(self, content, ground_truth):
        ans = content.get("answer") if isinstance(content, dict) else None
        try:
            return 0 if int(ans) == ground_truth else 1
        except (TypeError, ValueError):
            return 1

    # --- prompts ---
    def io_prompt(self, content):
        return (
            f"Text: {content['text']}\n\n{content['question']}\n"
            'Output only a JSON object like {"answer": 42}.'
        )

    def cot_prompt(self, content):
        return (
            f"Text: {content['text']}\n\n{content['question']}\n"
            "Work through the text carefully step by step, then answer. "
            'End your answer with a JSON object like {"answer": 42}.'
        )

    def generate_prompt(self, content):
        return self.io_prompt(content)

    def refine_prompt(self, content):
        return (
            f"Text: {content['text']}\n\n{content['question']}\n"
            f"A previous answer was {content.get('answer')}. Recheck it carefully. "
            'End your answer with a JSON object like {"answer": 42}.'
        )

    # --- parsing / scoring ---
    def parse(self, text, fallback=None):
        base = fallback if isinstance(fallback, dict) else {}
        for m in reversed(re.findall(r"\{[^{}]*\}", text)):
            try:
                obj = json.loads(m)
                if isinstance(obj, dict) and "answer" in obj:
                    return {**base, "answer": obj["answer"]}
            except json.JSONDecodeError:
                continue
        return fallback

    def validate(self, content):
        return isinstance(content, dict) and content.get("answer") is not None

    def score(self, content):
        # no free proxy for correctness of an arbitrary answer; well-formedness only
        if not isinstance(content, dict) or content.get("answer") is None:
            return float("inf")
        return 0

    def is_complete(self, content, root_content):
        return isinstance(content, dict) and content.get("answer") is not None

    # --- operation graphs ---
    def hybrid_operations(self):
        return [RouteAnswer()]  # router: code tool if possible, LLM fallback otherwise

    def got_operations(self):
        from got.operations import ConditionalRefine, Generate, KeepBest

        return [Generate(k=3), KeepBest(1), ConditionalRefine(attempts=1)]
