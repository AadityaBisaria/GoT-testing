"""Core GoT operations. Each operation transforms the graph frontier.

Faithful to Besta et al. 2023: LLM-based operations sample multiple candidates
(best-of-N), validate/retry on malformed output, and score candidates so
selection operations can prune.
"""
from .graph import Thought


def _attempt(llm, task, prompt, tries, ctx=None):
    """Call the LLM up to `tries` times; return list of parsed valid candidates.

    `ctx` is passed as the parse fallback so dict-state tasks keep their problem
    context (e.g. the original sets/text); a result identical to ctx means the
    output was unparseable and is discarded.
    """
    candidates = []
    for _ in range(tries):
        out = llm.chat(prompt)
        content = task.parse(out, fallback=ctx)
        if content is not None and content is not ctx and task.validate(content):
            candidates.append(content)
    return candidates


class Operation:
    def run(self, graph, task, llm):
        raise NotImplementedError


class Generate(Operation):
    """Branch k new thoughts from each frontier thought via the LLM.

    lazy=True samples one candidate at a time and stops as soon as one scores 0
    (verification is free, so don't pay for branches you won't need)."""

    def __init__(self, k=1, lazy=False):
        self.k = k
        self.lazy = lazy

    def run(self, graph, task, llm):
        new = []
        for t in graph.frontier:
            thoughts = []
            for _ in range(self.k):
                cands = _attempt(llm, task, task.generate_prompt(t.content), 1, ctx=t.content)
                if cands:
                    th = graph.add(Thought(cands[0], parents=[t], operation="generate"))
                    th.score = task.score_thought(th)
                    thoughts.append(th)
                    if self.lazy and th.score == 0:
                        break
            if not thoughts:
                th = graph.add(Thought(t.content, parents=[t], operation="generate"))
                th.score = task.score_thought(th)  # carry input forward, penalized
                thoughts.append(th)
            new.extend(thoughts)
        graph.frontier = new


class Split(Operation):
    """Split each frontier thought into sub-thoughts (programmatic, no LLM)."""

    def __init__(self, n=2):
        self.n = n

    def run(self, graph, task, llm):
        new = []
        for t in graph.frontier:
            for part in task.split(t.content, self.n):
                new.append(graph.add(Thought(part, parents=[t], operation="split")))
        graph.frontier = new


class Aggregate(Operation):
    """Merge frontier thoughts pairwise; best-of-`tries` candidates per merge."""

    def __init__(self, tries=3):
        self.tries = tries

    def run(self, graph, task, llm):
        frontier = list(graph.frontier)
        while len(frontier) > 1:
            merged = []
            for i in range(0, len(frontier) - 1, 2):
                a, b = frontier[i], frontier[i + 1]
                prompt = task.aggregate_prompt(a.content, b.content)
                ctx = task.merge_naive(a.content, b.content)
                best = None
                for content in _attempt(llm, task, prompt, self.tries, ctx=ctx):
                    th = Thought(content, parents=[a, b], operation="aggregate")
                    th.score = task.score_thought(th)
                    if best is None or th.score < best.score:
                        best = th
                if best is None:
                    # all attempts unparseable: naive non-solving merge, penalized by score
                    th = Thought(ctx, parents=[a, b], operation="aggregate-naive")
                    th.score = task.score_thought(th)
                    best = th
                merged.append(graph.add(best))
            if len(frontier) % 2 == 1:
                merged.append(frontier[-1])
            frontier = merged
        graph.frontier = frontier


class Refine(Operation):
    """Ask the LLM to improve each frontier thought; best-of-`attempts`, keep if better."""

    def __init__(self, attempts=1):
        self.attempts = attempts

    def run(self, graph, task, llm):
        new = []
        for t in graph.frontier:
            if t.score is None:
                t.score = task.score_thought(t)
            best = t
            for content in _attempt(llm, task, task.refine_prompt(t.content), self.attempts, ctx=t.content):
                cand = graph.add(Thought(content, parents=[t], operation="refine"))
                cand.score = task.score_thought(cand)
                if cand.score <= best.score:
                    best = cand
            new.append(best)
        graph.frontier = new


class Score(Operation):
    """Score frontier thoughts programmatically (lower = better)."""

    def run(self, graph, task, llm):
        for t in graph.frontier:
            if t.score is None:
                t.score = task.score_thought(t)


class KeepBest(Operation):
    """Prune frontier to the n best-scoring thoughts."""

    def __init__(self, n=1):
        self.n = n

    def run(self, graph, task, llm):
        scored = sorted(graph.frontier, key=lambda t: t.score if t.score is not None else float("inf"))
        graph.frontier = scored[: self.n]


class Execute(Operation):
    """Deterministic code step: transform each frontier thought with a pure
    function, no LLM calls. The hybrid neuro-symbolic pattern: LLM formalizes
    or generates fuzzy thoughts, code executes the checkable/mechanical steps."""

    def __init__(self, fn, name="execute", merge=False):
        self.fn = fn
        self.name = name
        self.merge = merge  # if True, fn takes the list of all frontier contents

    def run(self, graph, task, llm):
        if self.merge:
            contents = [t.content for t in graph.frontier]
            th = graph.add(Thought(self.fn(contents), parents=list(graph.frontier), operation=self.name))
            th.score = task.score_thought(th)
            graph.frontier = [th]
            return
        new = []
        for t in graph.frontier:
            th = graph.add(Thought(self.fn(t.content), parents=[t], operation=self.name))
            th.score = task.score_thought(th)
            new.append(th)
        graph.frontier = new


class ConditionalRefine(Operation):
    """Refine only thoughts whose (free, programmatic) check fails — score > 0.
    Verification asymmetry: checking is free, so correct thoughts exit early."""

    def __init__(self, attempts=2):
        self.attempts = attempts

    def run(self, graph, task, llm):
        new = []
        for t in graph.frontier:
            if t.score is None:
                t.score = task.score_thought(t)
            if t.score == 0:
                new.append(t)  # already verified correct: 0 LLM calls
                continue
            best = t
            for _ in range(self.attempts):
                # one call at a time so a successful fix stops spending calls
                cands = _attempt(llm, task, task.refine_prompt(best.content), 1, ctx=best.content)
                if cands:
                    cand = graph.add(Thought(cands[0], parents=[best], operation="refine"))
                    cand.score = task.score_thought(cand)
                    if cand.score <= best.score:
                        best = cand
                if best.score == 0:
                    break
            new.append(best)
        graph.frontier = new


class JudgeEnsemble(Operation):
    """Score fuzzy thoughts with an ensemble of LLM judges (odd n, style-varied
    prompts, median vote) to blunt single-judge noise. Judges rate quality 0-10;
    converted to an error term added on top of any programmatic score."""

    def __init__(self, n=3, weight=0.1):
        assert n % 2 == 1, "use an odd number of judges"
        self.n = n
        self.weight = weight

    def run(self, graph, task, llm):
        import re
        import statistics

        for t in graph.frontier:
            votes = []
            for j in range(self.n):
                out = llm.chat(task.judge_prompt(t.content, style=j), temperature=0.3)
                m = re.findall(r"\b(10|[0-9])\b", out)
                if m:
                    votes.append(int(m[-1]))
            if votes:
                judge_err = (10 - statistics.median(votes)) * self.weight
                t.score = (t.score or 0) + judge_err


class KeepBestPerParent(Operation):
    """Keep only the best-scoring thought among siblings sharing the same parent."""

    def run(self, graph, task, llm):
        groups = {}
        for t in graph.frontier:
            key = t.parents[0].id if t.parents else None
            cur = groups.get(key)
            score = t.score if t.score is not None else float("inf")
            if cur is None or score < (cur.score if cur.score is not None else float("inf")):
                groups[key] = t
        graph.frontier = list(groups.values())
