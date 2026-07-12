"""Thought graph: a DAG of thoughts produced during reasoning."""
import itertools


class Thought:
    _ids = itertools.count()

    def __init__(self, content, parents=None, operation="root"):
        self.id = next(Thought._ids)
        self.content = content  # task-specific state (e.g. a list of numbers)
        self.parents = parents or []
        self.operation = operation  # which operation produced this thought
        self.score = None  # lower = better (error); set by Score

    def to_dict(self):
        return {
            "id": self.id,
            "content": self.content,
            "parents": [p.id for p in self.parents],
            "operation": self.operation,
            "score": self.score,
        }


class ThoughtGraph:
    def __init__(self):
        self.thoughts = []
        self.frontier = []  # thoughts the next operation acts on

    def add(self, thought):
        self.thoughts.append(thought)
        return thought

    def to_dict(self):
        return [t.to_dict() for t in self.thoughts]
