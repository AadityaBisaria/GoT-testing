"""Controller: executes a graph-of-operations over a thought graph."""
from .graph import Thought, ThoughtGraph


class Controller:
    def __init__(self, task, llm, operations, early_exit=True):
        self.task = task
        self.llm = llm
        self.operations = operations
        self.early_exit = early_exit

    def run(self, input_content, verbose=False):
        graph = ThoughtGraph()
        root = graph.add(Thought(input_content, operation="root"))
        graph.frontier = [root]
        for op in self.operations:
            op.run(graph, self.task, self.llm)
            if verbose:
                print(f"  [{op.__class__.__name__}] frontier={len(graph.frontier)} "
                      f"scores={[t.score for t in graph.frontier]}")
            if self.early_exit:
                # a verified-correct FULL answer makes remaining ops pure waste
                done = [t for t in graph.frontier
                        if t.score == 0 and self.task.is_complete(t.content, root.content)]
                if done:
                    if verbose:
                        print("  [early-exit] verified complete answer, skipping remaining ops")
                    graph.frontier = done
                    break
        best = min(
            graph.frontier,
            key=lambda t: t.score if t.score is not None else self.task.score_thought(t),
        )
        return best, graph
