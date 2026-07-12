"""Task interface for GoT benchmarking."""
import json
import re


def extract_json_list(text):
    """Tolerant extraction of the last JSON list in model output."""
    matches = re.findall(r"\[[^\[\]]*\]", text)
    for m in reversed(matches):
        try:
            return json.loads(m)
        except json.JSONDecodeError:
            continue
    return None


class Task:
    name = "base"

    # --- instance generation & evaluation ---
    def make_instance(self, rng):
        """Return (input_content, ground_truth)."""
        raise NotImplementedError

    def evaluate(self, content, ground_truth):
        """Error vs ground truth (0 = solved)."""
        raise NotImplementedError

    # --- prompts ---
    def io_prompt(self, content):
        raise NotImplementedError

    def cot_prompt(self, content):
        raise NotImplementedError

    def generate_prompt(self, content):
        return self.io_prompt(content)

    def aggregate_prompt(self, a, b):
        raise NotImplementedError

    def refine_prompt(self, content):
        raise NotImplementedError

    # --- parsing & scoring ---
    def parse(self, text, fallback=None):
        raise NotImplementedError

    def score(self, content):
        """Ground-truth-free error estimate (lower = better)."""
        raise NotImplementedError

    def score_thought(self, thought):
        """Score a thought with access to its parents (default: content only)."""
        return self.score(thought.content)

    def validate(self, content):
        """Cheap well-formedness check on parsed output (ValidateAndImprove)."""
        return content is not None

    def is_complete(self, content, root_content):
        """Is this a FULL-problem answer (vs a chunk/partial thought)?
        Used by the controller's early exit; default is conservative (never)."""
        return False

    # --- structure ops ---
    def split(self, content, n):
        raise NotImplementedError

    def merge_naive(self, a, b):
        """Non-solving fallback merge when all LLM aggregation attempts fail.

        Must only do bookkeeping (e.g. concatenation), never solve the task —
        the score function should penalize the result.
        """
        raise NotImplementedError

    # --- operation graphs ---
    def got_operations(self):
        raise NotImplementedError

    def judge_prompt(self, content, style=0):
        """Prompt for an LLM judge to rate a fuzzy thought 0-10 (JudgeEnsemble)."""
        raise NotImplementedError

    def hybrid_operations(self):
        """Hybrid neuro-symbolic pipeline: LLM for fuzzy steps, code for
        deterministic ones, refinement only on failed checks."""
        raise NotImplementedError

    def tot_operations(self):
        """Tree of Thoughts baseline: beam search over full-problem thoughts."""
        from got.operations import Generate, KeepBest, Refine, Score

        return [
            Generate(k=4),
            Score(),
            KeepBest(2),
            Refine(attempts=2),
            Score(),
            KeepBest(1),
        ]
