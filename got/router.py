"""LLM router with a whitelisted tool registry and fail-loud fallback.

The LLM tags a thought as deterministic(tool, args) or fuzzy. Misclassification
is asymmetric by design:
- det-misroute (wrong/unknown tool, bad args, execution error, failed round-trip
  check) fails LOUDLY -> caught -> falls back to the fuzzy LLM path. Costs one
  wasted call, never a silent wrong answer.
- fuzzy-misroute (code could have done it) only costs tokens, never correctness.
The LLM never writes code; it can only pick from this registry.
"""
import json
import re

from .graph import Thought


def _is_num_list(x):
    return isinstance(x, list) and len(x) > 0 and all(isinstance(v, (int, float)) for v in x)


def _safe_arith(expr):
    if not re.fullmatch(r"[\d\s+\-*/().]+", expr):
        raise ValueError("arithmetic expr may contain only numbers and + - * / ( )")
    return eval(expr, {"__builtins__": {}}, {})  # noqa: S307 - charset-restricted above


REGISTRY = {
    "sort_numbers": {
        "fn": lambda args: sorted(args["numbers"]),
        "validate": lambda args: _is_num_list(args.get("numbers")),
        "desc": 'sort a list of numbers ascending; args: {"numbers": [..]}',
    },
    "sum_numbers": {
        "fn": lambda args: sum(args["numbers"]),
        "validate": lambda args: _is_num_list(args.get("numbers")),
        "desc": 'sum a list of numbers; args: {"numbers": [..]}',
    },
    "arithmetic": {
        "fn": lambda args: _safe_arith(args["expression"]),
        "validate": lambda args: isinstance(args.get("expression"), str),
        "desc": 'evaluate an arithmetic expression of plain numbers, e.g. "(12+7)*3"; args: {"expression": "..."}',
    },
    "set_intersection": {
        "fn": lambda args: sorted(set(args["a"]) & set(args["b"])),
        "validate": lambda args: _is_num_list(args.get("a")) and _is_num_list(args.get("b")),
        "desc": 'common elements of two number lists; args: {"a": [..], "b": [..]}',
    },
    "count_occurrences": {
        "fn": lambda args: len(re.findall(re.escape(args["word"]), args["text"], re.IGNORECASE)),
        "validate": lambda args: isinstance(args.get("word"), str) and isinstance(args.get("text"), str),
        "desc": 'count case-insensitive occurrences of a word in a text; args: {"word": "...", "text": "..."}',
    },
    "word_count": {
        "fn": lambda args: len(args["text"].split()),
        "validate": lambda args: isinstance(args.get("text"), str),
        "desc": 'number of words in a text; args: {"text": "..."}',
    },
    "sentence_count": {
        "fn": lambda args: len([s for s in re.split(r"[.!?]+", args["text"]) if s.strip()]),
        "validate": lambda args: isinstance(args.get("text"), str),
        "desc": 'number of sentences in a text; args: {"text": "..."}',
    },
}


def registry_help():
    return "\n".join(f"- {name}: {spec['desc']}" for name, spec in REGISTRY.items())


def route_and_execute(llm, task, content, sub_question, context_text=None):
    """Ask the LLM to route one sub-question; execute deterministically if safely
    possible, otherwise fall back to the fuzzy LLM path.

    Returns (answer, routing) where routing = {"route", "tool", "fallback_reason"}.
    """
    routing = {"route": "fuzzy", "tool": None, "fallback_reason": None}
    prompt = (
        "Decide how to answer the sub-question below. Available deterministic tools:\n"
        f"{registry_help()}\n\n"
        f"Sub-question: {sub_question}\n"
        + ("A reference text is available; for text tools just pass \"$TEXT\" as the \"text\" arg — do NOT copy the text.\n" if context_text else "")
        + 'If one tool answers it exactly, reply {"route": "deterministic", "tool": "<name>", "args": {...}}.\n'
        'Otherwise reply {"route": "fuzzy"}.\nReply with only the JSON object.'
    )
    out = llm.chat(prompt, temperature=0.2)
    try:
        matches = re.findall(r"\{.*\}", out, re.DOTALL)
        decision = json.loads(matches[-1]) if matches else {}
    except json.JSONDecodeError:
        decision = {}

    if decision.get("route") == "deterministic":
        tool = decision.get("tool")
        args = decision.get("args") or {}
        # $TEXT placeholder lets the router use large texts without copying them
        for k, v in list(args.items()):
            if v == "$TEXT" and context_text is not None:
                args[k] = context_text
        # never trust the LLM to relay the reference text: code injects it
        if context_text is not None and "text" in args:
            args["text"] = context_text
        spec = REGISTRY.get(tool)
        try:
            if spec is None:
                raise KeyError(f"unknown tool {tool!r}")
            if not spec["validate"](args):
                raise ValueError(f"invalid args for {tool}: {args}")
            result = spec["fn"](args)
            routing.update(route="deterministic", tool=tool)
            return result, routing
        except Exception as e:  # noqa: BLE001 - ANY det failure falls back to fuzzy
            routing["fallback_reason"] = f"{type(e).__name__}: {e}"

    # fuzzy path: LLM answers the sub-question directly
    fuzzy_prompt = (
        (f"Text: {context_text}\n\n" if context_text else "") +
        f"{sub_question}\nEnd your answer with a JSON object like {{\"answer\": ...}}."
    )
    out = llm.chat(fuzzy_prompt)
    m = re.findall(r"\{[^{}]*\}", out)
    ans = None
    for mm in reversed(m):
        try:
            obj = json.loads(mm)
            if "answer" in obj:
                ans = obj["answer"]
                break
        except json.JSONDecodeError:
            continue
    return ans, routing
