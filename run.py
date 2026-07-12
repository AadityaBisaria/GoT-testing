"""Benchmark GoT vs IO vs CoT across tasks on LM Studio."""
import argparse
import json
import os
import random
import time

from got.controller import Controller
from got.llm import LLM
from tasks.constrained_writing import ConstrainedWritingTask
from tasks.doc_merge import DocMergeTask
from tasks.doc_qa import DocQATask
from tasks.doc_summary import DocSummaryTask
from tasks.extract_compute import ExtractComputeTask
from tasks.keyword_count import KeywordCountTask
from tasks.set_intersection import SetIntersectionTask
from tasks.sorting import SortingTask

TASKS = {
    "sorting": lambda size: SortingTask(length=size),
    "set_intersection": lambda size: SetIntersectionTask(size=size),
    "keyword_count": lambda size: KeywordCountTask(sentences=size),
    "extract_compute": lambda size: ExtractComputeTask(facts=max(4, size // 8)),
    "constrained_writing": lambda size: ConstrainedWritingTask(max_words=size * 2),
    "doc_qa": lambda size: DocQATask(n_paras=max(2, size // 12)),
    "doc_merge": lambda size: DocMergeTask(pool=max(4, size // 6)),
    "doc_summary": lambda size: DocSummaryTask(n_paras=max(2, size // 8), max_words=size * 2),
}
METHODS = ["io", "cot", "tot", "got", "hybrid"]


def run_instance(task, method, llm, content, verbose=False):
    if method in ("io", "cot"):
        prompt = task.io_prompt(content) if method == "io" else task.cot_prompt(content)
        out = llm.chat(prompt)
        parsed = task.parse(out, fallback=None)
        if parsed is None:
            parsed = content if isinstance(content, dict) else []
        # for dict-state tasks, keep problem context alongside the answer
        if isinstance(content, dict) and isinstance(parsed, dict):
            parsed = {**content, **parsed}
        return parsed, None
    ops = {"tot": task.tot_operations, "hybrid": task.hybrid_operations,
           "got": task.got_operations}[method]()
    controller = Controller(task, llm, ops)
    best, graph = controller.run(content, verbose=verbose)
    return best.content, graph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=TASKS, default=None)
    ap.add_argument("--method", choices=METHODS, default=None)
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--size", type=int, default=32,
                     help="problem size (list length / set size / sentence count)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--all", action="store_true", help="run all tasks x methods")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--provider", choices=["lmstudio", "openai"], default=None,
                     help="LLM backend to use (default: $LLM_PROVIDER, or lmstudio)")
    ap.add_argument("--model", default=None, help="override the model id/name for the chosen provider")
    args = ap.parse_args()

    tasks = list(TASKS) if args.all or not args.task else [args.task]
    methods = METHODS if args.all or not args.method else [args.method]

    os.makedirs("results", exist_ok=True)
    summary = []
    for task_name in tasks:
        task = TASKS[task_name](args.size)
        for method in methods:
            rng = random.Random(args.seed)
            errors, solved, records = [], 0, []
            llm = LLM(provider=args.provider, model=args.model)
            t0 = time.time()
            for i in range(args.samples):
                content, gt = task.make_instance(rng)
                calls_before = llm.calls
                try:
                    answer, graph = run_instance(task, method, llm, content, verbose=args.verbose)
                    err = task.evaluate(answer, gt)
                except Exception as e:  # noqa: BLE001
                    print(f"  ! {task_name}/{method} sample {i}: {e}")
                    answer, graph, err = None, None, float("inf")
                errors.append(err)
                solved += err == 0
                routing = answer.get("routing") if isinstance(answer, dict) else None
                records.append({
                    "input": content, "ground_truth": gt, "answer": answer,
                    "routing": routing,
                    "error": err if err != float("inf") else None,
                    "llm_calls": llm.calls - calls_before,
                    "graph": graph.to_dict() if graph else None,
                })
                print(f"  {task_name}/{method} sample {i}: error={err} calls={llm.calls - calls_before}")
            finite = [e for e in errors if e != float("inf")]
            row = {
                "task": task_name, "method": method, "samples": args.samples, "size": args.size,
                "mean_error": sum(finite) / len(finite) if finite else None,
                "solve_rate": solved / args.samples,
                "total_llm_calls": llm.calls,
                "calls_per_solved": round(llm.calls / solved, 1) if solved else None,
                "routing": {
                    "deterministic": sum(1 for r in records if r.get("routing", {}) and r["routing"]["route"] == "deterministic"),
                    "fuzzy": sum(1 for r in records if r.get("routing", {}) and r["routing"]["route"] == "fuzzy"),
                    "fallbacks": sum(1 for r in records if r.get("routing", {}) and r["routing"].get("fallback_reason")),
                } if any(r.get("routing") for r in records) else None,
                "tokens": llm.stats(),
                "seconds": round(time.time() - t0, 1),
            }
            summary.append(row)
            with open(f"results/{task_name}_{method}_{args.size}.json", "w") as f:
                json.dump({"summary": row, "records": records}, f, indent=2, default=str)

    print("\n=== Summary ===")
    print(f"{'task':<20}{'method':<8}{'mean_err':<10}{'solved':<8}{'calls':<7}{'tokens':<9}{'calls/solved':<13}{'sec':<7}")
    for r in summary:
        me = f"{r['mean_error']:.2f}" if r["mean_error"] is not None else "-"
        cps = r["calls_per_solved"] if r["calls_per_solved"] is not None else "-"
        toks = r["tokens"]["prompt_tokens"] + r["tokens"]["completion_tokens"]
        print(f"{r['task']:<20}{r['method']:<8}{me:<10}{r['solve_rate']:<8.0%}"
              f"{r['total_llm_calls']:<7}{toks:<9}{str(cps):<13}{r['seconds']:<7}")


if __name__ == "__main__":
    main()
