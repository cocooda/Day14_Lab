import json
import os
import sys
from pathlib import Path


PLACEHOLDERS = ["X/Y", "0.XX", "X.X", "[Mô tả ngắn]", "[Mo ta ngan]", "..."]


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise AssertionError(f"{path} is not valid JSON: {exc}") from exc


def validate_dataset() -> None:
    path = Path("data/golden_set.jsonl")
    if not path.exists():
        raise AssertionError("data/golden_set.jsonl is missing. Run python data/synthetic_gen.py first.")
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(cases) < 50:
        raise AssertionError(f"golden_set.jsonl has {len(cases)} cases; expected at least 50.")
    required = {"id", "question", "expected_answer", "context", "expected_retrieval_ids", "metadata"}
    for case in cases:
        missing = required - set(case)
        if missing:
            raise AssertionError(f"{case.get('id', '<unknown>')} missing fields: {sorted(missing)}")
        metadata = case.get("metadata") or {}
        if metadata.get("type") != "out-of-context" and not case.get("expected_retrieval_ids"):
            raise AssertionError(f"{case['id']} is a normal case without expected_retrieval_ids.")
    print(f"OK dataset: {len(cases)} valid cases")


def validate_reports() -> None:
    summary_path = Path("reports/summary.json")
    results_path = Path("reports/benchmark_results.json")
    if not summary_path.exists():
        raise AssertionError("reports/summary.json is missing.")
    if not results_path.exists():
        raise AssertionError("reports/benchmark_results.json is missing.")
    summary = load_json(summary_path)
    results = load_json(results_path)
    if not isinstance(results, list):
        raise AssertionError("benchmark_results.json must be a list with one result per case.")
    if len(results) < 50:
        raise AssertionError(f"benchmark_results.json has {len(results)} results; expected at least 50.")

    for key in ["hit_rate", "mrr", "avg_score", "agreement_rate"]:
        if key not in summary.get("metrics", {}):
            raise AssertionError(f"summary.metrics.{key} is missing.")
    if "release_decision" not in summary.get("regression", {}):
        raise AssertionError("summary.regression.release_decision is missing.")
    if "judge_mode" not in summary.get("judge", {}):
        raise AssertionError("summary.judge.judge_mode is missing.")
    if summary["judge"]["judge_mode"].startswith("provider_deepseek"):
        raise AssertionError("DeepSeek provider mode is not allowed for this project.")

    required_result_fields = {"test_case_id", "question", "agent_response", "latency_ms", "retrieval", "judge", "status"}
    for result in results:
        missing = required_result_fields - set(result)
        if missing:
            raise AssertionError(f"{result.get('test_case_id', '<unknown>')} missing result fields: {sorted(missing)}")
    print(f"OK reports: {len(results)} benchmark results; decision={summary['regression']['release_decision']}")


def validate_failure_analysis() -> None:
    path = Path("analysis/failure_analysis.md")
    if not path.exists():
        raise AssertionError("analysis/failure_analysis.md is missing.")
    text = path.read_text(encoding="utf-8")
    for placeholder in PLACEHOLDERS:
        if placeholder in text:
            raise AssertionError(f"failure_analysis.md still contains placeholder: {placeholder}")
    for required in ["Benchmark Overview", "Failure Clustering", "5 Whys", "Root cause", "Action Plan"]:
        if required not in text:
            raise AssertionError(f"failure_analysis.md missing section/content: {required}")
    print("OK failure analysis: no obvious placeholders")


def validate_reflections() -> None:
    duc = Path("analysis/reflections/reflection_Duc_Nguyen.md")
    linh = Path("analysis/reflections/reflection_Linh.md")
    if not duc.exists():
        raise AssertionError("Duc reflection is missing.")
    if not linh.exists():
        print("WARN Linh reflection is missing; dataset ownership should be finalized before submission.")
    else:
        print("OK reflections: Duc and Linh files present")


def validate_lab() -> int:
    try:
        validate_dataset()
        validate_reports()
        validate_failure_analysis()
        validate_reflections()
    except AssertionError as exc:
        print(f"FAIL {exc}")
        return 1
    print("OK lab submission artifacts are valid.")
    return 0


if __name__ == "__main__":
    sys.exit(validate_lab())
