import asyncio
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv() -> bool:
        return False

from agent.main_agent import MainAgent
from engine.llm_judge import MultiJudgeConsensus
from engine.retrieval_eval import RetrievalEvaluator, build_document_corpus, expected_ids_for_case, load_markdown_corpus
from engine.runner import BenchmarkRunner, summarize_results


DATASET_PATH = Path("data/golden_set.jsonl")
REPORTS_DIR = Path("reports")
ANALYSIS_PATH = Path("analysis/failure_analysis.md")
REFLECTION_PATH = Path("analysis/reflections/reflection_Duc_Nguyen.md")


GATE_THRESHOLDS = {
    "min_agreement_rate": 0.60,
    "min_hit_rate_at_k": 0.75,
    "min_mrr_at_k": 0.60,
    "min_avg_judge_score": 3.5,
    "max_cost_increase_pct": 30.0,
    "max_p95_latency_ms": 120000.0,
}


def load_dataset(path: Path = DATASET_PATH) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError("Missing data/golden_set.jsonl. Run python data/synthetic_gen.py first.")
    with path.open("r", encoding="utf-8") as handle:
        cases = [json.loads(line) for line in handle if line.strip()]
    if len(cases) < 50:
        raise ValueError("Golden dataset must contain at least 50 cases.")
    for case in cases:
        case_type = (case.get("metadata") or {}).get("type", case.get("case_type"))
        expected_ids = case.get("expected_retrieval_ids") or case.get("ground_truth_ids") or []
        if case_type != "out-of-context" and not expected_ids:
            raise ValueError(f"Case {case.get('id') or case.get('case_id')} is missing expected_retrieval_ids.")
    return cases


def load_retrieval_corpus(dataset: List[Dict]) -> Dict[str, str]:
    corpus = load_markdown_corpus()
    if not corpus:
        return build_document_corpus(dataset)

    corpus_ids = set(corpus)
    missing_ids = sorted(
        {
            doc_id
            for case in dataset
            for doc_id in expected_ids_for_case(case)
            if doc_id not in corpus_ids
        }
    )
    if missing_ids:
        raise ValueError(
            "Golden dataset expected_retrieval_ids do not match data/corpus markdown stems: "
            + ", ".join(missing_ids)
        )
    return corpus


def load_environment() -> None:
    if load_dotenv():
        return
    env_path = Path(".env")
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


async def run_benchmark_with_results(agent_version: str, dataset: List[Dict], top_k: int, concurrency: int, force_offline_judges: bool = False) -> Tuple[List[Dict], Dict]:
    corpus = load_retrieval_corpus(dataset)
    evaluator = RetrievalEvaluator(corpus, top_k=top_k)
    if force_offline_judges:
        old_key = os.environ.pop("GEMINI_API_KEY", None)
        try:
            judge = MultiJudgeConsensus(
                timeout_seconds=float(os.getenv("JUDGE_TIMEOUT_SECONDS", "45")),
                max_retries=int(os.getenv("JUDGE_MAX_RETRIES", "2")),
            )
        finally:
            if old_key is not None:
                os.environ["GEMINI_API_KEY"] = old_key
    else:
        judge = MultiJudgeConsensus(
            timeout_seconds=float(os.getenv("JUDGE_TIMEOUT_SECONDS", "45")),
            max_retries=int(os.getenv("JUDGE_MAX_RETRIES", "2")),
        )
    runner = BenchmarkRunner(MainAgent(agent_version), evaluator, judge, concurrency=concurrency, top_k=top_k)
    start = time.perf_counter()
    results = await runner.run_all(dataset)
    runtime = time.perf_counter() - start
    return results, summarize_results(results, agent_version, runtime, concurrency, judge.quota_metadata())


def pct_delta(new: float, old: float) -> float:
    if old == 0:
        return 0.0 if new == 0 else 100.0
    return ((new - old) / old) * 100


def build_regression(v1_summary: Dict, v2_summary: Dict) -> Dict:
    v1 = v1_summary["metrics"]
    v2 = v2_summary["metrics"]
    v1_cost = v1.get("total_cost_usd", v1.get("estimated_total_cost_usd", 0.0))
    v2_cost = v2.get("total_cost_usd", v2.get("estimated_total_cost_usd", 0.0))
    deltas = {
        "hit_rate_delta": v2["hit_rate"] - v1["hit_rate"],
        "mrr_delta": v2["mrr"] - v1["mrr"],
        "score_delta": v2["avg_score"] - v1["avg_score"],
        "cost_delta_pct": pct_delta(v2_cost, v1_cost),
        "latency_delta_pct": pct_delta(v2["p95_latency_ms"], v1["p95_latency_ms"]),
    }
    rollback_reasons = []
    review_reasons = []
    block_reasons = []
    judge_mode = v2_summary.get("judge", {}).get("judge_mode") or v2_summary.get("metadata", {}).get("judge_mode")
    if v2_summary.get("metadata", {}).get("total", 0) < 50:
        block_reasons.append("benchmark has fewer than 50 cases")
    if not judge_mode:
        block_reasons.append("judge mode is missing")
    if v2["hit_rate"] < GATE_THRESHOLDS["min_hit_rate_at_k"]:
        rollback_reasons.append("hit_rate below threshold")
    if v2["mrr"] < GATE_THRESHOLDS["min_mrr_at_k"]:
        rollback_reasons.append("mrr below threshold")
    if v2["avg_score"] < GATE_THRESHOLDS["min_avg_judge_score"]:
        rollback_reasons.append("average judge score below threshold")
    if v2.get("agreement_rate", 0.0) < GATE_THRESHOLDS["min_agreement_rate"]:
        review_reasons.append("judge agreement below threshold")
    if judge_mode == "provider_gemini_single_model":
        review_reasons.append("single Gemini model mode is lower confidence")
    if judge_mode in {"mixed_provider_fallback", "provider_gemini_budget_blocked", "provider_gemini_budget_exhausted"}:
        review_reasons.append(f"{judge_mode} requires human review")
    if judge_mode == "offline_fallback":
        block_reasons.append("offline fallback mode is not valid for final provider scoring")
    if os.getenv("EVAL_LIMIT"):
        review_reasons.append("EVAL_LIMIT is set")
    if judge_mode == "provider_gemini_dual_model":
        judge = v2_summary.get("judge", {})
        if not judge.get("judge_b_actual_requests", 0):
            review_reasons.append("Judge B did not run")
    judge = v2_summary.get("judge", {})
    judge_a_model = judge.get("judge_a_model")
    judge_b_model = judge.get("judge_b_model")
    if judge_a_model and judge_b_model and judge_a_model == judge_b_model:
        review_reasons.append("Gemini judge A and judge B use the same model")
    if deltas["cost_delta_pct"] > GATE_THRESHOLDS["max_cost_increase_pct"]:
        rollback_reasons.append("cost increase above threshold")
    if v2["p95_latency_ms"] > GATE_THRESHOLDS["max_p95_latency_ms"]:
        rollback_reasons.append("p95 latency above threshold")
    if v2["hit_rate"] < v1["hit_rate"] or v2["mrr"] < v1["mrr"] or v2["avg_score"] < v1["avg_score"]:
        rollback_reasons.append("candidate is worse than baseline on at least one quality metric")
    decision = "BLOCK_RELEASE" if block_reasons else "NEEDS_REVIEW" if (rollback_reasons or review_reasons) else "APPROVE"
    reasons = block_reasons + rollback_reasons + review_reasons
    return {
        "baseline_version": "v1",
        "candidate_version": "v2",
        "thresholds": GATE_THRESHOLDS,
        **deltas,
        "release_decision": decision,
        "reasons": reasons or ["All quality, cost, latency, and judge reliability thresholds passed."],
    }


def write_failure_analysis(results: List[Dict], summary: Dict, path: Path = ANALYSIS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = summary["metrics"]
    cluster_counts = Counter(cluster for result in results for cluster in result["failure_clusters"] if cluster != "none")
    worst = sorted(
        results,
        key=lambda item: (
            item.get("status") == "pass",
            item["final_score"],
            item["retrieval"]["mrr"] if item["retrieval"].get("mrr") is not None else 1.0,
            -item["judge"].get("score_gap", 0),
            item["case_id"],
        ),
    )[:3]

    cluster_rows = "\n".join(
        f"| {cluster} | {count} | {cluster_reason(cluster)} |" for cluster, count in sorted(cluster_counts.items())
    ) or "| none | 0 | No low-score clusters detected. |"

    whys = []
    for index, result in enumerate(worst, start=1):
        primary = next((cluster for cluster in result["failure_clusters"] if cluster != "none"), "incomplete")
        answer_text = result["agent_response"]["answer"] if isinstance(result.get("agent_response"), dict) else str(result.get("agent_response", ""))
        whys.append(
            f"### Case #{index}: {result['case_id']} ({primary})\n"
            f"- Question: {result['question']}\n"
            f"- Expected answer summary: {result['expected_answer'][:180]}\n"
            f"- Agent answer summary: {answer_text[:180]}\n"
            f"- Retrieved ids: {result['retrieved_ids']}\n"
            f"- Hit Rate/MRR: {result['retrieval']['hit_rate']:.3f} / {format_nullable(result['retrieval'].get('mrr'))}\n"
            f"- Final score/agreement: {result['final_score']:.2f} / {result['judge'].get('agreement_rate', 0.0):.3f}\n"
            f"1. Symptom: Final score {result['final_score']:.2f}; verdict {result['final_verdict']}; primary cluster {primary}.\n"
            f"2. Why 1: The answer failed the strongest signal: {primary}.\n"
            f"3. Why 2: Retrieval returned {result['retrieved_ids']} for expected ids {result['expected_retrieval_ids']}.\n"
            f"4. Why 3: TF-IDF ranking depends on token overlap, so ambiguous, conflicting, or adversarial wording can move the right chunk down.\n"
            f"5. Why 4: The agent prompt uses retrieved context directly and has no reranker, semantic expansion, or citation repair pass.\n"
            f"6. Why 5: The pipeline is optimized for deterministic lab reproducibility rather than production retrieval robustness.\n"
            f"7. Root cause: {root_cause(primary)}.\n"
        )

    if not whys:
        whys.append("### No failing cases\nAll benchmark cases met the current pass threshold; continue monitoring edge cases.")

    content = f"""# Failure Analysis Report

## 1. Benchmark Overview
- **Total cases:** {summary['metadata']['total']}
- **Pass/Fail:** {metrics['pass_count']}/{metrics['fail_count']}
- **Pass rate:** {(metrics['pass_count'] / max(summary['metadata']['total'], 1)):.3f}
- **Average faithfulness:** {avg_judge_field(results, 'faithfulness'):.3f}
- **Average relevancy:** {avg_judge_field(results, 'relevancy'):.3f}
- **Average LLM judge score:** {metrics['avg_score']:.3f} / 5.0
- **Hit Rate@k:** {metrics['hit_rate']:.3f}
- **MRR@k:** {metrics['mrr']:.3f}
- **Agreement Rate:** {metrics['agreement_rate']:.3f}
- **Cohen's Kappa:** {metrics['cohen_kappa']:.3f}
- **Conflict Count/Rate:** {metrics.get('conflict_count', 0)} / {metrics.get('conflict_rate', 0.0):.3f}
- **Judge mode:** {summary['metadata'].get('judge_mode', 'unknown')}
- **Judge models:** A={summary.get('judge', {}).get('judge_a_model', 'n/a')}; B={summary.get('judge', {}).get('judge_b_model', 'n/a')}
- **Retriever:** {summary['metadata'].get('retriever_type', 'unknown')}
- **Release decision:** {summary.get('release_decision', 'unknown')}
- **Latency:** avg {metrics['avg_latency_ms']:.1f} ms; p95 {metrics['p95_latency_ms']:.1f} ms
- **Cost:** total ${metrics['total_cost_usd']:.6f}; per case ${metrics['cost_per_case_usd']:.6f}

## 2. Failure Clustering
| Failure cluster | Count | Expected system cause |
|---|---:|---|
{cluster_rows}

## Quota Strategy
- 2.5 Flash Judge B is enabled only when `FINAL_PROVIDER_RUN=true`, `ALLOW_JUDGE_B_FINAL=true`, and `EVAL_LIMIT` is unset.
- Judge batch size: {summary.get('judge', {}).get('batch_size', 'n/a')}
- Gemini limiter: {summary.get('judge', {}).get('requests_per_minute', 'n/a')} RPM shared by Judge A and Judge B.
- Judge B hard cap: {summary.get('judge', {}).get('judge_b_daily_request_budget', 'n/a')} requests/day with {summary.get('judge', {}).get('judge_b_reserved_retry_budget', 'n/a')} reserved for retries.
- Cache enabled: {summary.get('judge', {}).get('cache_enabled', 'n/a')} at `.cache/judge_results.json`.

## 3. 5 Whys For The 3 Worst Cases
{chr(10).join(whys)}

## 4. Action Plan
- Add semantic or hybrid retrieval and reranking for adversarial and multi-hop questions.
- Expand chunks around cited source ids so citation-sensitive answers have all required evidence.
- Route judge conflicts to a stricter citation-first adjudicator before release decisions.
- Cache judge outputs and escalate only low-confidence cases to expensive model judges.
- Add position-bias checks by shuffling retrieved contexts and comparing score stability.
"""
    path.write_text(content, encoding="utf-8")


def cluster_reason(cluster: str) -> str:
    return {
        "retrieval_miss": "Retrieval did not surface any ground truth id.",
        "wrong_chunk": "Retriever found partial evidence but missed at least one required source.",
        "citation_missing": "Generation omitted a required source marker.",
        "hallucination": "Answer included unsupported content.",
        "incomplete": "Answer was too thin relative to expected answer.",
        "tone_mismatch": "Safety/tone handling was not appropriate for adversarial input.",
        "judge_disagreement": "Judges differed by more than one score point.",
        "position_bias": "Correct source was retrieved but not ranked first.",
        "cost_high": "Cost exceeded expected budget.",
        "latency_high": "Latency exceeded expected threshold.",
        "provider_error": "A provider call failed after retries.",
        "parse_error": "A judge response could not be parsed as JSON.",
        "unknown": "The case failed without a known cluster.",
    }.get(cluster, "Unclassified low-score behavior.")


def root_cause(cluster: str) -> str:
    return {
        "retrieval_miss": "retrieval",
        "wrong_chunk": "chunking and retrieval",
        "citation_missing": "prompting and citation formatting",
        "hallucination": "prompting and safety guardrails",
        "incomplete": "prompting",
        "tone_mismatch": "prompting",
        "judge_disagreement": "judge disagreement",
        "position_bias": "position bias",
        "cost_high": "cost bottleneck",
        "latency_high": "latency bottleneck",
        "provider_error": "provider failure",
        "parse_error": "judge parsing",
        "unknown": "system evaluation coverage",
    }.get(cluster, "system evaluation coverage")


def format_nullable(value) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def avg_judge_field(results: List[Dict], field: str) -> float:
    values = [judge[field] for result in results for judge in result["judge"]["judge_results"] if field in judge]
    return sum(values) / len(values) if values else 0.0


def write_reflection(path: Path = REFLECTION_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """# Individual Reflection - Duc Nguyen

## Engineering Contribution
I owned the evaluation engine side of the two-person project: retrieval metrics integration, the Gemini multi-judge provider fix, async runner/report validation, regression release gate, failure analysis, check_lab validation, and documentation. Linh owns the dataset, SDG, and hard-case design; I only touched that path to keep the end-to-end benchmark runnable with the agreed schema.

## Technical Explanation
MRR measures retrieval ranking quality by taking the reciprocal rank of the first relevant document, then averaging across normal cases. A system can have high Hit Rate but lower MRR if the right source appears late, which matters because generation often overuses early context. Cohen's Kappa measures judge agreement after adjusting for chance agreement; it is stronger than raw agreement rate when judges have skewed score distributions. Position bias appears when a judge or generator prefers earlier options or retrieved chunks independent of content quality.

## Cost vs Quality Trade-off
The best production design is not to call expensive judges for every case. In this lab, Gemini dual-model mode is the preferred rubric-safe mode. Single-model mode is lower-confidence because one Gemini result is paired with a heuristic fallback, and offline fallback is only for local reproducibility. Caching judge results and limiting output tokens reduce cost without changing benchmark semantics.

## Problems And Strategy
The main issue was that the previous provider setup could claim provider scoring while using fallback-like behavior and DeepSeek configuration. I replaced it with the official `google-genai` client pattern, explicit Gemini A/B judges, parse-safe JSON scoring, and clear provider modes. Another challenge was connecting retrieval quality to answer quality without a vector database, so the TF-IDF retriever reports real Hit Rate and MRR from retrieved ids instead of simulated metrics.
""",
        encoding="utf-8",
    )
    linh_path = Path("analysis/reflections/reflection_Linh.md")
    if not linh_path.exists():
        linh_path.write_text(
            """# Individual Reflection - Linh

## Dataset / SDG Ownership
I own the golden dataset, synthetic data generation, and hard-case guide for the two-person Evaluation Factory project. The dataset uses stable case ids, expected answers, context, and `expected_retrieval_ids` so the engine can compute Hit Rate and MRR from real retrieval outputs.

## Hard Case Design
The case mix includes easy and medium fact checks, hard adversarial prompts, out-of-context questions, ambiguous cases, conflicting evidence, and multi-turn or multi-hop style questions. The red-team cases are meant to test whether the agent stays grounded in retrieved evidence instead of following unsafe or unsupported instructions.

## Handoff Note
This reflection is intentionally brief because the final dataset wording and any additional Linh-owned edits should be reviewed by Linh before submission.
""",
            encoding="utf-8",
        )


def write_docs() -> None:
    docs_path = Path("docs/eval_factory.md")
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    docs_path.write_text(
        """# Evaluation Factory

Run the lab in order:

```bash
python data/synthetic_gen.py
python main.py
python check_lab.py
```

`data/synthetic_gen.py` creates 60 deterministic Linh-owned golden cases with `expected_retrieval_ids`. `main.py` runs V1 and V2 benchmarks asynchronously, writes one V2 row per case to `reports/benchmark_results.json`, writes `reports/summary.json`, refreshes failure analysis, and creates/updates reflections.

## Gemini judge configuration

Provider mode uses the official `google-genai` SDK format:

```python
from google import genai

client = genai.Client()
response = client.models.generate_content(
    model=model_name,
    contents=prompt,
)
text = response.text
```

Environment:

```bash
GEMINI_API_KEY=...
GEMINI_JUDGE_A_MODEL=gemini-3.1-flash-lite
GEMINI_JUDGE_B_MODEL=gemini-2.5-flash
JUDGE_PROVIDER=gemini
JUDGE_TIMEOUT_SECONDS=45
JUDGE_MAX_RETRIES=2
JUDGE_CONCURRENCY=4
```

`provider_gemini_dual_model` is the preferred rubric-safe mode and requires both Gemini model calls to work. If only one Gemini model works, the runner records `provider_gemini_single_model` and pairs the successful Gemini judge with one heuristic judge; this is lower-confidence and should trigger review. If no key exists or both model calls fail, `offline_fallback` uses deterministic judges only for local reproducibility and must not be treated as final provider scoring.

Both Gemini judges receive the same rubric and must return strict JSON with score, faithfulness, relevance, completeness, citation correctness, hallucination risk, and a short reason. The parser strips markdown fences, extracts JSON from extra text, validates numeric ranges, and records `parse_error` with a safe failing score if parsing fails.

## Metrics and gate

Metrics include Hit Rate, MRR, Recall, average judge score, agreement rate, Cohen's Kappa, conflict count/rate, latency, approximate tokens, and estimated cost. Retrieval compares `retrieved_ids` against `expected_retrieval_ids`; out-of-context cases can have no expected ids and are excluded from MRR averaging. The local retriever reports `retriever_type=tfidf_vector_retriever`.

Multi-judge consensus averages scores within one point. Larger conflicts use citation correctness, faithfulness, and completeness as tie-breakers. Missing required citations cap final score at 3. High hallucination risk fails the case unless retrieval clearly supports the answer, and still caps the score.

The release gate compares V2 against V1 and thresholds for Hit Rate, MRR, average judge score, judge agreement, cost increase, and p95 latency. It outputs `APPROVE` only when all thresholds pass in reliable final dual-provider mode, `NEEDS_REVIEW` when quality, cost, latency, budget, single-model, fallback, or weak-agreement signals require human review, and `BLOCK_RELEASE` for offline-only final scoring or missing required artifacts.

## Ownership

- Linh owns dataset generation, SDG, hard cases, `data/synthetic_gen.py`, `data/golden_set.jsonl`, and `data/HARD_CASES_GUIDE.md`.
- Duc owns the evaluation engine, retrieval metrics integration, Gemini multi-judge provider, async runner, regression gate, reports, failure analysis, check_lab validation, documentation, and final validation.
""",
        encoding="utf-8",
    )


async def main() -> None:
    load_environment()
    dataset = load_dataset()
    concurrency = int(os.getenv("JUDGE_CONCURRENCY", os.getenv("EVAL_CONCURRENCY", "4")))
    final_provider_run = os.getenv("FINAL_PROVIDER_RUN", "").strip().lower() in {"1", "true", "yes", "on"}
    v1_results, v1_summary = await run_benchmark_with_results("Agent_V1_Base", dataset, top_k=3, concurrency=concurrency, force_offline_judges=final_provider_run)
    v2_results, v2_summary = await run_benchmark_with_results("Agent_V2_Optimized", dataset, top_k=3, concurrency=concurrency)
    regression = build_regression(v1_summary, v2_summary)
    v2_summary["regression"] = regression
    v2_summary["v1_metrics"] = v1_summary["metrics"]
    v2_summary["release_decision"] = regression["release_decision"]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "benchmark_results.json").write_text(json.dumps(v2_results, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORTS_DIR / "baseline_results_v1.json").write_text(json.dumps(v1_results, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORTS_DIR / "summary.json").write_text(json.dumps(v2_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_failure_analysis(v2_results, v2_summary)
    write_reflection()
    write_docs()

    print(f"Benchmark complete: {len(dataset)} cases, decision={regression['release_decision']}")
    print(f"Hit Rate={v2_summary['metrics']['hit_rate']:.3f}, MRR={v2_summary['metrics']['mrr']:.3f}, Avg Score={v2_summary['metrics']['avg_score']:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
