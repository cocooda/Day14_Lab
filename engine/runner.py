import asyncio
import statistics
import time
from dataclasses import asdict, dataclass
from typing import Dict, List

from engine.llm_judge import MultiJudgeConsensus, cohen_kappa
from engine.retrieval_eval import RetrievalEvaluator, expected_ids_for_case


@dataclass
class GoldenCase:
    case_id: str
    question: str
    expected_answer: str
    ground_truth_ids: List[str]
    case_type: str
    difficulty: str
    tags: List[str]


@dataclass
class RetrievalResult:
    retrieved_ids: List[str]
    hit_rate: float
    mrr: float
    recall: float
    avg_retrieved_count: int


@dataclass
class CaseBenchmarkResult:
    test_case_id: str
    case_id: str
    question: str
    expected_answer: str
    expected_retrieval_ids: List[str]
    case_type: str
    difficulty: str
    retrieved_ids: List[str]
    agent_response: str
    latency_ms: float
    retrieval: Dict
    judge: Dict
    token_usage: Dict
    cost_usd: float
    final_score: float
    final_verdict: str
    failure_clusters: List[str]
    status: str
    error_type: str | None = None
    error_message: str | None = None


def approx_tokens(text: str) -> int:
    return max(1, round(len(text.split()) * 1.25))


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    # Blended local estimate roughly matching cheap judge/generator pricing.
    return (input_tokens / 1_000_000 * 0.15) + (output_tokens / 1_000_000 * 0.60)


def classify_failures(case: Dict, retrieval: Dict, judge: Dict, answer: str, retrieved_ids: List[str]) -> List[str]:
    clusters = []
    if retrieval["hit_rate"] == 0:
        clusters.append("retrieval_miss")
    if retrieval["hit_rate"] == 1 and retrieval.get("recall") is not None and retrieval["recall"] < 1:
        clusters.append("wrong_chunk")
    if judge.get("required_citation_missing"):
        clusters.append("citation_missing")
    if judge.get("hallucination_detected"):
        clusters.append("hallucination")
    if judge["final_score"] < 3.5 and "hallucination" not in clusters:
        clusters.append("incomplete")
    case_type = (case.get("metadata") or {}).get("type", case.get("case_type"))
    if case_type == "adversarial" and "unsupported" in answer.lower():
        clusters.append("tone_mismatch")
    if judge.get("score_gap", 0) > 1:
        clusters.append("judge_disagreement")
    expected = expected_ids_for_case(case)
    if retrieved_ids and expected and expected[0] not in retrieved_ids[:1]:
        clusters.append("position_bias")
    return clusters or ["none"]


class BenchmarkRunner:
    def __init__(
        self,
        agent,
        evaluator: RetrievalEvaluator,
        judge: MultiJudgeConsensus,
        concurrency: int = 10,
        top_k: int = 5,
    ):
        self.agent = agent
        self.evaluator = evaluator
        self.judge = judge
        self.concurrency = concurrency
        self.top_k = top_k

    async def run_single_test(self, test_case: Dict) -> Dict:
        prepared = await self.prepare_single_test(test_case)
        if prepared.get("prepare_error"):
            judge_result = self._error_judge_result(prepared["error_message"])
        else:
            judge_result = await self.judge.evaluate(test_case, prepared["response"]["answer"])
        return self.finalize_single_test(prepared, judge_result)

    async def prepare_single_test(self, test_case: Dict) -> Dict:
        start = time.perf_counter()
        case_id = test_case.get("id") or test_case.get("case_id")
        expected_ids = expected_ids_for_case(test_case)
        case_type = (test_case.get("metadata") or {}).get("type", test_case.get("case_type", "unknown"))
        difficulty = (test_case.get("metadata") or {}).get("difficulty", test_case.get("difficulty", "unknown"))
        retrieved_ids: List[str] = []
        contexts: List[str] = []
        response = {"answer": "", "metadata": {}}
        retrieval_scores: Dict = {}
        error_type = None
        error_message = None
        try:
            retrieved_ids = [] if not expected_ids else self.evaluator.retrieve(test_case["question"], self.top_k)
            contexts = [f"{doc_id}: {self.evaluator.retriever.corpus.get(doc_id, '')}" for doc_id in retrieved_ids]
            response = await self.agent.query(test_case["question"], case=test_case, contexts=contexts, retrieved_ids=retrieved_ids)
            retrieval_scores = self.evaluator.evaluate_case(test_case, retrieved_ids)
        except Exception as exc:
            error_type = type(exc).__name__
            error_message = str(exc)[:500]
            retrieval_scores = self.evaluator.evaluate_case(test_case, retrieved_ids)
        latency_ms = (time.perf_counter() - start) * 1000
        response["contexts"] = contexts
        response["retrieved_ids"] = retrieved_ids
        return {
            "test_case": test_case,
            "case_id": case_id,
            "expected_ids": expected_ids,
            "case_type": case_type,
            "difficulty": difficulty,
            "retrieved_ids": retrieved_ids,
            "contexts": contexts,
            "response": response,
            "retrieval_scores": retrieval_scores,
            "latency_ms": latency_ms,
            "error_type": error_type,
            "error_message": error_message,
            "prepare_error": error_type is not None,
        }

    def _error_judge_result(self, error_message: str | None) -> Dict:
        message = error_message or "unknown error"
        return {
            "judge_mode": getattr(self.judge, "judge_mode", "unknown"),
            "final_score": 1.0,
            "final_verdict": "fail",
            "agreement": False,
            "agreement_rate": 0.0,
            "individual_scores": {},
            "judge_results": [],
            "reasoning": message,
            "score_gap": 0,
            "token_usage": {"input_tokens": 0, "output_tokens": 0},
        }

    def finalize_single_test(self, prepared: Dict, judge_result: Dict) -> Dict:
        test_case = prepared["test_case"]
        case_id = prepared["case_id"]
        expected_ids = prepared["expected_ids"]
        case_type = prepared["case_type"]
        difficulty = prepared["difficulty"]
        retrieved_ids = prepared["retrieved_ids"]
        contexts = prepared["contexts"]
        response = prepared["response"]
        retrieval_scores = prepared["retrieval_scores"]
        error_type = prepared["error_type"]
        error_message = prepared["error_message"]
        status = "fail" if prepared.get("prepare_error") or judge_result["final_verdict"] != "pass" else "pass"
        if prepared.get("prepare_error"):
            failures = ["provider_error" if "Gemini" in (error_message or "") else "unknown"]
        else:
            failures = classify_failures(test_case, retrieval_scores, judge_result, response["answer"], retrieved_ids)
        input_tokens = approx_tokens(test_case["question"] + " " + " ".join(contexts))
        output_tokens = approx_tokens(response.get("answer", ""))
        judge_tokens = judge_result.get("token_usage", {})
        total_input_tokens = input_tokens + int(judge_tokens.get("input_tokens", 0) or 0)
        total_output_tokens = output_tokens + int(judge_tokens.get("output_tokens", 0) or 0)
        cost = estimate_cost(total_input_tokens, total_output_tokens)
        metadata = response.setdefault("metadata", {})
        metadata.setdefault("model", getattr(self.agent, "name", "unknown"))
        metadata["tokens_used"] = total_input_tokens + total_output_tokens
        metadata["cost_usd"] = round(cost, 8)
        metadata["sources"] = retrieved_ids

        result = CaseBenchmarkResult(
            test_case_id=case_id,
            case_id=case_id,
            question=test_case["question"],
            expected_answer=test_case["expected_answer"],
            expected_retrieval_ids=expected_ids,
            case_type=case_type,
            difficulty=difficulty,
            retrieved_ids=retrieved_ids,
            agent_response=response,
            latency_ms=round(prepared["latency_ms"], 3),
            retrieval=retrieval_scores,
            judge=judge_result,
            token_usage={
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "agent_input_tokens": input_tokens,
                "agent_output_tokens": output_tokens,
                "judge_input_tokens": int(judge_tokens.get("input_tokens", 0) or 0),
                "judge_output_tokens": int(judge_tokens.get("output_tokens", 0) or 0),
            },
            cost_usd=round(cost, 8),
            final_score=judge_result["final_score"],
            final_verdict=judge_result["final_verdict"],
            failure_clusters=failures,
            status=status,
            error_type=error_type,
            error_message=error_message,
        )
        return asdict(result)

    async def run_all(self, dataset: List[Dict], batch_size: int | None = None) -> List[Dict]:
        semaphore = asyncio.Semaphore(batch_size or self.concurrency)

        async def guarded(case: Dict) -> Dict:
            async with semaphore:
                return await self.prepare_single_test(case)

        prepared_rows = await asyncio.gather(*(guarded(case) for case in dataset))
        judge_items = [
            {
                "case": prepared["test_case"],
                "answer": prepared["response"]["answer"],
                "retrieved_ids": prepared["retrieved_ids"],
                "retrieved_contexts": prepared["contexts"],
            }
            for prepared in prepared_rows
            if not prepared.get("prepare_error")
        ]
        judge_results = await self.judge.evaluate_batch(judge_items)
        judge_by_case = {
            (item["case"].get("id") or item["case"].get("case_id")): result
            for item, result in zip(judge_items, judge_results)
        }
        return [
            self.finalize_single_test(
                prepared,
                self._error_judge_result(prepared["error_message"])
                if prepared.get("prepare_error")
                else judge_by_case[prepared["case_id"]],
            )
            for prepared in prepared_rows
        ]


def summarize_results(results: List[Dict], version: str, runtime_seconds: float, concurrency: int, judge_quota: Dict | None = None) -> Dict:
    total = max(len(results), 1)
    latencies = sorted(result["latency_ms"] for result in results)
    p95_index = min(len(latencies) - 1, int(0.95 * len(latencies))) if latencies else 0
    score_pairs = [
        tuple(round(score) for score in result["judge"]["individual_scores"].values())
        for result in results
        if len(result["judge"]["individual_scores"]) >= 2
    ]
    agreement_count = sum(1 for result in results if result["judge"].get("agreement"))
    conflict_count = sum(1 for result in results if result["judge"].get("score_gap", 0) > 1)
    total_input = sum(result["token_usage"]["input_tokens"] for result in results)
    total_output = sum(result["token_usage"]["output_tokens"] for result in results)
    total_cost = sum(result["cost_usd"] for result in results)

    mrr_values = [result["retrieval"]["mrr"] for result in results if result["retrieval"].get("mrr") is not None]
    recall_values = [result["retrieval"]["recall"] for result in results if result["retrieval"].get("recall") is not None]
    modes = [result["judge"].get("judge_mode", "unknown") for result in results]
    if modes and all(mode == "provider_gemini_dual_model" for mode in modes):
        judge_mode = "provider_gemini_dual_model"
    elif any(mode in {"provider_gemini_dual_model", "provider_gemini_single_model"} for mode in modes):
        judge_mode = "provider_gemini_single_model"
    elif modes and all(mode == "offline_fallback" for mode in modes):
        judge_mode = "offline_fallback"
    else:
        judge_mode = max(set(modes), key=modes.count) if modes else "unknown"
    all_judge_rows = [row for result in results for row in result["judge"].get("judge_results", [])]
    judge_interfaces = sorted({row.get("judge") for row in all_judge_rows if row.get("judge")})
    quota = judge_quota or {}
    metrics = {
        "avg_score": sum(result["final_score"] for result in results) / total,
        "hit_rate": sum(result["retrieval"]["hit_rate"] for result in results) / total,
        "mrr": sum(mrr_values) / len(mrr_values) if mrr_values else 0.0,
        "recall": sum(recall_values) / len(recall_values) if recall_values else 0.0,
        "avg_retrieved_count": sum(result["retrieval"]["avg_retrieved_count"] for result in results) / total,
        "agreement_rate": agreement_count / total,
        "cohen_kappa": cohen_kappa(score_pairs),
        "conflict_count": conflict_count,
        "conflict_rate": conflict_count / total,
        "pass_count": sum(1 for result in results if result["final_verdict"] == "pass"),
        "fail_count": sum(1 for result in results if result["final_verdict"] != "pass"),
        "total_runtime_seconds": runtime_seconds,
        "runtime_seconds": runtime_seconds,
        "avg_latency_ms": statistics.mean(latencies) if latencies else 0.0,
        "avg_latency": statistics.mean(latencies) if latencies else 0.0,
        "p95_latency_ms": latencies[p95_index] if latencies else 0.0,
        "p95_latency": latencies[p95_index] if latencies else 0.0,
        "concurrency": concurrency,
        "async_concurrency": concurrency,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_tokens": total_input + total_output,
        "estimated_total_cost_usd": total_cost,
        "total_cost_usd": total_cost,
        "cost_per_case_usd": total_cost / total,
    }
    return {
        "metadata": {
            "version": version,
            "total": len(results),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "judge_mode": judge_mode,
            "judge_interfaces": judge_interfaces,
            "retriever_type": results[0]["retrieval"].get("retriever_type", "unknown") if results else "unknown",
            "fallback_behavior": (
                "Gemini A and Gemini B are used when GEMINI_API_KEY is configured and both models work. "
                "If one Gemini model fails, the successful Gemini judge is paired with one deterministic judge. "
                "If no Gemini key is present or both models fail, two deterministic heuristic judges score the case."
            ),
        },
        "metrics": metrics,
        "judge": {
            "provider": "gemini",
            "judge_mode": judge_mode,
            "judge_a_model": next((row.get("model") for row in all_judge_rows if row.get("judge") == "GeminiJudgeA"), None),
            "judge_b_model": next((row.get("model") for row in all_judge_rows if row.get("judge") == "GeminiJudgeB"), None),
            "dual_provider": judge_mode == "provider_gemini_dual_model",
            **quota,
            "conflict_count": conflict_count,
            "conflict_rate": conflict_count / total,
        },
        "retrieval": {
            "retriever_type": results[0]["retrieval"].get("retriever_type", "unknown") if results else "unknown",
            "top_k": results[0]["retrieval"].get("top_k", None) if results else None,
        },
        "retrieval_quality_note": (
            "Answer quality is gated by retrieval: misses and low recall reduce context support, "
            "which increases incomplete answers, missing citations, and hallucination risk."
        ),
        "cost_reduction_plan": [
            "Use the cheaper heuristic or small-model judge for easy/high-confidence cases.",
            "Escalate to expensive provider judges only for conflicts and release-borderline cases.",
            "Cache judge results by case_id, answer hash, and rubric version.",
            "Lower max output tokens for factual and citation-only checks.",
        ],
    }
