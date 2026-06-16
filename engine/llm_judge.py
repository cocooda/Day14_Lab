import asyncio
import hashlib
import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from math import ceil
from pathlib import Path
from typing import Dict, List, Tuple

from engine.retrieval_eval import expected_ids_for_case, tokenize


JUDGE_RUBRIC_VERSION = "gemini_consensus_rubric_v2"


def _overlap_ratio(reference: str, answer: str) -> float:
    ref_tokens = set(tokenize(reference))
    answer_tokens = set(tokenize(answer))
    if not ref_tokens:
        return 0.0
    return len(ref_tokens.intersection(answer_tokens)) / len(ref_tokens)


def _score_from_ratio(ratio: float) -> float:
    if ratio >= 0.72:
        return 5.0
    if ratio >= 0.55:
        return 4.0
    if ratio >= 0.38:
        return 3.0
    if ratio >= 0.2:
        return 2.0
    return 1.0


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _case_id(case: Dict) -> str:
    return str(case.get("id") or case.get("case_id") or "unknown")


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _eval_limit_set() -> bool:
    return bool(str(os.getenv("EVAL_LIMIT", "")).strip())


def judge_b_final_enabled() -> bool:
    return (
        _truthy(os.getenv("FINAL_PROVIDER_RUN"))
        and _truthy(os.getenv("ALLOW_JUDGE_B_FINAL"))
        and bool(str(os.getenv("GEMINI_JUDGE_B_MODEL", "")).strip())
        and not _eval_limit_set()
    )


def plan_provider_requests(
    total_cases: int,
    batch_size: int,
    uncached_a: int | None = None,
    uncached_b: int | None = None,
) -> Dict:
    batch = max(1, int(batch_size or 1))
    a_cases = total_cases if uncached_a is None else uncached_a
    b_cases = total_cases if uncached_b is None else uncached_b
    judge_a_planned = ceil(max(0, a_cases) / batch)
    judge_b_planned = ceil(max(0, b_cases) / batch)
    daily_budget = int(os.getenv("JUDGE_B_DAILY_REQUEST_BUDGET", "20"))
    retry_reserved = int(os.getenv("JUDGE_B_RESERVED_RETRY_BUDGET", "4"))
    usable_planned_budget = max(0, daily_budget - retry_reserved)
    max_planned = int(os.getenv("JUDGE_B_MAX_PLANNED_REQUESTS", str(usable_planned_budget)))
    budget_blocked = judge_b_planned > max_planned
    return {
        "total_cases": total_cases,
        "batch_size": batch,
        "judge_a_planned_requests": judge_a_planned,
        "judge_b_planned_requests": judge_b_planned,
        "total_planned_provider_requests": judge_a_planned + judge_b_planned,
        "judge_b_daily_request_budget": daily_budget,
        "judge_b_reserved_retry_budget": retry_reserved,
        "judge_b_max_planned_requests": max_planned,
        "judge_b_usable_planned_budget": usable_planned_budget,
        "budget_blocked": budget_blocked,
        "budget_block_reason": (
            f"Judge B planned requests {judge_b_planned} exceed JUDGE_B_MAX_PLANNED_REQUESTS={max_planned}."
            if budget_blocked
            else ""
        ),
    }


@dataclass
class JudgeResult:
    name: str
    score: float
    reason: str
    faithfulness: float
    relevance: float
    completeness: float
    citation_score: float
    hallucination_risk: str
    hallucination_detected: bool
    extra: Dict = field(default_factory=dict)


def _extract_json_object(text: str) -> Dict:
    stripped = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        stripped = fence.group(1).strip()
    if not stripped.startswith("{"):
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start != -1 and end != -1 and end > start:
            stripped = stripped[start : end + 1]
    return json.loads(stripped)


def parse_judge_json(text: str, judge_name: str) -> JudgeResult:
    try:
        payload = _extract_json_object(text)
        hallucination_risk = str(payload.get("hallucination_risk", "high")).lower()
        if hallucination_risk not in {"low", "medium", "high"}:
            hallucination_risk = "high"
        score = _clamp(float(payload["score"]), 1.0, 5.0)
        faithfulness = _clamp(float(payload.get("faithfulness", 0.0)), 0.0, 1.0)
        relevance = _clamp(float(payload.get("relevance", payload.get("relevancy", 0.0))), 0.0, 1.0)
        completeness = _clamp(float(payload.get("completeness", 0.0)), 0.0, 1.0)
        citation = _clamp(float(payload.get("citation_correctness", payload.get("citation_score", 0.0))), 0.0, 1.0)
        return JudgeResult(
            name=judge_name,
            score=score,
            reason=str(payload.get("reason", ""))[:500],
            faithfulness=faithfulness,
            relevance=relevance,
            completeness=completeness,
            citation_score=citation,
            hallucination_risk=hallucination_risk,
            hallucination_detected=hallucination_risk == "high",
        )
    except Exception as exc:
        return JudgeResult(
            name=judge_name,
            score=1.0,
            reason="Judge response could not be parsed as strict rubric JSON.",
            faithfulness=0.0,
            relevance=0.0,
            completeness=0.0,
            citation_score=0.0,
            hallucination_risk="high",
            hallucination_detected=True,
            extra={"parse_error": str(exc)[:200]},
        )


def _result_from_payload(payload: Dict, judge_name: str) -> JudgeResult:
    hallucination_risk = str(payload.get("hallucination_risk", "high")).lower()
    if hallucination_risk not in {"low", "medium", "high"}:
        hallucination_risk = "high"
    score = _clamp(float(payload["score"]), 1.0, 5.0)
    faithfulness = _clamp(float(payload.get("faithfulness", 0.0)), 0.0, 1.0)
    relevance = _clamp(float(payload.get("relevance", payload.get("relevancy", 0.0))), 0.0, 1.0)
    completeness = _clamp(float(payload.get("completeness", 0.0)), 0.0, 1.0)
    citation = _clamp(float(payload.get("citation_correctness", payload.get("citation_score", 0.0))), 0.0, 1.0)
    return JudgeResult(
        name=judge_name,
        score=score,
        reason=str(payload.get("reason", ""))[:500],
        faithfulness=faithfulness,
        relevance=relevance,
        completeness=completeness,
        citation_score=citation,
        hallucination_risk=hallucination_risk,
        hallucination_detected=hallucination_risk == "high",
    )


def parse_judge_batch_json(text: str, judge_name: str, expected_case_ids: List[str]) -> Dict[str, JudgeResult]:
    try:
        payload = _extract_json_object(text)
        rows = payload.get("results", [])
        parsed: Dict[str, JudgeResult] = {}
        for row in rows:
            case_id = str(row.get("test_case_id", ""))
            if case_id:
                parsed[case_id] = _result_from_payload(row, judge_name)
        missing = [case_id for case_id in expected_case_ids if case_id not in parsed]
        if missing:
            raise ValueError(f"missing results for case ids: {', '.join(missing)}")
        return parsed
    except Exception as exc:
        return {
            case_id: JudgeResult(
                name=judge_name,
                score=1.0,
                reason="Judge batch response could not be parsed as strict rubric JSON.",
                faithfulness=0.0,
                relevance=0.0,
                completeness=0.0,
                citation_score=0.0,
                hallucination_risk="high",
                hallucination_detected=True,
                extra={"parse_error": str(exc)[:200]},
            )
            for case_id in expected_case_ids
        }


class BaseJudge:
    name = "base"

    async def score(self, case: Dict, answer: str) -> JudgeResult:
        raise NotImplementedError


class FaithfulnessHeuristicJudge(BaseJudge):
    name = "FaithfulnessHeuristicJudge"

    async def score(self, case: Dict, answer: str) -> JudgeResult:
        ratio = _overlap_ratio(case["expected_answer"], answer)
        hallucination = any(marker in answer.lower() for marker in ["private data", "hidden prompt", "unsupported claim"])
        score = _score_from_ratio(ratio)
        if hallucination:
            score = min(score, 2.0)
        return JudgeResult(
            name=self.name,
            score=score,
            reason=f"Expected-answer token overlap is {ratio:.2f}; hallucination={hallucination}.",
            faithfulness=ratio,
            relevance=min(1.0, ratio + 0.1),
            completeness=ratio,
            citation_score=1.0,
            hallucination_risk="high" if hallucination else "low",
            hallucination_detected=hallucination,
        )


class RelevanceCitationHeuristicJudge(BaseJudge):
    name = "RelevanceCitationHeuristicJudge"

    async def score(self, case: Dict, answer: str) -> JudgeResult:
        ratio = _overlap_ratio(case["question"] + " " + case["expected_answer"], answer)
        required = expected_ids_for_case(case)
        present = sum(1 for doc_id in required if f"[{doc_id}]" in answer)
        citation_score = present / len(required) if required else 1.0
        score = _score_from_ratio(ratio)
        if required and citation_score < 1.0:
            score = min(score, 3.0)
        return JudgeResult(
            name=self.name,
            score=score,
            reason=f"Question/answer relevance overlap is {ratio:.2f}; citation coverage is {citation_score:.2f}.",
            faithfulness=max(0.0, ratio - 0.05),
            relevance=ratio,
            completeness=ratio,
            citation_score=citation_score,
            hallucination_risk="low",
            hallucination_detected=False,
        )


class GeminiJudge(BaseJudge):
    def __init__(self, name: str, model: str, timeout_seconds: float):
        self.name = name
        self.model = model
        self.timeout_seconds = timeout_seconds

    def build_prompt(self, case: Dict, answer: str) -> str:
        expected_ids = expected_ids_for_case(case)
        return (
            "You are an independent benchmark judge. Evaluate the answer against the question, "
            "expected answer, retrieved context, and required source ids. Use the same rubric for every answer: "
            "faithfulness, relevance, completeness, citation correctness, and hallucination risk.\n"
            "Return strict JSON only with this schema:\n"
            '{"score": 1, "faithfulness": 0.0, "relevance": 0.0, "completeness": 0.0, '
            '"citation_correctness": 0.0, "hallucination_risk": "low", "reason": "short explanation"}\n'
            "Score must be an integer or float from 1 to 5. Rubric fields must be 0.0 to 1.0.\n\n"
            f"Case ID: {_case_id(case)}\n"
            f"Question: {case.get('question')}\n"
            f"Expected answer: {case.get('expected_answer')}\n"
            f"Retrieved/grounding context: {case.get('context', '')}\n"
            f"Required source IDs: {', '.join(expected_ids) if expected_ids else 'none'}\n"
            f"Case type: {(case.get('metadata') or {}).get('type', case.get('case_type', 'unknown'))}\n"
            f"Answer to judge: {answer}\n"
        )

    def build_batch_prompt(self, items: List[Dict]) -> str:
        batch_cases = []
        for item in items:
            case = item["case"]
            retrieved_contexts = item.get("retrieved_contexts") or item.get("contexts") or []
            truncated_contexts = [str(context)[:1000] for context in retrieved_contexts[:2]]
            batch_cases.append(
                {
                    "test_case_id": _case_id(case),
                    "question": case.get("question", ""),
                    "expected_answer": case.get("expected_answer", ""),
                    "agent_answer": item.get("answer", ""),
                    "expected_retrieval_ids": expected_ids_for_case(case),
                    "retrieved_ids": item.get("retrieved_ids", []),
                    "retrieved_contexts": truncated_contexts,
                }
            )
        return (
            "You are an independent benchmark judge. Evaluate each answer against its question, expected answer, "
            "retrieved ids, and retrieved contexts. Use faithfulness, relevance, completeness, citation correctness, "
            "and hallucination risk. Return one result for every input test_case_id.\n"
            "Return only JSON. No markdown fences. No extra prose. score must be 1-5. Numeric submetrics must be 0-1. "
            'hallucination_risk must be "low", "medium", or "high".\n'
            "Return exactly this shape:\n"
            '{"results":[{"test_case_id":"case_001","score":4,"faithfulness":0.9,"relevance":0.9,'
            '"completeness":0.8,"citation_correctness":1.0,"hallucination_risk":"low",'
            '"reason":"short explanation"}]}\n\n'
            f"Cases:\n{json.dumps(batch_cases, ensure_ascii=False)}"
        )

    def _generate_text(self, prompt: str) -> str:
        try:
            from google import genai
        except Exception as exc:
            raise RuntimeError("google-genai package is not installed") from exc

        client = genai.Client()
        response = client.models.generate_content(model=self.model, contents=prompt)
        return response.text or ""

    async def score(self, case: Dict, answer: str) -> JudgeResult:
        prompt = self.build_prompt(case, answer)
        text = await asyncio.to_thread(self._generate_text, prompt)
        result = parse_judge_json(text, self.name)
        result.extra.update(
            {
                "provider": "gemini",
                "model": self.model,
                "input_tokens": len(tokenize(prompt)),
                "output_tokens": len(tokenize(text)),
            }
        )
        return result

    async def score_batch(self, items: List[Dict]) -> Dict[str, JudgeResult]:
        prompt = self.build_batch_prompt(items)
        text = await asyncio.to_thread(self._generate_text, prompt)
        case_ids = [_case_id(item["case"]) for item in items]
        results = parse_judge_batch_json(text, self.name, case_ids)
        for result in results.values():
            result.extra.update(
                {
                    "provider": "gemini",
                    "model": self.model,
                    "input_tokens": len(tokenize(prompt)) // max(1, len(items)),
                    "output_tokens": len(tokenize(text)) // max(1, len(items)),
                }
            )
        return results


class GeminiRateLimiter:
    def __init__(self, requests_per_minute: int):
        self.requests_per_minute = max(1, requests_per_minute)
        self.min_interval_seconds = 60.0 / self.requests_per_minute
        self._lock = asyncio.Lock()
        self._last_request_at = 0.0

    async def wait(self) -> None:
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_request_at
            wait_seconds = self.min_interval_seconds - elapsed
            if wait_seconds > 0:
                await asyncio.sleep(wait_seconds)
            self._last_request_at = time.monotonic()


class JudgeResultCache:
    def __init__(self, path: str, enabled: bool):
        self.path = Path(path)
        self.enabled = enabled
        self.data: Dict[str, Dict] = {}
        if self.enabled and self.path.exists():
            try:
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                self.data = {}

    def key(self, model_name: str, item: Dict) -> str:
        case = item["case"]
        payload = {
            "question": case.get("question", ""),
            "expected_answer": case.get("expected_answer", ""),
            "agent_answer": item.get("answer", ""),
            "retrieved_ids": item.get("retrieved_ids", []),
            "retrieved_contexts": item.get("retrieved_contexts") or item.get("contexts") or [],
        }
        digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
        return "|".join([model_name, JUDGE_RUBRIC_VERSION, _case_id(case), digest])

    def get(self, model_name: str, item: Dict, judge_name: str) -> JudgeResult | None:
        if not self.enabled:
            return None
        cached = self.data.get(self.key(model_name, item))
        if not cached:
            return None
        result = JudgeResult(**cached)
        result.name = judge_name
        result.extra.setdefault("cache_hit", True)
        return result

    def set(self, model_name: str, item: Dict, result: JudgeResult) -> None:
        if not self.enabled:
            return
        self.data[self.key(model_name, item)] = {
            "name": result.name,
            "score": result.score,
            "reason": result.reason,
            "faithfulness": result.faithfulness,
            "relevance": result.relevance,
            "completeness": result.completeness,
            "citation_score": result.citation_score,
            "hallucination_risk": result.hallucination_risk,
            "hallucination_detected": result.hallucination_detected,
            "extra": {key: value for key, value in result.extra.items() if key != "cache_hit"},
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")


class MultiJudgeConsensus:
    """Gemini-first two-judge consensus engine with explicit fallback modes."""

    def __init__(self, timeout_seconds: float | None = None, max_retries: int | None = None):
        self.timeout_seconds = timeout_seconds or float(os.getenv("JUDGE_TIMEOUT_SECONDS", "45"))
        self.max_retries = max_retries if max_retries is not None else int(os.getenv("JUDGE_MAX_RETRIES", "2"))
        self.retry_base_seconds = float(os.getenv("JUDGE_RETRY_BASE_SECONDS", "15"))
        self.retry_max_seconds = float(os.getenv("JUDGE_RETRY_MAX_SECONDS", "120"))
        self.batch_size = max(1, int(os.getenv("JUDGE_BATCH_SIZE", "1")))
        self.requests_per_minute = max(1, int(os.getenv("JUDGE_REQUESTS_PER_MINUTE", "6")))
        self.judge_a_model = os.getenv("GEMINI_JUDGE_A_MODEL", "gemini-3.1-flash-lite")
        self.judge_b_model = os.getenv("GEMINI_JUDGE_B_MODEL", "gemini-2.5-flash")
        self.final_provider_run = _truthy(os.getenv("FINAL_PROVIDER_RUN"))
        self.allow_judge_b_final = _truthy(os.getenv("ALLOW_JUDGE_B_FINAL"))
        self.judge_b_daily_request_budget = int(os.getenv("JUDGE_B_DAILY_REQUEST_BUDGET", "20"))
        self.judge_b_reserved_retry_budget = int(os.getenv("JUDGE_B_RESERVED_RETRY_BUDGET", "4"))
        usable = max(0, self.judge_b_daily_request_budget - self.judge_b_reserved_retry_budget)
        self.judge_b_max_planned_requests = int(os.getenv("JUDGE_B_MAX_PLANNED_REQUESTS", str(usable)))
        self.cache_enabled = _truthy(os.getenv("JUDGE_CACHE_ENABLED", "true"))
        self.cache = JudgeResultCache(os.getenv("JUDGE_CACHE_PATH", ".cache/judge_results.json"), self.cache_enabled)
        self.rate_limiter = GeminiRateLimiter(self.requests_per_minute)
        self.cache_hits = 0
        self.cache_misses = 0
        self.judge_a_actual_requests = 0
        self.judge_b_actual_requests = 0
        self.rate_limit_retry_count = 0
        self.provider_error_count = 0
        self.expected_provider_requests = 0
        self.judge_a_planned_requests = 0
        self.judge_b_planned_requests = 0
        self.budget_block_reason = ""
        self.judge_mode = "offline_fallback"
        self.judges: List[BaseJudge] = self._build_judges()

    def _build_judges(self) -> List[BaseJudge]:
        if str(os.getenv("JUDGE_PROVIDER", "")).strip().lower() == "offline":
            self.judge_mode = "offline_fallback"
            return [FaithfulnessHeuristicJudge(), RelevanceCitationHeuristicJudge()]
        if os.getenv("GEMINI_API_KEY"):
            judge_a = GeminiJudge("GeminiJudgeA", self.judge_a_model, self.timeout_seconds)
            if judge_b_final_enabled():
                self.judge_mode = "provider_gemini_dual_model"
                return [judge_a, GeminiJudge("GeminiJudgeB", self.judge_b_model, self.timeout_seconds)]
            self.judge_mode = "provider_gemini_single_model"
            return [judge_a, RelevanceCitationHeuristicJudge()]
        self.judge_mode = "offline_fallback"
        return [FaithfulnessHeuristicJudge(), RelevanceCitationHeuristicJudge()]

    def can_call_judge_b(self) -> bool:
        return self.judge_b_actual_requests < self.judge_b_daily_request_budget

    def quota_metadata(self) -> Dict:
        return {
            "batch_size": self.batch_size,
            "requests_per_minute": self.requests_per_minute,
            "final_provider_run": self.final_provider_run,
            "allow_judge_b_final": self.allow_judge_b_final,
            "expected_provider_requests": self.expected_provider_requests,
            "judge_a_planned_requests": self.judge_a_planned_requests,
            "judge_b_planned_requests": self.judge_b_planned_requests,
            "judge_a_actual_requests": self.judge_a_actual_requests,
            "judge_b_actual_requests": self.judge_b_actual_requests,
            "judge_b_daily_request_budget": self.judge_b_daily_request_budget,
            "judge_b_reserved_retry_budget": self.judge_b_reserved_retry_budget,
            "judge_b_budget_remaining": max(0, self.judge_b_daily_request_budget - self.judge_b_actual_requests),
            "cache_enabled": self.cache_enabled,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "rate_limit_retry_count": self.rate_limit_retry_count,
            "provider_error_count": self.provider_error_count,
            "budget_block_reason": self.budget_block_reason,
        }

    async def _score_with_retry(self, judge: BaseJudge, case: Dict, answer: str) -> JudgeResult:
        for attempt in range(self.max_retries + 1):
            try:
                return await asyncio.wait_for(judge.score(case, answer), timeout=self.timeout_seconds)
            except Exception as exc:
                if attempt >= self.max_retries:
                    detail = str(exc)[:500]
                    return JudgeResult(
                        name=judge.name,
                        score=1.0,
                        reason=f"Judge failed after retry: {type(exc).__name__}: {detail}",
                        faithfulness=0.0,
                        relevance=0.0,
                        completeness=0.0,
                        citation_score=0.0,
                        hallucination_risk="high",
                        hallucination_detected=True,
                        extra={"provider_error": f"{type(exc).__name__}: {detail}"},
                    )
                await asyncio.sleep(0.25 * (2**attempt))
        raise RuntimeError("unreachable")

    async def _score_provider_batches(self, judge: GeminiJudge, items: List[Dict], is_judge_b: bool) -> Dict[str, JudgeResult]:
        results: Dict[str, JudgeResult] = {}
        uncached: List[Dict] = []
        for item in items:
            cached = self.cache.get(judge.model, item, judge.name)
            if cached is not None:
                self.cache_hits += 1
                results[_case_id(item["case"])] = cached
            else:
                self.cache_misses += 1
                uncached.append(item)

        batches = [uncached[index : index + self.batch_size] for index in range(0, len(uncached), self.batch_size)]
        for batch_index, batch in enumerate(batches, start=1):
            if is_judge_b and not self.can_call_judge_b():
                self.judge_mode = "provider_gemini_budget_exhausted"
                self.provider_error_count += len(batch)
                for item in batch:
                    case_id = _case_id(item["case"])
                    results[case_id] = JudgeResult(
                        name=judge.name,
                        score=1.0,
                        reason="Judge B budget exhausted before this batch.",
                        faithfulness=0.0,
                        relevance=0.0,
                        completeness=0.0,
                        citation_score=0.0,
                        hallucination_risk="high",
                        hallucination_detected=True,
                        extra={"budget_exhausted": True, "model": judge.model, "provider": "gemini"},
                    )
                continue

            planned_number = self.judge_b_actual_requests + 1 if is_judge_b else self.judge_a_actual_requests + 1
            print(f"Gemini judge request model={judge.model} batch={batch_index}/{len(batches)} planned_request={planned_number}")
            for attempt in range(self.max_retries + 1):
                if is_judge_b and not self.can_call_judge_b():
                    self.judge_mode = "provider_gemini_budget_exhausted"
                    break
                if is_judge_b:
                    self.judge_b_actual_requests += 1
                else:
                    self.judge_a_actual_requests += 1
                try:
                    await self.rate_limiter.wait()
                    batch_results = await asyncio.wait_for(judge.score_batch(batch), timeout=self.timeout_seconds)
                    for item in batch:
                        case_id = _case_id(item["case"])
                        result = batch_results[case_id]
                        self.cache.set(judge.model, item, result)
                        results[case_id] = result
                    break
                except Exception as exc:
                    detail = str(exc)[:500]
                    retryable = "429" in detail or "RESOURCE_EXHAUSTED" in detail.upper()
                    if retryable:
                        self.rate_limit_retry_count += 1
                    if attempt >= self.max_retries or (is_judge_b and not self.can_call_judge_b()):
                        self.provider_error_count += len(batch)
                        for item in batch:
                            case_id = _case_id(item["case"])
                            results[case_id] = JudgeResult(
                                name=judge.name,
                                score=1.0,
                                reason=f"Judge failed after retry: {type(exc).__name__}: {detail}",
                                faithfulness=0.0,
                                relevance=0.0,
                                completeness=0.0,
                                citation_score=0.0,
                                hallucination_risk="high",
                                hallucination_detected=True,
                                extra={"provider_error": f"{type(exc).__name__}: {detail}", "model": judge.model, "provider": "gemini"},
                            )
                        break
                    delay = min(self.retry_max_seconds, self.retry_base_seconds * (2**attempt))
                    delay += random.uniform(0, min(1.0, delay * 0.1))
                    await asyncio.sleep(delay)
        return results

    async def _score_heuristic_batch(self, judge: BaseJudge, items: List[Dict]) -> Dict[str, JudgeResult]:
        scored = await asyncio.gather(*(judge.score(item["case"], item.get("answer", "")) for item in items))
        return {_case_id(item["case"]): result for item, result in zip(items, scored)}

    async def _score_active_judges(self, case: Dict, answer: str) -> Tuple[str, List[JudgeResult]]:
        results = await asyncio.gather(*(self._score_with_retry(judge, case, answer) for judge in self.judges))
        provider_failures = [result for result in results if "provider_error" in result.extra]
        if self.judge_mode != "provider_gemini_dual_model" or not provider_failures:
            return self.judge_mode, results

        successful = [result for result in results if "provider_error" not in result.extra]
        if len(successful) == 1:
            heuristic = await RelevanceCitationHeuristicJudge().score(case, answer)
            return "provider_gemini_single_model", [successful[0], heuristic]

        fallback_results = await asyncio.gather(
            FaithfulnessHeuristicJudge().score(case, answer),
            RelevanceCitationHeuristicJudge().score(case, answer),
        )
        for failure in provider_failures:
            fallback_results[0].extra.setdefault("provider_errors", []).append(failure.extra["provider_error"])
        return "offline_fallback", list(fallback_results)

    def _combine_results(self, case: Dict, answer: str, judge_mode: str, results: List[JudgeResult]) -> Dict:
        score_a, score_b = results[0].score, results[1].score
        required = expected_ids_for_case(case)
        required_citation_missing = bool(required) and any(f"[{doc_id}]" not in answer for doc_id in required)
        hallucination_high = any(result.hallucination_risk == "high" for result in results)

        if abs(score_a - score_b) <= 1:
            final_score = (score_a + score_b) / 2
            resolution = "average_within_one_point"
        else:
            def tie_break_value(result: JudgeResult) -> float:
                hallucination_penalty = 0.0 if result.hallucination_risk == "high" else 0.1
                return ((result.citation_score * 0.4) + (result.faithfulness * 0.35) + (result.completeness * 0.25) + hallucination_penalty) * 5

            final_score = min(score_a, score_b, max(tie_break_value(result) for result in results))
            resolution = "tie_break_citation_faithfulness_completeness_hallucination"

        if required_citation_missing:
            final_score = min(final_score, 3.0)
            resolution += "_citation_cap"
        if hallucination_high:
            retrieval_supported = bool(required) and not required_citation_missing
            final_score = min(final_score, 2.0 if not retrieval_supported else 3.0)
            resolution += "_hallucination_cap"

        rounded_scores = [round(score_a), round(score_b)]
        final_verdict = "pass" if final_score >= 3.5 and not hallucination_high and not required_citation_missing else "fail"
        judge_rows = []
        for result in results:
            row = {
                "judge": result.name,
                "score": result.score,
                "reason": result.reason,
                "faithfulness": result.faithfulness,
                "relevance": result.relevance,
                "relevancy": result.relevance,
                "completeness": result.completeness,
                "citation_correctness": result.citation_score,
                "citation_score": result.citation_score,
                "hallucination_risk": result.hallucination_risk,
            }
            row.update(result.extra)
            judge_rows.append(row)

        return {
            "judge_mode": judge_mode,
            "rubric_version": JUDGE_RUBRIC_VERSION,
            "judge_results": judge_rows,
            "individual_scores": {result.name: result.score for result in results},
            "judge_a_score": score_a,
            "judge_b_score": score_b,
            "final_score": round(final_score, 3),
            "final_verdict": final_verdict,
            "agreement": rounded_scores[0] == rounded_scores[1],
            "agreement_rate": 1.0 if rounded_scores[0] == rounded_scores[1] else 0.0,
            "score_gap": abs(score_a - score_b),
            "conflict_resolution_reason": resolution,
            "resolution": resolution,
            "reasoning": "; ".join(result.reason for result in results)[:1000],
            "required_citation_missing": required_citation_missing,
            "hallucination_detected": hallucination_high,
            "token_usage": {
                "input_tokens": sum(int(result.extra.get("input_tokens", 0) or 0) for result in results),
                "output_tokens": sum(int(result.extra.get("output_tokens", 0) or 0) for result in results),
            },
        }

    async def evaluate_batch(self, items: List[Dict]) -> List[Dict]:
        if not items:
            return []

        active_mode = self.judge_mode
        if active_mode == "provider_gemini_dual_model":
            judge_a = self.judges[0]
            judge_b = self.judges[1]
            uncached_a = sum(1 for item in items if self.cache.get(getattr(judge_a, "model", ""), item, judge_a.name) is None)
            uncached_b = sum(1 for item in items if self.cache.get(getattr(judge_b, "model", ""), item, judge_b.name) is None)
            plan = plan_provider_requests(len(items), self.batch_size, uncached_a=uncached_a, uncached_b=uncached_b)
            self.expected_provider_requests = plan["total_planned_provider_requests"]
            self.judge_a_planned_requests = plan["judge_a_planned_requests"]
            self.judge_b_planned_requests = plan["judge_b_planned_requests"]
            if plan["budget_blocked"]:
                self.judge_mode = "provider_gemini_budget_blocked"
                self.budget_block_reason = plan["budget_block_reason"]
                active_mode = self.judge_mode
                a_results = await self._score_provider_batches(judge_a, items, is_judge_b=False)
                b_results = await self._score_heuristic_batch(RelevanceCitationHeuristicJudge(), items)
            else:
                a_results = await self._score_provider_batches(judge_a, items, is_judge_b=False)
                b_results = await self._score_provider_batches(judge_b, items, is_judge_b=True)
                if any(result.extra.get("provider_error") or result.extra.get("budget_exhausted") for result in b_results.values()):
                    active_mode = self.judge_mode if self.judge_mode.startswith("provider_gemini_budget") else "mixed_provider_fallback"
            return [
                self._combine_results(
                    item["case"],
                    item.get("answer", ""),
                    active_mode,
                    [a_results[_case_id(item["case"])], b_results[_case_id(item["case"])]],
                )
                for item in items
            ]

        if active_mode == "provider_gemini_single_model":
            judge_a = self.judges[0]
            uncached_a = sum(1 for item in items if self.cache.get(getattr(judge_a, "model", ""), item, judge_a.name) is None)
            plan = plan_provider_requests(len(items), self.batch_size, uncached_a=uncached_a, uncached_b=0)
            self.expected_provider_requests = plan["total_planned_provider_requests"]
            self.judge_a_planned_requests = plan["judge_a_planned_requests"]
            self.judge_b_planned_requests = 0
            a_results = await self._score_provider_batches(judge_a, items, is_judge_b=False)
            b_results = await self._score_heuristic_batch(RelevanceCitationHeuristicJudge(), items)
            return [
                self._combine_results(
                    item["case"],
                    item.get("answer", ""),
                    active_mode,
                    [a_results[_case_id(item["case"])], b_results[_case_id(item["case"])]],
                )
                for item in items
            ]

        first = await self._score_heuristic_batch(FaithfulnessHeuristicJudge(), items)
        second = await self._score_heuristic_batch(RelevanceCitationHeuristicJudge(), items)
        return [
            self._combine_results(
                item["case"],
                item.get("answer", ""),
                active_mode,
                [first[_case_id(item["case"])], second[_case_id(item["case"])]],
            )
            for item in items
        ]

    async def evaluate(self, case: Dict, answer: str) -> Dict:
        return (await self.evaluate_batch([{"case": case, "answer": answer}]))[0]

    async def evaluate_multi_judge(self, question: str, answer: str, ground_truth: str) -> Dict:
        case = {
            "id": "adhoc",
            "question": question,
            "expected_answer": ground_truth,
            "context": ground_truth,
            "expected_retrieval_ids": re.findall(r"doc_\d+|DOC-\d+", ground_truth),
            "metadata": {"type": "fact-check", "difficulty": "medium"},
        }
        return await self.evaluate(case, answer)


def cohen_kappa(pairs: List[Tuple[int, int]]) -> float:
    if not pairs:
        return 0.0
    labels = sorted({label for pair in pairs for label in pair})
    total = len(pairs)
    observed = sum(1 for a, b in pairs if a == b) / total
    left_counts = {label: sum(1 for a, _ in pairs if a == label) for label in labels}
    right_counts = {label: sum(1 for _, b in pairs if b == label) for label in labels}
    expected = sum((left_counts[label] / total) * (right_counts[label] / total) for label in labels)
    if expected >= 1.0:
        return 1.0
    return (observed - expected) / (1 - expected)


LLMJudge = MultiJudgeConsensus
