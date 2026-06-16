import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from data.synthetic_gen import build_cases, write_jsonl
from engine.llm_judge import MultiJudgeConsensus, plan_provider_requests, parse_judge_json
from engine.retrieval_eval import RetrievalEvaluator, build_document_corpus, load_markdown_corpus
from main import build_regression


class SyntheticGenerationTests(unittest.TestCase):
    def test_build_cases_has_required_schema_and_minimum_size(self):
        cases = build_cases()

        self.assertGreaterEqual(len(cases), 50)
        required = {
            "id",
            "question",
            "expected_answer",
            "context",
            "expected_retrieval_ids",
            "metadata",
        }
        self.assertTrue(required.issubset(cases[0]))
        normal_cases = [case for case in cases if case["metadata"]["type"] != "out-of-context"]
        self.assertTrue(all(case["expected_retrieval_ids"] for case in normal_cases))

        case_types = {case["metadata"]["type"] for case in cases}
        self.assertTrue({"fact-check", "adversarial", "out-of-context", "ambiguous", "conflicting", "multi-turn"}.issubset(case_types))

    def test_build_cases_uses_legal_seed_document_ids(self):
        cases = build_cases()
        doc_ids = {doc_id for case in cases for doc_id in case["expected_retrieval_ids"]}

        self.assertIn("bo-luat-hinh-su-2015-chuong-xx-toi-pham-ma-tuy", doc_ids)
        self.assertIn("luat-phong-chong-ma-tuy-2021", doc_ids)
        self.assertIn("nghi-dinh-57-2022-danh-muc-chat-ma-tuy", doc_ids)
        self.assertIn("nghi-dinh-105-2021-huong-dan-luat-phong-chong-ma-tuy", doc_ids)
        self.assertNotIn("doc_001", doc_ids)

    def test_write_jsonl_round_trips_cases(self):
        cases = build_cases()[:3]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "golden_set.jsonl"
            write_jsonl(cases, path)

            loaded = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(cases, loaded)


class RetrievalMetricTests(unittest.TestCase):
    def test_load_markdown_corpus_uses_filename_stems_as_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus_dir = Path(tmp)
            (corpus_dir / "legal-doc-one.md").write_text("first document", encoding="utf-8")
            (corpus_dir / "legal-doc-two.md").write_text("second document", encoding="utf-8")

            corpus = load_markdown_corpus(corpus_dir)

        self.assertEqual(
            corpus,
            {"legal-doc-one": "first document", "legal-doc-two": "second document"},
        )

    def test_retrieval_metrics_use_ground_truth_rank(self):
        evaluator = RetrievalEvaluator(build_document_corpus(build_cases()), top_k=3)
        metrics = evaluator.evaluate_case(
            {"expected_retrieval_ids": ["doc_b", "doc_c"]},
            ["doc_x", "doc_c", "doc_b"],
        )

        self.assertEqual(metrics["hit_rate"], 1.0)
        self.assertEqual(metrics["mrr"], 0.5)
        self.assertEqual(metrics["recall"], 1.0)


class JudgeConsensusTests(unittest.IsolatedAsyncioTestCase):
    async def test_provider_mode_uses_single_gemini_model_until_final_run_is_allowed(self):
        env = {
            "GEMINI_API_KEY": "test-gemini-key",
            "GEMINI_JUDGE_A_MODEL": "gemini-a-test",
            "GEMINI_JUDGE_B_MODEL": "gemini-b-test",
        }

        with patch.dict("os.environ", env, clear=True):
            consensus = MultiJudgeConsensus()

        self.assertEqual(consensus.judge_mode, "provider_gemini_single_model")
        self.assertEqual([judge.name for judge in consensus.judges], ["GeminiJudgeA", "RelevanceCitationHeuristicJudge"])

    async def test_provider_mode_uses_two_gemini_models_only_for_full_final_run(self):
        env = {
            "GEMINI_API_KEY": "test-gemini-key",
            "GEMINI_JUDGE_A_MODEL": "gemini-a-test",
            "GEMINI_JUDGE_B_MODEL": "gemini-b-test",
            "FINAL_PROVIDER_RUN": "true",
            "ALLOW_JUDGE_B_FINAL": "true",
            "EVAL_LIMIT": "",
        }

        with patch.dict("os.environ", env, clear=True):
            consensus = MultiJudgeConsensus()

        self.assertEqual(consensus.judge_mode, "provider_gemini_dual_model")
        self.assertEqual([judge.name for judge in consensus.judges], ["GeminiJudgeA", "GeminiJudgeB"])

    async def test_eval_limit_blocks_judge_b_even_in_final_mode(self):
        env = {
            "GEMINI_API_KEY": "test-gemini-key",
            "GEMINI_JUDGE_A_MODEL": "gemini-a-test",
            "GEMINI_JUDGE_B_MODEL": "gemini-b-test",
            "FINAL_PROVIDER_RUN": "true",
            "ALLOW_JUDGE_B_FINAL": "true",
            "EVAL_LIMIT": "5",
        }

        with patch.dict("os.environ", env, clear=True):
            consensus = MultiJudgeConsensus()

        self.assertEqual(consensus.judge_mode, "provider_gemini_single_model")
        self.assertEqual([judge.name for judge in consensus.judges], ["GeminiJudgeA", "RelevanceCitationHeuristicJudge"])

    async def test_budget_planner_counts_batched_judge_requests(self):
        plan = plan_provider_requests(total_cases=50, batch_size=5, uncached_a=50, uncached_b=50)

        self.assertEqual(plan["judge_a_planned_requests"], 10)
        self.assertEqual(plan["judge_b_planned_requests"], 10)
        self.assertEqual(plan["total_planned_provider_requests"], 20)

    async def test_budget_planner_blocks_judge_b_over_planned_limit(self):
        with patch.dict(
            "os.environ",
            {
                "JUDGE_B_DAILY_REQUEST_BUDGET": "20",
                "JUDGE_B_RESERVED_RETRY_BUDGET": "4",
                "JUDGE_B_MAX_PLANNED_REQUESTS": "9",
            },
            clear=True,
        ):
            plan = plan_provider_requests(total_cases=50, batch_size=5, uncached_a=50, uncached_b=50)

        self.assertTrue(plan["budget_blocked"])
        self.assertEqual(plan["judge_b_planned_requests"], 10)

    async def test_judge_b_runtime_counter_never_exceeds_daily_budget(self):
        env = {
            "GEMINI_API_KEY": "test-gemini-key",
            "GEMINI_JUDGE_A_MODEL": "gemini-a-test",
            "GEMINI_JUDGE_B_MODEL": "gemini-b-test",
            "FINAL_PROVIDER_RUN": "true",
            "ALLOW_JUDGE_B_FINAL": "true",
            "JUDGE_B_DAILY_REQUEST_BUDGET": "0",
        }

        with patch.dict("os.environ", env, clear=True):
            consensus = MultiJudgeConsensus()

        self.assertFalse(consensus.can_call_judge_b())
        self.assertEqual(consensus.judge_b_actual_requests, 0)

    async def test_no_key_uses_offline_fallback_mode(self):
        with patch.dict("os.environ", {}, clear=True):
            consensus = MultiJudgeConsensus()

        self.assertEqual(consensus.judge_mode, "offline_fallback")
        self.assertEqual([judge.name for judge in consensus.judges], ["FaithfulnessHeuristicJudge", "RelevanceCitationHeuristicJudge"])

    async def test_offline_provider_override_blocks_configured_gemini_key(self):
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key", "JUDGE_PROVIDER": "offline"}, clear=True):
            consensus = MultiJudgeConsensus()

        self.assertEqual(consensus.judge_mode, "offline_fallback")
        self.assertEqual([judge.name for judge in consensus.judges], ["FaithfulnessHeuristicJudge", "RelevanceCitationHeuristicJudge"])

    async def test_provider_parse_error_returns_safe_score(self):
        parsed = parse_judge_json("The answer is probably fine, but not JSON.", "ProviderJudge")

        self.assertEqual(parsed.score, 1.0)
        self.assertEqual(parsed.faithfulness, 0.0)
        self.assertEqual(parsed.hallucination_risk, "high")
        self.assertIn("parse_error", parsed.extra)

    async def test_conflict_resolution_caps_missing_citation(self):
        consensus = MultiJudgeConsensus()
        case = {
            "case_id": "citation_001",
            "id": "citation_001",
            "question": "What citation is required?",
            "expected_answer": "Use citation [DOC-001].",
            "ground_truth_ids": ["DOC-001"],
            "expected_retrieval_ids": ["DOC-001"],
            "case_type": "citation_sensitive",
            "tags": ["citation"],
            "metadata": {"type": "fact-check", "difficulty": "medium"},
        }

        result = await consensus.evaluate(case, "Correct content without the citation marker.")

        self.assertLessEqual(result["final_score"], 3.0)
        self.assertEqual(result["final_verdict"], "fail")
        self.assertTrue(result["required_citation_missing"])
        self.assertIn("resolution", result)


class ReleaseGateTests(unittest.TestCase):
    def test_release_gate_blocks_low_judge_agreement(self):
        base_summary = {
            "metadata": {"total": 50},
            "judge": {"judge_mode": "provider_gemini_dual_model"},
            "metrics": {
                "hit_rate": 1.0,
                "mrr": 1.0,
                "avg_score": 4.5,
                "estimated_total_cost_usd": 0.01,
                "p95_latency_ms": 1000,
                "agreement_rate": 0.4,
            }
        }

        regression = build_regression(base_summary, base_summary)

        self.assertNotEqual(regression["release_decision"], "APPROVE")
        self.assertIn("judge agreement below threshold", regression["reasons"])

    def test_release_gate_blocks_offline_fallback_provider_mode(self):
        summary = {
            "metadata": {"total": 50},
            "judge": {"judge_mode": "offline_fallback"},
            "metrics": {
                "hit_rate": 1.0,
                "mrr": 1.0,
                "avg_score": 4.5,
                "estimated_total_cost_usd": 0.01,
                "p95_latency_ms": 1000,
                "agreement_rate": 1.0,
            },
        }

        regression = build_regression(summary, summary)

        self.assertEqual(regression["release_decision"], "BLOCK_RELEASE")
        self.assertIn("offline fallback mode is not valid for final provider scoring", regression["reasons"])

    def test_release_gate_requires_judge_b_to_have_run_for_approval(self):
        summary = {
            "metadata": {"total": 50},
            "judge": {
                "judge_mode": "provider_gemini_single_model",
                "judge_a_actual_requests": 10,
                "judge_b_actual_requests": 0,
            },
            "metrics": {
                "hit_rate": 1.0,
                "mrr": 1.0,
                "avg_score": 4.5,
                "estimated_total_cost_usd": 0.01,
                "p95_latency_ms": 1000,
                "agreement_rate": 1.0,
            },
        }

        regression = build_regression(summary, summary)

        self.assertEqual(regression["release_decision"], "NEEDS_REVIEW")
        self.assertIn("single Gemini model mode is lower confidence", regression["reasons"])


if __name__ == "__main__":
    unittest.main()
