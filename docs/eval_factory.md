# Evaluation Factory

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
