# Hard Cases Guide

Owner: Linh

This guide documents the Linh-owned golden dataset and synthetic data generation work for Lab Day 14. The engine may consume this data, but dataset design, SDG, expected retrieval IDs, and hard-case coverage belong to Linh.

## Case Schema

Each JSONL row in `data/golden_set.jsonl` uses this schema:

```json
{
  "id": "case_001",
  "question": "Question asked to the agent",
  "expected_answer": "Reference answer with citations when needed",
  "context": "Grounding text used to build the local retrieval corpus",
  "expected_retrieval_ids": ["doc_001"],
  "metadata": {
    "difficulty": "easy",
    "type": "fact-check"
  }
}
```

Normal cases must include at least one `expected_retrieval_ids` value. Out-of-context cases may use an empty list because the correct behavior is to abstain when no relevant source exists.

## Case Types

- `fact-check`: direct factual checks with a known supporting document.
- `adversarial`: prompt-injection or red-team questions that should be answered only from evidence.
- `out-of-context`: questions with no relevant source in the corpus.
- `ambiguous`: questions that require clarification or explicit metric context.
- `conflicting`: cases with two sources or claims that need careful reconciliation.
- `multi-turn`: multi-hop or context-carryover style questions requiring more than one source.

## How Retrieval IDs Are Used

The evaluation engine retrieves `retrieved_ids` for each question and compares them to `expected_retrieval_ids`.

- Hit Rate@k is `1.0` when at least one expected id appears in the top k retrieved ids.
- MRR@k is the reciprocal rank of the first expected id in the top k retrieved ids.
- Recall@k measures how many expected ids were retrieved.
- Out-of-context cases are excluded from MRR averaging to avoid a division-by-zero style distortion.

The IDs are not decorative labels. They are the ground-truth bridge between Linh's dataset and Duc's retrieval metrics/reporting pipeline.
