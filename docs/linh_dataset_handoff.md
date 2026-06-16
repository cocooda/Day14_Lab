# Linh Dataset Handoff

This guide describes how Linh should add the official dataset and corpus for the Lab Day 14 Evaluation Factory.

## 1. Corpus Placement

Place the legal markdown corpus files in:

```text
data/corpus/
```

Each markdown file is treated as one retrievable document. The document ID is the filename without `.md`.

Required document IDs:

- `bo-luat-hinh-su-2015-chuong-xx-toi-pham-ma-tuy`
- `luat-phong-chong-ma-tuy-2021`
- `nghi-dinh-57-2022-danh-muc-chat-ma-tuy`
- `nghi-dinh-105-2021-huong-dan-luat-phong-chong-ma-tuy`

## 2. Golden Seed Placement

If using a seed JSON file, place it at:

```text
data/golden_seed.json
```

For the temporary validation run, Duc used `data/golden_dataset.json` as the seed source. Linh can rename or adapt the official seed path when finalizing the dataset branch.

## 3. Generate Golden JSONL

Run:

```bash
uv run python data/synthetic_gen.py
```

If `uv` is unavailable:

```bash
python data/synthetic_gen.py
```

The generated output is:

```text
data/golden_set.jsonl
```

Do not commit `data/golden_set.jsonl` if the README says it should be generated before benchmarking.

## 4. Required JSONL Schema

Each line in `data/golden_set.jsonl` must be one JSON object:

```json
{
  "id": "case_001",
  "question": "...",
  "expected_answer": "...",
  "context": "...",
  "expected_retrieval_ids": ["doc_id"],
  "metadata": {
    "difficulty": "easy|medium|hard",
    "type": "fact-check|adversarial|out-of-context|ambiguous|conflicting|multi-turn"
  }
}
```

## 5. Minimum Requirements

- Include at least 50 cases.
- Prefer 60 cases if the extra 10 cases are high quality and not duplicates.
- Normal cases must have `expected_retrieval_ids`.
- Out-of-context cases may have empty `expected_retrieval_ids` only when the correct behavior is insufficient-context/refusal.
- `expected_retrieval_ids` must match corpus filename stems exactly.
- Include hard/red-team cases, including adversarial, ambiguous, conflicting, out-of-context, and multi-turn cases.

## 6. Validation Commands

Run:

```bash
uv run python data/synthetic_gen.py
uv run python main.py
uv run python check_lab.py
```

If `uv` is unavailable:

```bash
python data/synthetic_gen.py
python main.py
python check_lab.py
```

## 7. Common Mistakes

- Document ID mismatch between `expected_retrieval_ids` and corpus filename stems.
- Committing generated `data/golden_set.jsonl` when the README says it should be generated.
- Normal cases missing `expected_retrieval_ids`.
- Weak duplicate cases added only to inflate count.
- Out-of-context cases incorrectly expecting a retrieval hit.
- Seed cases using old synthetic `doc_###` IDs instead of the legal corpus document IDs.

