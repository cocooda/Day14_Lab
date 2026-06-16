# Individual Reflection - Đoàn Thị Thu Linh (2A202600964)

## My Contribution

My primary ownership in this two-person Evaluation Factory project was the **dataset pipeline** and **domain knowledge layer**:

1. **Corpus documents** — I researched and authored the four Vietnamese legal source documents placed in `data/corpus/`:
   - `bo-luat-hinh-su-2015-chuong-xx-toi-pham-ma-tuy.md` (BLHS 2015, Chapter XX)
   - `luat-phong-chong-ma-tuy-2021.md` (Drug Prevention Law 2021, effective 01/01/2022)
   - `nghi-dinh-57-2022-danh-muc-chat-ma-tuy.md` (Decree 57/2022 drug classification)
   - `nghi-dinh-105-2021-huong-dan-luat-phong-chong-ma-tuy.md` (Decree 105/2021 guidance)

2. **Golden dataset design** — I designed the 62-case `data/golden_dataset.json` covering all required case types: fact-check (27), multi-turn (10), adversarial/red-team (10), ambiguous (5), conflicting evidence (5), and out-of-context (5).

3. **Hard-case rationale** — For adversarial cases (e.g., prompt injection, out-of-scope requests), the expected answer is a refusal grounded in Vietnamese law, not a generic "I don't know." This tests whether the agent stays faithful to retrieved evidence even under adversarial pressure.

4. **`expected_retrieval_ids` accuracy** — Each case carries the exact corpus document IDs the retriever must surface to answer correctly. Inaccurate ground-truth IDs would make Hit Rate and MRR meaningless, so I cross-checked every case against the actual document content.

## What I Learned

- **Retrieval ground truth is harder than it looks.** Writing `expected_retrieval_ids` required reading all four legal documents closely. One mistake (e.g., citing the wrong decree for a penalty question) corrupts the retrieval metrics for that whole case type.

- **Adversarial cases expose a real gap.** The TF-IDF retriever has no semantic understanding, so prompt-injection questions that paraphrase or contradict the legal text frequently retrieve the wrong document. This is visible in the retrieval_miss cluster in `analysis/failure_analysis.md`.

- **Domain vocabulary matters for TF-IDF.** Including precise legal terms (`tù chung thân`, `Điều 255`, `cai nghiện bắt buộc`) in both the corpus and the questions significantly improved Hit Rate from early drafts.

## Honest Assessment

The biggest weakness in my contribution was the multi-turn cases: I designed them as independent question-answer pairs rather than true stateful conversations, because the current agent architecture does not maintain conversation history. A stronger dataset would include cases where the agent must reference a previous turn's answer — this would require agent-side changes beyond the dataset.

The retrieval miss rate on adversarial questions (cases 038–047) is a known limitation. I chose questions that are genuinely hard for keyword-based retrieval on purpose, so the failure cluster in the report reflects real system weakness rather than bad test design.
