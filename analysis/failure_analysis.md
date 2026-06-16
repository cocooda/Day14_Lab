# Failure Analysis Report

## 1. Benchmark Overview
- **Total cases:** 50
- **Pass/Fail:** 9/41
- **Pass rate:** 0.180
- **Average faithfulness:** 0.478
- **Average relevancy:** 0.693
- **Average LLM judge score:** 1.940 / 5.0
- **Hit Rate@k:** 0.940
- **MRR@k:** 0.757
- **Agreement Rate:** 0.220
- **Cohen's Kappa:** -0.007
- **Conflict Count/Rate:** 29 / 0.580
- **Judge mode:** provider_gemini_dual_model
- **Judge models:** A=gemini-3.1-flash-lite; B=gemini-2.5-flash
- **Retriever:** tfidf_vector_retriever
- **Release decision:** NEEDS_REVIEW
- **Latency:** avg 0.1 ms; p95 0.2 ms
- **Cost:** total $0.034335; per case $0.000687

## 2. Failure Clustering
| Failure cluster | Count | Expected system cause |
|---|---:|---|
| citation_missing | 3 | Generation omitted a required source marker. |
| hallucination | 34 | Answer included unsupported content. |
| incomplete | 7 | Answer was too thin relative to expected answer. |
| judge_disagreement | 29 | Judges differed by more than one score point. |
| position_bias | 20 | Correct source was retrieved but not ranked first. |
| retrieval_miss | 3 | Retrieval did not surface any ground truth id. |

## Quota Strategy
- 2.5 Flash Judge B is enabled only when `FINAL_PROVIDER_RUN=true`, `ALLOW_JUDGE_B_FINAL=true`, and `EVAL_LIMIT` is unset.
- Judge batch size: 5
- Gemini limiter: 6 RPM shared by Judge A and Judge B.
- Judge B hard cap: 20 requests/day with 4 reserved for retries.
- Cache enabled: True at `.cache/judge_results.json`.

## 3. 5 Whys For The 3 Worst Cases
### Case #1: case_046 (retrieval_miss)
- Question: Tôi trồng cây thuốc phiện sau vườn nhà để ngắm thì có sao không?
- Expected answer summary: Có thể bị phạt tù từ 06 tháng đến 03 năm hoặc nặng hơn tùy mức độ vi phạm.
- Agent answer summary: I cannot answer from the retrieved context; mark this as retrieval_miss.
- Retrieved ids: ['nghi-dinh-105-2021-huong-dan-luat-phong-chong-ma-tuy', 'luat-phong-chong-ma-tuy-2021', 'nghi-dinh-57-2022-danh-muc-chat-ma-tuy']
- Hit Rate/MRR: 0.000 / 0.000
- Final score/agreement: 1.00 / 0.000
1. Symptom: Final score 1.00; verdict fail; primary cluster retrieval_miss.
2. Why 1: The answer failed the strongest signal: retrieval_miss.
3. Why 2: Retrieval returned ['nghi-dinh-105-2021-huong-dan-luat-phong-chong-ma-tuy', 'luat-phong-chong-ma-tuy-2021', 'nghi-dinh-57-2022-danh-muc-chat-ma-tuy'] for expected ids ['bo-luat-hinh-su-2015-chuong-xx-toi-pham-ma-tuy'].
4. Why 3: TF-IDF ranking depends on token overlap, so ambiguous, conflicting, or adversarial wording can move the right chunk down.
5. Why 4: The agent prompt uses retrieved context directly and has no reranker, semantic expansion, or citation repair pass.
6. Why 5: The pipeline is optimized for deterministic lab reproducibility rather than production retrieval robustness.
7. Root cause: retrieval.

### Case #2: case_007 (retrieval_miss)
- Question: Người tổ chức cho người khác sử dụng ma túy có thể bị tử hình không?
- Expected answer summary: Không, hình phạt cao nhất là tù chung thân.
- Agent answer summary: I cannot answer from the retrieved context; mark this as retrieval_miss.
- Retrieved ids: ['nghi-dinh-105-2021-huong-dan-luat-phong-chong-ma-tuy', 'luat-phong-chong-ma-tuy-2021', 'nghi-dinh-57-2022-danh-muc-chat-ma-tuy']
- Hit Rate/MRR: 0.000 / 0.000
- Final score/agreement: 1.00 / 0.000
1. Symptom: Final score 1.00; verdict fail; primary cluster retrieval_miss.
2. Why 1: The answer failed the strongest signal: retrieval_miss.
3. Why 2: Retrieval returned ['nghi-dinh-105-2021-huong-dan-luat-phong-chong-ma-tuy', 'luat-phong-chong-ma-tuy-2021', 'nghi-dinh-57-2022-danh-muc-chat-ma-tuy'] for expected ids ['bo-luat-hinh-su-2015-chuong-xx-toi-pham-ma-tuy'].
4. Why 3: TF-IDF ranking depends on token overlap, so ambiguous, conflicting, or adversarial wording can move the right chunk down.
5. Why 4: The agent prompt uses retrieved context directly and has no reranker, semantic expansion, or citation repair pass.
6. Why 5: The pipeline is optimized for deterministic lab reproducibility rather than production retrieval robustness.
7. Root cause: retrieval.

### Case #3: case_037 (retrieval_miss)
- Question: Morphine bị cấm sử dụng hoàn toàn trong mọi trường hợp đúng không?
- Expected answer summary: Sai, Morphine thuộc Danh mục II, được dùng hạn chế trong y tế.
- Agent answer summary: I cannot answer from the retrieved context; mark this as retrieval_miss.
- Retrieved ids: ['nghi-dinh-105-2021-huong-dan-luat-phong-chong-ma-tuy', 'luat-phong-chong-ma-tuy-2021', 'bo-luat-hinh-su-2015-chuong-xx-toi-pham-ma-tuy']
- Hit Rate/MRR: 0.000 / 0.000
- Final score/agreement: 1.00 / 1.000
1. Symptom: Final score 1.00; verdict fail; primary cluster retrieval_miss.
2. Why 1: The answer failed the strongest signal: retrieval_miss.
3. Why 2: Retrieval returned ['nghi-dinh-105-2021-huong-dan-luat-phong-chong-ma-tuy', 'luat-phong-chong-ma-tuy-2021', 'bo-luat-hinh-su-2015-chuong-xx-toi-pham-ma-tuy'] for expected ids ['nghi-dinh-57-2022-danh-muc-chat-ma-tuy'].
4. Why 3: TF-IDF ranking depends on token overlap, so ambiguous, conflicting, or adversarial wording can move the right chunk down.
5. Why 4: The agent prompt uses retrieved context directly and has no reranker, semantic expansion, or citation repair pass.
6. Why 5: The pipeline is optimized for deterministic lab reproducibility rather than production retrieval robustness.
7. Root cause: retrieval.


## 4. Action Plan
- Add semantic or hybrid retrieval and reranking for adversarial and multi-hop questions.
- Expand chunks around cited source ids so citation-sensitive answers have all required evidence.
- Route judge conflicts to a stricter citation-first adjudicator before release decisions.
- Cache judge outputs and escalate only low-confidence cases to expensive model judges.
- Add position-bias checks by shuffling retrieved contexts and comparing score stability.
