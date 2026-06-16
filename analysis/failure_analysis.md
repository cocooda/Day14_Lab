# Failure Analysis Report

## 1. Benchmark Overview
- **Total cases:** 62
- **Pass/Fail:** 50/12
- **Pass rate:** 0.806
- **Average faithfulness:** 0.832
- **Average relevancy:** 0.832
- **Average LLM judge score:** 4.290 / 5.0
- **Hit Rate@k:** 0.903
- **MRR@k:** 0.763
- **Agreement Rate:** 1.000
- **Cohen's Kappa:** 1.000
- **Conflict Count/Rate:** 0 / 0.000
- **Judge mode:** mixed_provider_fallback
- **Judge models:** A=gemini-2.5-flash-lite; B=gemini-2.5-flash-lite
- **Retriever:** tfidf_vector_retriever
- **Release decision:** NEEDS_REVIEW
- **Latency:** avg 0.1 ms; p95 0.3 ms
- **Cost:** total $0.052680; per case $0.000850

## 2. Failure Clustering
| Failure cluster | Count | Expected system cause |
|---|---:|---|
| citation_missing | 6 | Generation omitted a required source marker. |
| hallucination | 10 | Answer included unsupported content. |
| incomplete | 2 | Answer was too thin relative to expected answer. |
| position_bias | 24 | Correct source was retrieved but not ranked first. |
| retrieval_miss | 6 | Retrieval did not surface any ground truth id. |

## Quota Strategy
- 2.5 Flash Judge B is enabled only when `FINAL_PROVIDER_RUN=true`, `ALLOW_JUDGE_B_FINAL=true`, and `EVAL_LIMIT` is unset.
- Judge batch size: 5
- Gemini limiter: 6 RPM shared by Judge A and Judge B.
- Judge B hard cap: 20 requests/day with 4 reserved for retries.
- Cache enabled: True at `.cache/judge_results.json`.

## 3. 5 Whys For The 3 Worst Cases
### Case #1: case_007 (retrieval_miss)
- Question: Người tổ chức cho người khác sử dụng ma túy có thể bị tử hình không?
- Expected answer summary: Không. Theo Điều 255 Bộ luật Hình sự 2015, tội tổ chức sử dụng trái phép chất ma túy có hình phạt cao nhất là tù chung thân, không có hình phạt tử hình. Cite [bo-luat-hinh-su-2015-
- Agent answer summary: I cannot answer from the retrieved context; mark this as retrieval_miss.
- Retrieved ids: ['luat-phong-chong-ma-tuy-2021', 'nghi-dinh-57-2022-danh-muc-chat-ma-tuy', 'nghi-dinh-105-2021-huong-dan-luat-phong-chong-ma-tuy']
- Hit Rate/MRR: 0.000 / 0.000
- Final score/agreement: 1.00 / 1.000
1. Symptom: Final score 1.00; verdict fail; primary cluster retrieval_miss.
2. Why 1: The answer failed the strongest signal: retrieval_miss.
3. Why 2: Retrieval returned ['luat-phong-chong-ma-tuy-2021', 'nghi-dinh-57-2022-danh-muc-chat-ma-tuy', 'nghi-dinh-105-2021-huong-dan-luat-phong-chong-ma-tuy'] for expected ids ['bo-luat-hinh-su-2015-chuong-xx-toi-pham-ma-tuy'].
4. Why 3: TF-IDF ranking depends on token overlap, so ambiguous, conflicting, or adversarial wording can move the right chunk down.
5. Why 4: The agent prompt uses retrieved context directly and has no reranker, semantic expansion, or citation repair pass.
6. Why 5: The pipeline is optimized for deterministic lab reproducibility rather than production retrieval robustness.
7. Root cause: retrieval.

### Case #2: case_028 (retrieval_miss)
- Question: Người bị bắt vì vừa tàng trữ heroin vừa lôi kéo người khác sử dụng ma túy thì phạm những tội gì theo Bộ luật Hình sự 2015?
- Expected answer summary: Người này phạm 2 tội: (1) Tàng trữ trái phép chất ma túy (Điều 249) và (2) Cưỡng bức, lôi kéo người khác sử dụng trái phép chất ma túy (Điều 257) theo Bộ luật Hình sự 2015. Hình ph
- Agent answer summary: I cannot answer from the retrieved context; mark this as retrieval_miss.
- Retrieved ids: ['luat-phong-chong-ma-tuy-2021', 'nghi-dinh-105-2021-huong-dan-luat-phong-chong-ma-tuy', 'nghi-dinh-57-2022-danh-muc-chat-ma-tuy']
- Hit Rate/MRR: 0.000 / 0.000
- Final score/agreement: 1.00 / 1.000
1. Symptom: Final score 1.00; verdict fail; primary cluster retrieval_miss.
2. Why 1: The answer failed the strongest signal: retrieval_miss.
3. Why 2: Retrieval returned ['luat-phong-chong-ma-tuy-2021', 'nghi-dinh-105-2021-huong-dan-luat-phong-chong-ma-tuy', 'nghi-dinh-57-2022-danh-muc-chat-ma-tuy'] for expected ids ['bo-luat-hinh-su-2015-chuong-xx-toi-pham-ma-tuy'].
4. Why 3: TF-IDF ranking depends on token overlap, so ambiguous, conflicting, or adversarial wording can move the right chunk down.
5. Why 4: The agent prompt uses retrieved context directly and has no reranker, semantic expansion, or citation repair pass.
6. Why 5: The pipeline is optimized for deterministic lab reproducibility rather than production retrieval robustness.
7. Root cause: retrieval.

### Case #3: case_032 (retrieval_miss)
- Question: Người trồng 50 cây thuốc phiện để lấy nhựa bán cho người khác thì có thể phạm những tội gì theo Bộ luật Hình sự 2015?
- Expected answer summary: Người này có thể phạm: (1) Tội trồng cây thuốc phiện (Điều 247 khoản 2 – quy mô lớn) và (2) Tội sản xuất và/hoặc mua bán trái phép chất ma túy (Điều 248, Điều 251) nếu đã chế biến 
- Agent answer summary: I cannot answer from the retrieved context; mark this as retrieval_miss.
- Retrieved ids: ['luat-phong-chong-ma-tuy-2021', 'nghi-dinh-105-2021-huong-dan-luat-phong-chong-ma-tuy', 'nghi-dinh-57-2022-danh-muc-chat-ma-tuy']
- Hit Rate/MRR: 0.000 / 0.000
- Final score/agreement: 1.00 / 1.000
1. Symptom: Final score 1.00; verdict fail; primary cluster retrieval_miss.
2. Why 1: The answer failed the strongest signal: retrieval_miss.
3. Why 2: Retrieval returned ['luat-phong-chong-ma-tuy-2021', 'nghi-dinh-105-2021-huong-dan-luat-phong-chong-ma-tuy', 'nghi-dinh-57-2022-danh-muc-chat-ma-tuy'] for expected ids ['bo-luat-hinh-su-2015-chuong-xx-toi-pham-ma-tuy'].
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
