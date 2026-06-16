# Individual Reflection - Duc Nguyen

## Engineering Contribution
I owned the evaluation engine side of the two-person project: retrieval metrics integration, the Gemini multi-judge provider fix, async runner/report validation, regression release gate, failure analysis, check_lab validation, and documentation. Linh owns the dataset, SDG, and hard-case design; I only touched that path to keep the end-to-end benchmark runnable with the agreed schema.

## Technical Explanation
MRR measures retrieval ranking quality by taking the reciprocal rank of the first relevant document, then averaging across normal cases. A system can have high Hit Rate but lower MRR if the right source appears late, which matters because generation often overuses early context. Cohen's Kappa measures judge agreement after adjusting for chance agreement; it is stronger than raw agreement rate when judges have skewed score distributions. Position bias appears when a judge or generator prefers earlier options or retrieved chunks independent of content quality.

## Cost vs Quality Trade-off
The best production design is not to call expensive judges for every case. In this lab, Gemini dual-model mode is the preferred rubric-safe mode. Single-model mode is lower-confidence because one Gemini result is paired with a heuristic fallback, and offline fallback is only for local reproducibility. Caching judge results and limiting output tokens reduce cost without changing benchmark semantics.

## Problems And Strategy
The main issue was that the previous provider setup could claim provider scoring while using fallback-like behavior and DeepSeek configuration. I replaced it with the official `google-genai` client pattern, explicit Gemini A/B judges, parse-safe JSON scoring, and clear provider modes. Another challenge was connecting retrieval quality to answer quality without a vector database, so the TF-IDF retriever reports real Hit Rate and MRR from retrieved ids instead of simulated metrics.
