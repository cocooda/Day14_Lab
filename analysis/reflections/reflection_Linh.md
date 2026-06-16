# Individual Reflection - Linh

## Dataset / SDG Ownership
I own the golden dataset, synthetic data generation, and hard-case guide for the two-person Evaluation Factory project. The dataset uses stable case ids, expected answers, context, and `expected_retrieval_ids` so the engine can compute Hit Rate and MRR from real retrieval outputs.

## Hard Case Design
The case mix includes easy and medium fact checks, hard adversarial prompts, out-of-context questions, ambiguous cases, conflicting evidence, and multi-turn or multi-hop style questions. The red-team cases are meant to test whether the agent stays grounded in retrieved evidence instead of following unsafe or unsupported instructions.

## Handoff Note
This reflection is intentionally brief because the final dataset wording and any additional Linh-owned edits should be reviewed by Linh before submission.
