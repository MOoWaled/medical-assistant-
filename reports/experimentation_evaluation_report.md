# Experimentation and Evaluation Report

## 1. Purpose and scope

This report evaluates the local **Grounded FAISS Baseline** used by the Medical Condition Assistant. The baseline represents symptom text with TF-IDF and uses FAISS cosine similarity to retrieve condition records from the local NHS Inform knowledge base.

This is a reproducible functional-regression evaluation. It verifies that representative, manually curated symptom prompts retrieve their expected known condition. It is **not** a clinical validation study, a diagnosis-accuracy claim, or an estimate of performance on unseen patients.

## 2. Evaluated system

| Item | Configuration |
| --- | --- |
| Dataset | 413 local condition records: `usable_dataset.csv` plus reviewed NHS supplements |
| Query field | Symptoms only |
| Retriever | TF-IDF (word 1-2 grams, L2 normalisation) + FAISS `IndexFlatIP` |
| Returned content | Stored condition, symptoms, causes, warnings and recommendations only |
| Test suite | 10 fixed cases in `evaluation/baseline_regression_cases.csv` |
| Metrics | Top-1 accuracy, Top-3 recall, MRR@3 |

Qwen and Mistral were not scored here because they depend on user-operated or remote endpoints whose availability and model versions are not fixed. Both receive the same local retrieval candidates and require a separate endpoint-stable selection evaluation.

## 3. Reproducibility

From the project root, rebuild the index and run the evaluation:

```powershell
.\.venv\Scripts\python.exe model_API\train_baseline.py --force
.\.venv\Scripts\python.exe evaluation\evaluate_baseline.py --top-k 3
```

The legacy command remains supported and runs the same canonical suite:

```powershell
.\.venv\Scripts\python.exe evaluation\evaluate_retrieval.py --top-k 3
```

The evaluator prints the ranked candidates for every case. A failure should be investigated before changing a test label or claiming an improvement.

## 4. Results

Evaluation date: 2026-08-17

| Metric | Result |
| --- | ---: |
| Cases | 10 |
| Top-1 accuracy | 80.0% (8/10) |
| Top-3 recall | 100.0% (10/10) |
| MRR@3 | 0.90 |

| Case | Expected condition | Rank | Top-1 result | Status |
| --- | --- | ---: | --- | --- |
| Athlete's foot | Athlete's foot | 1 | Athlete's foot | Pass |
| Asthma | Asthma | 1 | Asthma | Pass |
| Appendicitis | Appendicitis | 1 | Appendicitis | Pass |
| Chickenpox | Chickenpox | 2 | Discoid eczema | Top-3 only |
| Food poisoning | Food poisoning | 1 | Food poisoning | Pass |
| Generalised anxiety disorder | Generalised anxiety disorder (GAD) | 1 | Generalised anxiety disorder (GAD) | Pass |
| Heart attack | Heart attack | 1 | Heart attack | Pass |
| Migraine | Migraine | 2 | Skin light sensitivity (photosensitivity) | Top-3 only |
| Type 2 diabetes | Type 2 diabetes | 1 | Type 2 diabetes | Pass |
| ADHD regression | ADHD in adults | 1 | ADHD in adults | Pass |

## 5. Key findings

1. The prior ADHD error was a data-coverage problem: `ADHD in adults` was absent from the local condition collection. A reviewed NHS Inform supplemental record was added, the index was rebuilt, and the exact regression prompt now ranks ADHD first.
2. The expected condition appears in the first three candidates for every case in this small suite.
3. Two Top-1 misses remain. The Chickenpox prompt is confused with another rash condition, while the Migraine prompt is over-influenced by light sensitivity. These are known retrieval limitations and are intentionally retained in the report.

## 6. Limitations and safety

- The suite is small, manually curated and contains known conditions. It cannot measure clinical usefulness, population performance, fairness or safety for real patients.
- The source dataset contains roughly one document per condition. A random classifier train/test split would be misleading; this is why the current baseline is retrieval-based.
- The system currently always returns nearest candidates. It has no calibrated out-of-distribution or low-confidence abstention rule, so unrelated prompts still need user caution and clinician review.
- Age and gender are optional request context only. They must not change retrieved medical facts until the knowledge base contains clinically reviewed demographic eligibility data.
- The displayed content is educational information, not a diagnosis or treatment plan.

## 7. Next evaluation actions

1. Ask a clinician or domain reviewer to create an independent, de-identified evaluation set with expected conditions and emergency flags.
2. Add ambiguous, unrelated and emergency prompts; measure abstention behaviour and emergency-recall performance.
3. Improve Top-1 ranking for rash and headache cases with a clinically reviewed reranker or weighted symptom phrases, then rerun this unchanged suite.
4. When Qwen and Mistral endpoints are stable, evaluate their exact-candidate selection accuracy separately from retrieval quality.
5. Store each run's dataset hash, model configuration, metrics and timestamp in MongoDB/GridFS to create a reproducible model registry.
