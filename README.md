# Validation-tuned text-view fusion vs fixed-rule retrieval (20 Newsgroups stand-in)

Q2E (Query-to-Event Decomposition for Zero-Shot Multilingual Text-to-Video Retrieval, 2025) improves zero-shot text-to-video retrieval by expanding queries into events and fusing visual and speech evidence with a fixed entropy-based rule. This repository asks a narrower question that can be checked cheaply: when two text views are fused, does tuning the fusion weights on a validation split beat a single fixed score rule? It uses a public text corpus as a stand-in, so it does not test the paper's video pipeline.

**Question.** Can a validation-tuned fusion of two text views rank relevant documents better (NDCG@10) than a single fixed-rule score, as a stand-in for the paper's fixed entropy-based fusion of visual and speech evidence?

**Dataset.** 20 Newsgroups, loaded with scikit-learn's `fetch_20newsgroups`. Each post is treated as a "video" and each newsgroup has a short event-style query. This is not the paper's own dataset (`same_as_paper: false`).

**Baseline vs improved.**
- Baseline: single-view word TF-IDF cosine retrieval with a fixed score rule, with no tuning.
- Improved: a weighted fusion of word TF-IDF, character n-gram TF-IDF and LSA cosine scores. The weights are chosen by grid search on a validation split carved from the training data. The test set is used once per model.

Both models share the same data, splits, vectorisers, query set and evaluation code. The main metric is NDCG@10 on the test split. MRR and Recall@10 are reported as secondary metrics.

## How to run

```bash
pip install -r requirements.txt
python analysis.py
```

Results: written to `results.json` (charts go to `figures/`).

## Results (latest run)

On the 20 Newsgroups stand-in (newsgroup descriptions as queries, 20 queries, same-newsgroup relevance), validation-tuned fusion of word TF-IDF, char n-gram TF-IDF and LSA reached NDCG@10 of 75.7%, versus 61.7% for single-view word TF-IDF cosine retrieval on the same test set.

| Metric | Value |
|---|---|
| baseline_score | 0.6165 |
| improved_score | 0.7566 |
| improvement | 0.1401 |
| baseline_score_std | 0 |
| improved_score_std | 0.006525 |
| baseline_mrr | 0.7642 |
| improved_mrr | 0.8653 |
| baseline_recall_at_10 | 0.02574 |
| improved_recall_at_10 | 0.03126 |
| best_weight_word | 0 |
| best_weight_char | 0.75 |
| best_weight_lsa | 1 |

Run on 2026-10-21. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Stand-in for the paper's zero-shot text-to-video setting, not a replication of Q2E; no visual or speech input. Queries are hand-written newsgroup descriptions and relevance is binary (same newsgroup), so only 20 queries are evaluated. Validation is a stratified 20% holdout of the non-test data. Only LSA is randomized (seeds 42-44); the baseline is deterministic, so its std is 0. The full dataset (~18.8k rows) is used, with no subsampling. The 60 s timeout is applied via the socket default, since scikit-learn does not expose a timeout parameter.

## Reference

This project is motivated by the following paper:

> SR Dipta, F Ferraro (2025). *Q2E: Query-to-Event Decomposition for Zero-Shot Multilingual Text-to-Video Retrieval*. Proceedings of the 14th International Joint. https://aclanthology.org/2025.ijcnlp-long.121/

Paper author: Prof. Frank Ferraro, University of Maryland, Baltimore County (ferraro@umbc.edu)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
