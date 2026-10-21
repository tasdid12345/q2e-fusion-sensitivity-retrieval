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

## Reference

This project is motivated by the following paper:

> SR Dipta, F Ferraro (2025). *Q2E: Query-to-Event Decomposition for Zero-Shot Multilingual Text-to-Video Retrieval*. Proceedings of the 14th International Joint. https://aclanthology.org/2025.ijcnlp-long.121/

Paper author: Prof. Frank Ferraro, University of Maryland, Baltimore County (ferraro@umbc.edu)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
