"""Stand-in experiment: validation-tuned fusion of text views vs a fixed single-view TF-IDF rule.

Documents = 20 Newsgroups posts (stand-ins for videos); queries = one event-style
description per newsgroup; a document is relevant to a query if it belongs to that newsgroup.
"""
import itertools, json, socket, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

SEED, SEEDS, K = 42, [42, 43, 44], 10   # seeds for the randomized LSA view; K for NDCG@K
OUT = "results.json"

# One short event-style query per newsgroup (hand-written, not taken from the paper).
QUERIES = {
    "alt.atheism": "debate about atheism and whether God exists",
    "comp.graphics": "questions about computer graphics software and image formats",
    "comp.os.ms-windows.misc": "help with Microsoft Windows operating system problems",
    "comp.sys.ibm.pc.hardware": "troubleshooting IBM PC hardware and disk controllers",
    "comp.sys.mac.hardware": "discussion of Macintosh computer hardware and upgrades",
    "comp.windows.x": "programming with the X Window System and window managers",
    "misc.forsale": "advertisement of items for sale and prices",
    "rec.autos": "reviews and repair talk about cars",
    "rec.motorcycles": "riding and maintaining motorcycles",
    "rec.sport.baseball": "baseball games, players and league standings",
    "rec.sport.hockey": "hockey games, teams and playoff results",
    "sci.crypt": "encryption algorithms and cryptography policy",
    "sci.electronics": "circuit design and electronics components",
    "sci.med": "medical treatments, diseases and health advice",
    "sci.space": "space missions, NASA launches and orbit",
    "soc.religion.christian": "Christian church doctrine and scripture",
    "talk.politics.guns": "gun control legislation and firearm rights",
    "talk.politics.mideast": "Middle East conflict and Israeli politics",
    "talk.politics.misc": "general political debate and government policy",
    "talk.religion.misc": "comparative religion and philosophical beliefs",
}
DISC = 1 / np.log2(np.arange(2, K + 2))  # DCG discounts for ranks 1..K


def load_data():
    """Download via scikit-learn with a 60 s socket timeout and exponential-backoff retries."""
    socket.setdefaulttimeout(60)  # urllib (used by sklearn) honours this timeout
    for attempt in range(5):
        try:
            return fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"),
                                      shuffle=True, random_state=SEED)
        except Exception as err:  # HTTP 429/5xx, URLError, socket timeout
            wait = 2 ** attempt
            print(f"[download] attempt {attempt + 1} failed: {err}; retrying in {wait}s")
            time.sleep(wait)
    sys.exit("[download] could not fetch 20 Newsgroups")


def cos(Q, D):
    """Cosine similarity matrix (rows of Q and D are L2-normalised), dense output."""
    S = Q @ D.T
    return S.toarray() if hasattr(S, "toarray") else S


def metrics(S, rel):
    """Per-query NDCG@K, reciprocal rank and Recall@K for score matrix S (queries x docs)."""
    order = np.argsort(-S, axis=1, kind="stable")
    r = np.take_along_axis(rel, order, axis=1)           # relevance in ranked order
    n_rel = rel.sum(1)
    ndcg = (r[:, :K] * DISC).sum(1) / np.array([DISC[:int(n)].sum() for n in n_rel])
    mrr = 1.0 / (np.argmax(r > 0, axis=1) + 1)
    recall = r[:, :K].sum(1) / n_rel
    return np.stack([ndcg, mrr, recall])


def main():
    data = load_data()
    names = list(data.target_names)
    if set(names) != set(QUERIES):
        sys.exit("query descriptions do not match newsgroup names")
    texts, y = np.array(data.data, dtype=object), np.asarray(data.target)
    print(f"[data] {len(y)} documents, {len(names)} newsgroups")

    # Split ONCE: 25% test (stratified); 20% of the rest held out as validation for tuning.
    all_idx = np.arange(len(y))
    tr_val, te = train_test_split(all_idx, test_size=0.25, stratify=y, random_state=SEED)
    tr, va = train_test_split(tr_val, test_size=0.2, stratify=y[tr_val], random_state=SEED)
    print(f"[split] train={len(tr)} val={len(va)} test={len(te)}")

    # Vectorisers are fitted on training documents only.
    word = TfidfVectorizer(sublinear_tf=True, min_df=2, stop_words="english").fit(texts[tr])
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True,
                           min_df=3, max_features=200000).fit(texts[tr])
    Xw, Xc = word.transform(texts), char.transform(texts)
    qtext = [QUERIES[n] for n in names]
    Qw, Qc = word.transform(qtext), char.transform(qtext)
    print("[features] word and char TF-IDF views built")

    # Fusion weight grid for (word, char, LSA) cosine scores; all-zero excluded.
    W = [0.0, 0.25, 0.5, 0.75, 1.0]
    grid = [w for w in itertools.product(W, repeat=3) if sum(w) > 0]
    rel = {k: (y[idx][None, :] == np.arange(len(names))[:, None]).astype(float)
           for k, idx in (("va", va), ("te", te))}

    runs = {"baseline": [], "improved": []}
    chosen = []
    for s in SEEDS:
        # The only randomized component is the LSA (truncated SVD) view.
        svd = TruncatedSVD(n_components=200, random_state=s).fit(Xw[tr])
        Z, Zq = normalize(svd.transform(Xw)), normalize(svd.transform(Qw))
        comp = {k: np.stack([cos(Qw, Xw[idx]), cos(Qc, Xc[idx]), cos(Zq, Z[idx])])
                for k, idx in (("va", va), ("te", te))}
        # Grid search of fusion weights on the VALIDATION split only.
        best_w, best_v = None, -1.0
        for w in grid:
            v = metrics(np.tensordot(w, comp["va"], axes=1), rel["va"])[0].mean()
            if v > best_v:
                best_w, best_v = w, v
        # Test set used once per model: baseline = fixed word-only cosine rule.
        b = metrics(comp["te"][0], rel["te"])
        i = metrics(np.tensordot(best_w, comp["te"], axes=1), rel["te"])
        runs["baseline"].append(b.mean(1))
        runs["improved"].append(i.mean(1))
        chosen.append(best_w)
        print(f"[seed {s}] weights(word,char,lsa)={best_w} val NDCG@{K}={best_v:.4f} "
              f"test NDCG@{K}: baseline={b[0].mean():.4f} improved={i[0].mean():.4f}")
        if s == SEEDS[0]:  # per-query reporting for the first seed
            for n, qb, qi in zip(names, b[0], i[0]):
                print(f"    {n:28s} baseline={qb:.3f} improved={qi:.3f}")

    def agg(key):  # mean and std across seeds of [NDCG, MRR, Recall]
        arr = np.array(runs[key])
        return arr.mean(0), arr.std(0)

    (bm, bs), (im, ims) = agg("baseline"), agg("improved")
    out_metrics = {
        "baseline_score": float(bm[0]), "improved_score": float(im[0]),
        "improvement": float(im[0] - bm[0]),
        "baseline_score_std": float(bs[0]), "improved_score_std": float(ims[0]),
        "baseline_mrr": float(bm[1]), "improved_mrr": float(im[1]),
        "baseline_recall_at_10": float(bm[2]), "improved_recall_at_10": float(im[2]),
        "best_weight_word": float(chosen[0][0]), "best_weight_char": float(chosen[0][1]),
        "best_weight_lsa": float(chosen[0][2]),
    }
    if not np.all(np.isfinite(list(out_metrics.values()))):
        sys.exit("non-finite metrics; aborting")

    # Optional chart: baseline vs improved NDCG@10 (mean +/- std over seeds).
    import os
    os.makedirs("figures", exist_ok=True)
    plt.figure(figsize=(4, 4))
    plt.bar(["baseline", "improved"], [bm[0], im[0]], yerr=[bs[0], ims[0]], capsize=6,
            color=["grey", "steelblue"])
    plt.ylabel(f"Test NDCG@{K}")
    plt.title("Baseline vs validation-tuned fusion")
    plt.tight_layout()
    plt.savefig("figures/ndcg_comparison.png", dpi=120)
    plt.close()

    results = {
        "question": ("Can a validation-tuned fusion of two text views rank relevant documents "
                     "better (NDCG@10) than a single fixed-rule score, as a stand-in for the "
                     "paper's fixed entropy-based fusion of visual and speech evidence?"),
        "method": ("Stand-in retrieval on 20 Newsgroups: documents as videos, newsgroup "
                   "descriptions as queries; fusion weights of word TF-IDF, char n-gram TF-IDF "
                   "and LSA cosine scores grid-searched on a validation split."),
        "dataset": {
            "name": "20 Newsgroups (scikit-learn fetch_20newsgroups, subset=all, headers/footers/quotes removed)",
            "source_url": "https://scikit-learn.org/stable/datasets/real_world.html#the-20-newsgroups-text-dataset",
            "n_rows": int(len(y)), "n_train": int(len(tr)), "n_test": int(len(te)),
            "same_as_paper": False, "baseline_is_papers_method": False,
        },
        "metric_name": f"NDCG@{K}",
        "higher_is_better": True,
        "baseline": "Single-view word TF-IDF cosine retrieval with a fixed score rule (no tuning).",
        "improved": ("Weighted fusion of word TF-IDF, char_wb n-gram TF-IDF and LSA (200-d) cosine "
                     "scores; weights chosen by grid search on the validation split only."),
        "metrics": out_metrics,
        "notes": ("Stand-in for the paper's zero-shot text-to-video setting, not a replication of Q2E; "
                  "no visual or speech input. Queries are hand-written newsgroup descriptions and "
                  "relevance is binary (same newsgroup), so only 20 queries are evaluated. "
                  "Validation is a stratified 20% holdout of the non-test data. Only LSA is randomized "
                  "(seeds 42-44); the baseline is deterministic, so its std is 0. The full dataset "
                  "(~18.8k rows) is used, with no subsampling. The 60 s timeout is applied via the "
                  "socket default, since scikit-learn does not expose a timeout parameter."),
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[done] wrote {OUT} in {time.time() - t0:.0f}s" if (t0 := TIME0) else "")


TIME0 = time.time()
if __name__ == "__main__":
    main()
