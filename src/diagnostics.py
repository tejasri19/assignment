"""
Diagnostic toolkit: run this BEFORE choosing an intervention. It quantifies:

  1. Per-class IoU (not just mIoU) on source vs target.
  2. Per-band radiometric shift (KS statistic + mean/std deltas).
  3. Domain separability ("proxy A-distance") via a tiny classifier trained
     to tell source chips from target chips using only images (no labels).
  4. 2D embedding projection (PCA, or t-SNE if scikit-learn is available)
     of encoder bottleneck features, source vs target.

Everything degrades gracefully if optional deps (scipy, scikit-learn,
matplotlib) are not installed -- only numpy + torch are required for the
core numbers; plots are skipped with a warning if matplotlib is missing.
"""

import os
import json
import numpy as np
import torch


def per_class_iou(preds, labels, num_classes, ignore_index=-1):
    """preds, labels: numpy arrays of any shape, same shape, integer class ids."""
    ious = np.full(num_classes, np.nan)
    valid = labels != ignore_index
    preds = preds[valid]
    labels = labels[valid]
    for c in range(num_classes):
        pred_c = preds == c
        label_c = labels == c
        union = np.logical_or(pred_c, label_c).sum()
        if union == 0:
            continue  # class absent from both -> undefined, skip
        intersection = np.logical_and(pred_c, label_c).sum()
        ious[c] = intersection / union
    return ious


def mean_iou(ious):
    return float(np.nanmean(ious))


def confusion_matrix(preds, labels, num_classes, ignore_index=-1):
    valid = labels != ignore_index
    preds = preds[valid]
    labels = labels[valid]
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    idx = labels * num_classes + preds
    counts = np.bincount(idx, minlength=num_classes * num_classes)
    return counts.reshape(num_classes, num_classes)


def band_shift_report(source_images, target_images):
    """source_images, target_images: (N, C, H, W) numpy arrays.
    Returns per-band dict with mean/std deltas and KS statistic (if scipy
    available) between the two pixel-value distributions."""
    C = source_images.shape[1]
    report = []
    try:
        from scipy.stats import ks_2samp
        have_scipy = True
    except ImportError:
        have_scipy = False

    for c in range(C):
        s = source_images[:, c].ravel()
        t = target_images[:, c].ravel()
        entry = {
            "band": c,
            "source_mean": float(s.mean()), "source_std": float(s.std()),
            "target_mean": float(t.mean()), "target_std": float(t.std()),
            "mean_shift": float(t.mean() - s.mean()),
            "std_ratio": float(t.std() / (s.std() + 1e-8)),
        }
        if have_scipy:
            # Subsample for speed on large chip sets.
            rng = np.random.default_rng(0)
            s_sub = rng.choice(s, size=min(5000, s.size), replace=False)
            t_sub = rng.choice(t, size=min(5000, t.size), replace=False)
            stat, pvalue = ks_2samp(s_sub, t_sub)
            entry["ks_statistic"] = float(stat)
            entry["ks_pvalue"] = float(pvalue)
        report.append(entry)
    return report


def domain_classifier_accuracy(source_images, target_images, test_frac=0.3, seed=0):
    """Trains a tiny classifier to distinguish source vs target chips using
    only per-image summary statistics (mean & std per band) as features --
    no labels needed. High accuracy (>>0.5) confirms strong covariate shift
    ("proxy A-distance" = 2*(1 - 2*error)).

    Falls back to a manual logistic-regression-via-gradient-descent in pure
    numpy if scikit-learn isn't installed.
    """
    def summarize(images):
        # (N, C, H, W) -> (N, 2*C) of [mean_per_band, std_per_band]
        mean = images.mean(axis=(2, 3))
        std = images.std(axis=(2, 3))
        return np.concatenate([mean, std], axis=1)

    X_s = summarize(source_images)
    X_t = summarize(target_images)
    X = np.concatenate([X_s, X_t], axis=0)
    y = np.concatenate([np.zeros(len(X_s)), np.ones(len(X_t))])

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]
    n_test = int(len(X) * test_frac)
    X_test, y_test = X[:n_test], y[:n_test]
    X_train, y_train = X[n_test:], y[n_test:]

    # Standardize features.
    mu, sigma = X_train.mean(0), X_train.std(0) + 1e-8
    X_train = (X_train - mu) / sigma
    X_test = (X_test - mu) / sigma

    try:
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
    except ImportError:
        acc = _numpy_logreg_accuracy(X_train, y_train, X_test, y_test)

    a_distance = 2 * (1 - 2 * (1 - acc))
    return {"domain_classifier_accuracy": float(acc), "proxy_a_distance": float(a_distance)}


def _numpy_logreg_accuracy(X_train, y_train, X_test, y_test, lr=0.1, epochs=300):
    n, d = X_train.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(epochs):
        z = X_train @ w + b
        p = 1 / (1 + np.exp(-z))
        grad_w = X_train.T @ (p - y_train) / n
        grad_b = (p - y_train).mean()
        w -= lr * grad_w
        b -= lr * grad_b
    z_test = X_test @ w + b
    p_test = 1 / (1 + np.exp(-z_test))
    preds = (p_test > 0.5).astype(float)
    return float((preds == y_test).mean())


@torch.no_grad()
def extract_embeddings(model, images_tensor, device="cpu", batch_size=32):
    """images_tensor: (N, C, H, W) torch tensor. Returns (N, feature_dim) numpy."""
    model.eval()
    feats = []
    for i in range(0, images_tensor.shape[0], batch_size):
        batch = images_tensor[i:i + batch_size].to(device)
        _, feat = model(batch, return_features=True)
        feats.append(feat.cpu().numpy())
    return np.concatenate(feats, axis=0)


def project_2d(embeddings):
    """t-SNE if scikit-learn available, else PCA via SVD (pure numpy)."""
    try:
        from sklearn.manifold import TSNE
        return TSNE(n_components=2, init="pca", random_state=0).fit_transform(embeddings)
    except ImportError:
        centered = embeddings - embeddings.mean(axis=0, keepdims=True)
        u, s, vt = np.linalg.svd(centered, full_matrices=False)
        return centered @ vt[:2].T


def plot_domain_embeddings(source_emb2d, target_emb2d, save_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed -- skipping embedding plot.")
        return
    plt.figure(figsize=(6, 6))
    plt.scatter(source_emb2d[:, 0], source_emb2d[:, 1], label="source (Iowa)", alpha=0.6, s=15)
    plt.scatter(target_emb2d[:, 0], target_emb2d[:, 1], label="target (Sahel)", alpha=0.6, s=15)
    plt.legend()
    plt.title("Encoder bottleneck embeddings: source vs target")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved embedding plot to {save_path}")


def save_json_report(report_dict, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(report_dict, f, indent=2)
    print(f"Saved diagnostics report to {path}")
