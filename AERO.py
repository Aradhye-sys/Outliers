from __future__ import annotations

import numpy as np
from scipy.stats import chi2
from sklearn.covariance import LedoitWolf, MinCovDet
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

__all__ = [
    "aero_anomaly_scores",
    "AERO",
    "AERO_blocks",
    "recommend_aero_removal",
]

_AERO_BLOCK_TRAJECTORY_CACHE = {}

def _ensure_mxn(spectra, wn=None) -> np.ndarray:
    x = np.asarray(spectra, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"spectra must be 2D, got {x.shape}")
    if wn is None:
        return x
    w = np.asarray(wn, dtype=float).ravel()
    if x.shape[0] == w.size:
        return x
    if x.shape[1] == w.size:
        return x.T
    raise ValueError(f"Neither spectra axis matches len(wn). spectra={x.shape}, len(wn)={w.size}")

def _robust_snv_rows(x: np.ndarray) -> np.ndarray:
    med = np.median(x, axis=1, keepdims=True)
    mad = np.median(np.abs(x - med), axis=1, keepdims=True) + 1e-12
    return (x - med) / mad

def _robust_z(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    med = np.median(v)
    mad = np.median(np.abs(v - med)) + 1e-12
    return (v - med) / (1.4826 * mad)

def _percentile_rank(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(v.size, dtype=float)
    ranks[order] = np.arange(v.size, dtype=float)
    return ranks / max(v.size - 1, 1)

def _normalise01(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    lo, hi = np.nanpercentile(v, [1, 99])
    return np.clip((v - lo) / (hi - lo + 1e-12), 0.0, 1.0)

def _bh_fdr(pvals: np.ndarray, q: float):
    p = np.asarray(pvals, dtype=float)
    n = p.size
    if n == 0:
        return np.zeros(0, dtype=bool), np.nan
    order = np.argsort(p)
    p_sorted = p[order]
    thresh = (np.arange(1, n + 1) / n) * float(q)
    below = p_sorted <= thresh
    if not np.any(below):
        return np.zeros(n, dtype=bool), np.nan
    kmax = np.max(np.where(below)[0])
    p_cut = p_sorted[kmax]
    return p <= p_cut, float(p_cut)

def _robust_md(scores_use: np.ndarray, cov_mode: str = "mcd") -> np.ndarray:
    s = np.asarray(scores_use, dtype=float)
    if s.ndim != 2:
        raise ValueError(f"scores_use must be 2D, got {s.shape}")
    if s.shape[1] == 1:
        x = s[:, 0]
        med = np.median(x)
        mad = np.median(np.abs(x - med)) + 1e-12
        return np.abs((x - med) / (1.4826 * mad))

    mode = cov_mode.lower()
    if mode == "mcd":
        try:
            return np.sqrt(MinCovDet(random_state=0).fit(s).mahalanobis(s))
        except Exception:
            mode = "lw"
    if mode == "lw":
        cov = LedoitWolf().fit(s)
        diff = s - cov.location_
        invcov = np.linalg.pinv(cov.covariance_)
        md2 = np.einsum("ij,jk,ik->i", diff, invcov, diff)
        return np.sqrt(np.maximum(md2, 0.0))
    if mode == "emp":
        mu = s.mean(axis=0, keepdims=True)
        diff = s - mu
        cov = np.cov(diff, rowvar=False) + np.eye(s.shape[1]) * 1e-9
        invcov = np.linalg.pinv(cov)
        md2 = np.einsum("ij,jk,ik->i", diff, invcov, diff)
        return np.sqrt(np.maximum(md2, 0.0))
    raise ValueError("cov_mode must be one of: 'mcd', 'lw', 'emp'")

def _map_core_sensitivity(sensitivity: float):
    s = float(np.clip(sensitivity, 0.0, 1.0))
    if s < 0.35:
        cov_mode = "emp"
    elif s < 0.70:
        cov_mode = "lw"
    else:
        cov_mode = "mcd"
    return {
        "n_sig_pcs": int(np.clip(int(round(4 + 8 * s)), 2, 50)),
        "cov_mode": cov_mode,
        "max_remove_frac": float(0.02 * (1 - s) + 0.12 * s),
        "fdr_q": float(0.005 * (1 - s) + 0.10 * s),
    }

def _pca_md_score(
    features_nxm: np.ndarray,
    *,
    n_components: int,
    exclude_pc1: bool,
    robust_feature_scale: bool = False,
    random_state: int = 0,
) -> np.ndarray:
    x = np.asarray(features_nxm, dtype=float)
    if robust_feature_scale:
        center = np.median(x, axis=0, keepdims=True)
        scale = np.median(np.abs(x - center), axis=0, keepdims=True) + 1e-12
        x = (x - center) / scale
    else:
        x = x - x.mean(axis=0, keepdims=True)

    n_components = int(min(n_components, x.shape[0] - 2, x.shape[1]))
    if n_components < 2:
        return np.zeros(x.shape[0], dtype=float)

    scores = PCA(n_components=n_components, random_state=random_state).fit_transform(x)
    scores_use = scores[:, 1:] if exclude_pc1 else scores
    dim = max(scores_use.shape[1], 1)

    try:
        cov = MinCovDet(random_state=random_state).fit(scores_use)
        md2 = cov.mahalanobis(scores_use)
    except Exception:
        cov = LedoitWolf().fit(scores_use)
        diff = scores_use - cov.location_
        invcov = np.linalg.pinv(cov.covariance_)
        md2 = np.einsum("ij,jk,ik->i", diff, invcov, diff)

    pvals = np.maximum(1.0 - chi2.cdf(np.maximum(md2, 0.0), df=dim), 1e-300)
    return -np.log10(pvals)

def _pca_reconstruction_score(features_nxm: np.ndarray, n_components: int = 14, random_state: int = 0) -> np.ndarray:
    x = np.asarray(features_nxm, dtype=float)
    center = np.median(x, axis=0, keepdims=True)
    scale = np.median(np.abs(x - center), axis=0, keepdims=True) + 1e-12
    xs = (x - center) / scale
    n_components = int(min(n_components, xs.shape[0] - 2, xs.shape[1]))
    if n_components < 2:
        return np.zeros(xs.shape[0], dtype=float)
    pca = PCA(n_components=n_components, random_state=random_state)
    latent = pca.fit_transform(xs)
    recon = pca.inverse_transform(latent)
    return np.abs(_robust_z(np.sqrt(np.mean((xs - recon) ** 2, axis=1))))

def _spectral_qc_components(raw_nxm: np.ndarray, robust_shape_nxm: np.ndarray) -> dict[str, np.ndarray]:
    d1 = np.diff(robust_shape_nxm, axis=1)
    d2 = np.diff(robust_shape_nxm, n=2, axis=1)
    median_shape = np.median(robust_shape_nxm, axis=0)
    median_norm = np.linalg.norm(median_shape) + 1e-12
    corr_distance = 1.0 - (
        robust_shape_nxm @ median_shape
    ) / ((np.linalg.norm(robust_shape_nxm, axis=1) * median_norm) + 1e-12)

    return {
        "mean_intensity": np.abs(_robust_z(raw_nxm.mean(axis=1))),
        "intensity_std": np.abs(_robust_z(raw_nxm.std(axis=1))),
        "dynamic_range": np.abs(_robust_z(np.ptp(raw_nxm, axis=1))),
        "roughness": np.abs(_robust_z(np.mean(np.abs(d1), axis=1))),
        "max_jump": np.abs(_robust_z(np.max(np.abs(d1), axis=1))),
        "curvature": np.abs(_robust_z(np.mean(np.abs(d2), axis=1))),
        "median_mismatch": np.abs(_robust_z(corr_distance)),
    }

def _spectral_qc_score(qc: dict[str, np.ndarray]) -> np.ndarray:
    parts = np.column_stack(list(qc.values()))
    return np.max(parts, axis=1) + 0.5 * np.sum(parts > 4.0, axis=1)

def _tail_separation(rank_values: np.ndarray) -> float:
    r = np.asarray(rank_values, dtype=float)
    hi = np.nanmedian(r[r >= np.nanpercentile(r, 90)])
    mid = np.nanmedian(r[(r >= np.nanpercentile(r, 40)) & (r <= np.nanpercentile(r, 60))])
    spread = np.nanpercentile(r, 75) - np.nanpercentile(r, 25) + 1e-12
    return float(np.clip((hi - mid) / spread, 0.0, 4.0) / 4.0)

def _view_agreement(name: str, ranks: dict[str, np.ndarray]) -> float:
    target = np.asarray(ranks[name], dtype=float)
    others = [np.asarray(v, dtype=float) for k, v in ranks.items() if k != name]
    if not others:
        return 0.5
    vals = []
    for other in others:
        if np.std(target) < 1e-12 or np.std(other) < 1e-12:
            continue
        vals.append(np.corrcoef(target, other)[0, 1])
    if not vals:
        return 0.5
    # Agreement helps, but a view is allowed to be complementary rather than identical.
    return float(np.clip(0.5 + 0.5 * np.nanmedian(vals), 0.15, 1.0))

def _bulk_penalty(rank_values: np.ndarray) -> float:
    r = np.asarray(rank_values, dtype=float)
    high_fraction = float(np.mean(r > 0.8))
    # Penalise views that call almost everything high, but do not suppress broad contamination completely.
    return float(np.clip(1.15 - high_fraction, 0.45, 1.0))

def _adaptive_weights(ranks: dict[str, np.ndarray]) -> dict[str, float]:
    priors = {
        "shape_no_pc1": 1.35,
        "shape_with_pc1": 1.15,
        "standardized_shape": 1.10,
        "first_derivative": 0.85,
        "second_derivative": 0.55,
        "reconstruction_error": 0.95,
        "spectral_qc": 1.05,
    }
    raw = {}
    for name, values in ranks.items():
        sep = _tail_separation(values)
        agree = _view_agreement(name, ranks)
        penalty = _bulk_penalty(values)
        raw[name] = priors.get(name, 1.0) * (0.45 + 0.55 * sep) * (0.60 + 0.40 * agree) * penalty
    total = sum(raw.values()) + 1e-12
    return {name: float(val / total) for name, val in raw.items()}

def _score_tail_recommendation(score: np.ndarray, stability: np.ndarray | None = None) -> dict:
    score = np.asarray(score, dtype=float)
    n = score.size
    ordered = np.sort(score)
    x = np.linspace(0, 1, n)
    y = _normalise01(ordered)

    best = None
    for k in range(max(3, int(0.50 * n)), max(4, int(0.95 * n))):
        p1 = np.polyfit(x[: k + 1], y[: k + 1], 1)
        p2 = np.polyfit(x[k:], y[k:], 1)
        err = np.sum((y[: k + 1] - np.polyval(p1, x[: k + 1])) ** 2)
        err += np.sum((y[k:] - np.polyval(p2, x[k:])) ** 2)
        ratio = (p2[0] + 1e-12) / (p1[0] + 1e-12)
        if ratio < 1.4:
            err *= 1.35
        if best is None or err < best[0]:
            best = (err, k, ratio)
    elbow_pct = 100.0 * (1.0 - x[best[1]])

    med = np.median(score)
    mad = np.median(np.abs(score - med)) + 1e-12
    robust_pct = 100.0 * np.mean(score > med + 3.0 * 1.4826 * mad)

    try:
        gm = GaussianMixture(n_components=2, covariance_type="full", n_init=20, random_state=0).fit(score.reshape(-1, 1))
        high = int(np.argmax(gm.means_.ravel()))
        mix_pct = 100.0 * np.mean(gm.predict_proba(score.reshape(-1, 1))[:, high] > 0.5)
    except Exception:
        mix_pct = np.nan

    broad_cues = np.asarray([elbow_pct, mix_pct], dtype=float)
    broad_finite = broad_cues[np.isfinite(broad_cues)]
    broad = float(np.nanmedian(broad_finite)) if broad_finite.size else 10.0

    stability_75 = np.nan
    stability_90 = np.nan
    if stability is not None:
        st = np.asarray(stability, dtype=float)
        stability_75 = 100.0 * float(np.mean(st > 0.75))
        stability_90 = 100.0 * float(np.mean(st > 0.90))

    # The broad elbow/mixture estimate marks the suspicious tail. It can be too
    # aggressive when the dataset has continuous drift, so the default operating
    # suggestion is anchored to robust separation and bootstrap stability.
    core_cues = np.asarray([robust_pct, stability_75, stability_90], dtype=float)
    core_cues = core_cues[np.isfinite(core_cues)]
    positive_core = core_cues[core_cues > 0.0]
    if positive_core.size:
        core = float(np.nanmedian(positive_core))
    else:
        core = float(0.5 * broad)

    recommended = 0.55 * core + 0.45 * broad
    low = min(core, recommended)
    high = max(broad, recommended)

    return {
        "recommended_percent": float(np.clip(recommended, 1.0, 45.0)),
        "recommended_range": (float(np.clip(low, 1.0, 60.0)), float(np.clip(high, 1.0, 60.0))),
        "core_percent": float(np.clip(core, 0.0, 60.0)),
        "broad_tail_percent": float(np.clip(broad, 0.0, 60.0)),
        "elbow_percent": float(np.clip(elbow_pct, 0.0, 100.0)),
        "robust_z3_percent": float(np.clip(robust_pct, 0.0, 100.0)),
        "mixture_percent": float(mix_pct) if np.isfinite(mix_pct) else np.nan,
        "stability75_percent": float(stability_75) if np.isfinite(stability_75) else np.nan,
        "stability90_percent": float(stability_90) if np.isfinite(stability_90) else np.nan,
        "tail_slope_ratio": float(best[2]),
    }

def _bootstrap_stability(
    ranks: dict[str, np.ndarray],
    weights: dict[str, float],
    n_boot: int,
    random_state: int,
) -> np.ndarray:
    names = list(weights)
    base = sum(weights[name] * ranks[name] for name in names)
    rec = _score_tail_recommendation(base)
    pct = rec["recommended_percent"]
    n = base.size
    n_select = int(np.clip(round(n * pct / 100.0), 1, n))
    rng = np.random.default_rng(random_state)
    selected = np.zeros(n, dtype=float)
    for _ in range(int(max(n_boot, 1))):
        view_names = list(rng.choice(names, size=len(names), replace=True))
        score_b = np.zeros(n, dtype=float)
        for name in view_names:
            score_b += weights[name] * ranks[name]
        order = np.argsort(score_b)
        selected[order[-n_select:]] += 1.0
    return selected / float(max(n_boot, 1))

def _aero_iterative_fdr_core(
    spectra_mxn: np.ndarray,
    aero_score: np.ndarray,
    *,
    sensitivity: float = 0.64,
    random_state: int = 0,
    exclude_pc1: bool = True,
    max_loops: int = 50,
    min_remaining: int = 20,
) -> dict:
    x_mxn = np.asarray(spectra_mxn, dtype=float)
    x = x_mxn.T.copy()
    n = x.shape[0]
    params = _map_core_sensitivity(sensitivity)
    aero_score = np.asarray(aero_score, dtype=float)

    x -= x.mean(axis=0, keepdims=True)
    keep_mask = np.ones(n, dtype=bool)
    core_rank = np.zeros(n, dtype=float)
    core_pvalue = np.ones(n, dtype=float)
    core_md = np.zeros(n, dtype=float)
    history = []

    for loop in range(int(max_loops)):
        idx = np.where(keep_mask)[0]
        n_cur = idx.size
        if n_cur < max(int(min_remaining), int(params["n_sig_pcs"]) + 2):
            break

        x_cur = x[idx]
        n_components = int(min(params["n_sig_pcs"], x_cur.shape[0] - 1, x_cur.shape[1]))
        if n_components < 2:
            break

        scores = PCA(n_components=n_components, random_state=int(random_state)).fit_transform(x_cur)
        scores_use = scores[:, 1:] if exclude_pc1 else scores
        dim = max(scores_use.shape[1], 1)
        md = _robust_md(scores_use, cov_mode=params["cov_mode"])
        md2 = md ** 2
        pvals = np.maximum(1.0 - chi2.cdf(md2, df=dim), 1e-300)
        reject, p_cut = _bh_fdr(pvals, q=params["fdr_q"])
        out_local = np.where(reject)[0]

        max_remove = int(np.floor(params["max_remove_frac"] * n_cur))
        if max_remove < 1 and out_local.size > 0:
            max_remove = 1
        if out_local.size > max_remove and max_remove > 0:
            # Primary order follows the FDR p-value; AERO evidence breaks ties.
            order = np.lexsort((-aero_score[idx[out_local]], pvals[out_local]))
            out_local = out_local[order[:max_remove]]

        out_idx = idx[out_local]
        if out_idx.size:
            iter_strength = 1.0 - (loop / max(int(max_loops), 1))
            p_strength = _normalise01(-np.log10(np.maximum(pvals[out_local], 1e-300)))
            core_rank[out_idx] = np.maximum(
                core_rank[out_idx],
                0.70 * iter_strength + 0.30 * p_strength,
            )
            core_pvalue[out_idx] = np.minimum(core_pvalue[out_idx], pvals[out_local])
            core_md[out_idx] = np.maximum(core_md[out_idx], md[out_local])

        history.append({
            "loop": int(loop),
            "n_current": int(n_cur),
            "n_components": int(n_components),
            "score_dim_used": int(dim),
            "n_outliers_found": int(out_idx.size),
            "removed_fraction": float(out_idx.size / n_cur) if n_cur else 0.0,
            "cov_mode": params["cov_mode"],
            "fdr_q": float(params["fdr_q"]),
            "p_cut": float(p_cut) if np.isfinite(p_cut) else np.nan,
            "max_remove_frac": float(params["max_remove_frac"]),
        })

        if out_idx.size == 0:
            break
        keep_mask[out_idx] = False

    core_mask = core_rank > 0
    return {
        "core_mask": core_mask,
        "core_rank": core_rank,
        "core_pvalue": core_pvalue,
        "core_md": core_md,
        "core_indices": np.where(core_mask)[0].astype(int),
        "params": params,
        "history": history,
    }

def _budgeted_fdr_selection(
    score: np.ndarray,
    core: dict,
    remove_percent: float,
) -> tuple[np.ndarray, str, int]:
    score = np.asarray(score, dtype=float)
    n = score.size
    n_remove = int(np.clip(round(n * float(remove_percent) / 100.0), 0, n))
    out_mask = np.zeros(n, dtype=bool)
    if n_remove == 0:
        return out_mask, "manual_budget_empty", 0

    core_idx = np.asarray(core["core_indices"], dtype=int)
    if core_idx.size >= n_remove:
        order = np.lexsort((-score[core_idx], core["core_pvalue"][core_idx]))
        out_mask[core_idx[order[:n_remove]]] = True
        return out_mask, "manual_budget_capped_fdr_core", int(core_idx.size)

    out_mask[core_idx] = True
    remaining = np.where(~out_mask)[0]
    needed = n_remove - core_idx.size
    if needed > 0 and remaining.size:
        order = np.argsort(score[remaining])
        out_mask[remaining[order[-needed:]]] = True
    return out_mask, "manual_budget_fdr_core_plus_aero_fill", int(core_idx.size)

def _select_from_rank(rank: np.ndarray, n_remove: int) -> np.ndarray:
    rank = np.asarray(rank, dtype=float)
    out_mask = np.zeros(rank.size, dtype=bool)
    n_remove = int(np.clip(n_remove, 0, rank.size))
    if n_remove:
        order = np.argsort(rank)
        out_mask[order[-n_remove:]] = True
    return out_mask

def _blocks_signature(blocks, wn, random_state: int):
    parts = [int(random_state)]
    if wn is not None:
        w = np.asarray(wn, dtype=float).ravel()
        parts.extend([w.size, round(float(np.nanmean(w)), 6), round(float(np.nanstd(w)), 6)])
    for block in blocks:
        x = np.asarray(block, dtype=float)
        parts.extend([
            x.shape,
            round(float(np.nanmean(x)), 8),
            round(float(np.nanstd(x)), 8),
            round(float(np.nanmin(x)), 8),
            round(float(np.nanmax(x)), 8),
        ])
    return tuple(parts)

def _ipmd_fdr_trajectory_for_blocks(blocks, wn=None, random_state: int = 0):
    key = _blocks_signature(blocks, wn, random_state)
    cached = _AERO_BLOCK_TRAJECTORY_CACHE.get(key)
    if cached is not None:
        return cached

    try:
        try:
            from a_iPMD import detect_outliers_pca_md_auto_4
        except ModuleNotFoundError:
            from Outlier_methods.a_iPMD import detect_outliers_pca_md_auto_4
    except Exception:
        return None

    sensitivities = np.linspace(0.01, 0.8, 30)
    total_n = int(sum(np.asarray(block).shape[1] for block in blocks))
    records = []
    first_seen = np.full(total_n, np.inf, dtype=float)

    for sens in sensitivities:
        global_mask = np.zeros(total_n, dtype=bool)
        start = 0
        for block in blocks:
            n = np.asarray(block).shape[1]
            try:
                out_idx, out_mask, keep_mask, info = detect_outliers_pca_md_auto_4(
                    block,
                    wn=wn,
                    sensitivity=float(sens),
                    random_state=int(random_state),
                    verbose=False,
                    log_diagnostics=False,
                )
                out_idx = np.asarray(out_idx, dtype=int).ravel()
                out_idx = out_idx[(out_idx >= 0) & (out_idx < n)]
                global_mask[start + out_idx] = True
            except Exception:
                pass
            start += n
        first_seen[global_mask & ~np.isfinite(first_seen)] = float(sens)
        records.append({
            "sensitivity": float(sens),
            "mask": global_mask,
            "n_remove": int(global_mask.sum()),
            "removed_percent": float(global_mask.mean() * 100.0),
        })

    trajectory_rank = np.zeros(total_n, dtype=float)
    seen = np.isfinite(first_seen)
    if np.any(seen):
        trajectory_rank[seen] = 1.0 - _normalise01(first_seen[seen])

    out = {
        "records": records,
        "trajectory_rank": trajectory_rank,
        "first_seen_sensitivity": first_seen,
    }
    _AERO_BLOCK_TRAJECTORY_CACHE[key] = out
    return out

def aero_anomaly_scores(
    spectra,
    wn=None,
    random_state: int = 0,
    n_bootstrap: int = 80,
    return_details: bool = True,
):

    spectra_mxn = _ensure_mxn(spectra, wn=wn)
    raw = spectra_mxn.T
    robust_shape = _robust_snv_rows(raw)
    d1 = np.diff(robust_shape, axis=1)
    d2 = np.diff(robust_shape, n=2, axis=1)

    qc_components = _spectral_qc_components(raw, robust_shape)
    components = {
        "shape_no_pc1": _pca_md_score(robust_shape, n_components=10, exclude_pc1=True, random_state=random_state),
        "shape_with_pc1": _pca_md_score(robust_shape, n_components=10, exclude_pc1=False, random_state=random_state),
        "standardized_shape": _pca_md_score(raw, n_components=10, exclude_pc1=True, robust_feature_scale=True, random_state=random_state),
        "first_derivative": _pca_md_score(d1, n_components=8, exclude_pc1=False, random_state=random_state),
        "second_derivative": _pca_md_score(d2, n_components=6, exclude_pc1=False, random_state=random_state),
        "reconstruction_error": _pca_reconstruction_score(robust_shape, n_components=14, random_state=random_state),
        "spectral_qc": _spectral_qc_score(qc_components),
    }
    ranks = {name: _percentile_rank(values) for name, values in components.items()}
    weights = _adaptive_weights(ranks)
    adaptive_severity = sum(weights[name] * ranks[name] for name in weights)

    # Preserve the iPMD-like behaviour as the ranking backbone. The auxiliary
    # AERO views still contribute, but they no longer dilute the early PCA-MD
    # tail that is often most relevant for downstream calibration.
    ipmd_backbone = np.maximum(ranks["shape_no_pc1"], ranks["standardized_shape"])
    shape_context = 0.72 * ipmd_backbone + 0.28 * ranks["shape_with_pc1"]
    severity = (
        0.56 * _normalise01(adaptive_severity)
        + 0.34 * _normalise01(shape_context)
        + 0.10 * ipmd_backbone
    )
    stability = _bootstrap_stability(ranks, weights, n_bootstrap, random_state=random_state)
    score = (
        0.76 * _normalise01(severity)
        + 0.08 * stability
        + 0.16 * ipmd_backbone
    )
    fdr_core = _aero_iterative_fdr_core(
        spectra_mxn,
        score,
        sensitivity=0.64,
        random_state=random_state,
    )
    recommendation = _score_tail_recommendation(score, stability=stability)

    if not return_details:
        return score

    reason_scores = {
        "shape_outlier": np.maximum(ranks["shape_no_pc1"], ranks["standardized_shape"]),
        "intensity_outlier": ranks["shape_with_pc1"],
        "derivative_outlier": np.maximum(ranks["first_derivative"], ranks["second_derivative"]),
        "reconstruction_outlier": ranks["reconstruction_error"],
        "spectral_qc_outlier": ranks["spectral_qc"],
        **{f"qc_{name}": _percentile_rank(value) for name, value in qc_components.items()},
    }
    return score, {
        "components": components,
        "component_ranks": ranks,
        "component_weights": weights,
        "adaptive_severity_score": adaptive_severity,
        "ipmd_backbone_score": ipmd_backbone,
        "shape_context_score": shape_context,
        "severity_score": severity,
        "stability_score": stability,
        "fdr_core": fdr_core,
        "recommendation": recommendation,
        "reason_scores": reason_scores,
        "qc_components": qc_components,
    }


def recommend_aero_removal(spectra, wn=None, random_state: int = 0, n_bootstrap: int = 80) -> dict:
    score, details = aero_anomaly_scores(
        spectra, wn=wn, random_state=random_state, n_bootstrap=n_bootstrap, return_details=True
    )
    out = dict(details["recommendation"])
    out["score"] = score
    return out

def AERO_blocks(
    blocks,
    wn=None,
    remove_percent: float | None = None,
    random_state: int = 0,
    n_bootstrap: int = 80,
    return_details: bool = True,
):
    block_arrays = [_ensure_mxn(block, wn=wn) for block in blocks]
    block_details = []
    total_n = int(sum(block.shape[1] for block in block_arrays))
    if total_n == 0:
        raise ValueError("AERO_blocks received no spectra")

    scores = []
    core_ranks = []
    aero_ranks = []
    for block in block_arrays:
        score, details = aero_anomaly_scores(
            block,
            wn=wn,
            random_state=random_state,
            n_bootstrap=n_bootstrap,
            return_details=True,
        )
        core = details["fdr_core"]
        core_rank = np.asarray(core["core_rank"], dtype=float)
        fdr_strength = _normalise01(-np.log10(np.maximum(core["core_pvalue"], 1e-300)))
        aero_rank = _percentile_rank(score)
        block_rank = 0.62 * core_rank + 0.24 * fdr_strength + 0.14 * aero_rank
        scores.append(score)
        core_ranks.append(core_rank)
        aero_ranks.append(block_rank)
        block_details.append(details)

    global_score = np.concatenate(scores)
    if remove_percent is None:
        recs = [d["recommendation"]["recommended_percent"] for d in block_details]
        remove_percent = float(np.nanmedian(recs))
        selection_mode = "auto_blocks_fdr_budget"
    else:
        selection_mode = "manual_blocks_fdr_budget"

    n_remove = int(np.clip(round(total_n * float(remove_percent) / 100.0), 0, total_n))
    aero_global_rank = _percentile_rank(global_score)
    trajectory = _ipmd_fdr_trajectory_for_blocks(block_arrays, wn=wn, random_state=random_state)
    trajectory_record = None
    if trajectory is not None and trajectory["records"]:
        trajectory_record = min(
            trajectory["records"],
            key=lambda rec: (abs(rec["n_remove"] - n_remove), abs(rec["removed_percent"] - float(remove_percent))),
        )
        # The FDR trajectory is the removal backbone; AERO evidence only refines
        # ordering at the requested budget.
        global_rank = (
            100.0 * trajectory_record["mask"].astype(float)
            + 10.0 * trajectory["trajectory_rank"]
            + 0.25 * aero_global_rank
        )
        selection_mode += "_ipmd_trajectory_ranked"
    else:
        global_rank = np.concatenate(aero_ranks)

    global_mask = _select_from_rank(global_rank, n_remove)

    out_indices = []
    out_masks = []
    keep_masks = []
    start = 0
    for block in block_arrays:
        n = block.shape[1]
        mask = global_mask[start:start + n]
        out_masks.append(mask.copy())
        keep_masks.append(~mask)
        out_indices.append(np.where(mask)[0].astype(int))
        start += n

    info = {
        "N_total": total_n,
        "N_outliers": int(global_mask.sum()),
        "N_kept": int(total_n - global_mask.sum()),
        "selection_mode": selection_mode,
        "remove_percent": float(remove_percent),
        "score": global_score,
        "block_details": block_details,
        "trajectory_record": trajectory_record,
        "params": {
            "method": "AERO_blocks_iterative_robust_fdr_budget",
            "remove_percent": float(remove_percent),
            "random_state": int(random_state),
            "n_bootstrap": int(n_bootstrap),
        },
    }
    if return_details:
        info["global_rank"] = global_rank
        info["global_mask"] = global_mask
    return out_indices, out_masks, keep_masks, info

def AERO(
    spectra,
    wn=None,
    remove_percent: float | None = None,
    random_state: int = 0,
    n_bootstrap: int = 80,
    return_details: bool = True,
):
    spectra_mxn = _ensure_mxn(spectra, wn=wn)
    score, details = aero_anomaly_scores(
        spectra_mxn, random_state=random_state, n_bootstrap=n_bootstrap, return_details=True
    )
    requested_remove_percent = remove_percent
    if remove_percent is None:
        selection_mode = "auto"
        remove_percent = details["recommendation"]["recommended_percent"]
    else:
        selection_mode = "manual"

    n = score.size
    if requested_remove_percent is None:
        out_mask, selection_mode, core_available = _budgeted_fdr_selection(
            score,
            details["fdr_core"],
            details["recommendation"]["recommended_percent"],
        )
        selection_mode = "auto_" + selection_mode
    else:
        out_mask, selection_mode, core_available = _budgeted_fdr_selection(
            score,
            details["fdr_core"],
            float(remove_percent),
        )

    n_remove = int(out_mask.sum())
    threshold = float(np.min(score[out_mask])) if n_remove else np.inf

    out_idx = np.where(out_mask)[0].astype(int)
    keep_mask = ~out_mask
    info = {
        "N_total": int(n),
        "N_outliers": int(out_mask.sum()),
        "N_kept": int(keep_mask.sum()),
        "selection_mode": selection_mode,
        "requested_remove_percent": None if requested_remove_percent is None else float(requested_remove_percent),
        "remove_percent": float(remove_percent),
        "recommended_percent": float(details["recommendation"]["recommended_percent"]),
        "recommended_range": tuple(float(x) for x in details["recommendation"]["recommended_range"]),
        "core_percent": float(details["recommendation"]["core_percent"]),
        "broad_tail_percent": float(details["recommendation"]["broad_tail_percent"]),
        "score_threshold": threshold,
        "score": score,
        "fdr_core_available": int(core_available),
        "params": {
            "method": "AERO_iterative_robust_fdr_budget",
            "selection_mode": selection_mode,
            "requested_remove_percent": None if requested_remove_percent is None else float(requested_remove_percent),
            "remove_percent": float(remove_percent),
            "fdr_core_sensitivity": 0.64,
            "random_state": int(random_state),
            "n_bootstrap": int(n_bootstrap),
        },
    }
    if return_details:
        info.update(details)
    return out_idx, out_mask, keep_mask, info
