"""SID assignment + metrics + Spearman analysis.

  1. assign_sids(..., tie_break='popularity'): when multiple items collide on
     base SID (c1..cL), the most popular one gets cnt=0. GPT2 tends to predict
     cnt=0, so this routes the "default" mass to the popular item -- expected to help Recall@1.
  2. compute_sid_metrics: a single call that returns the full vector of SID
     descriptors for a given (rqvae, embeds, popularity, sequences). Use these
     across many checkpoints to build a Spearman table against Recall/NDCG.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# Popularity and sequences

def load_sequences(path: str) -> List[List[int]]:
    """Read sequential_data.txt: 'user_id item1 item2 ...' per line."""
    seqs = []
    with open(path, 'r') as f:
        for line in f:
            toks = line.strip().split()
            if len(toks) < 2:
                continue
            seqs.append([int(t) for t in toks[1:]])
    return seqs


def compute_item_popularity(sequences: Sequence[Sequence[int]], n_items: int) -> np.ndarray:
    """Item-frequency vector. Index is internal item id (0..n_items-1)."""
    pop = np.zeros(n_items, dtype=np.int64)
    for seq in sequences:
        for it in seq:
            idx = it - 1  # 1-based -> 0-based
            if 0 <= idx < n_items:
                pop[idx] += 1
    return pop


# SID assignment

@torch.no_grad()
def encode_base_sids(rqvae, embeds: torch.Tensor, device: str, batch_size: int = 1024
                     ) -> List[Tuple[int, ...]]:
    """Run RQ-VAE encoder over all items, return tuple SID per item """
    rqvae.eval()
    n = embeds.shape[0]
    base = [None] * n
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        out = rqvae(embeds[start:end].to(device))
        codes = torch.stack(out['sids'], dim=1).cpu().tolist()  # B x L
        for i, c in zip(range(start, end), codes):
            base[i] = tuple(c)
    return base


def assign_sids(base: List[Tuple[int, ...]], tie_break: str = 'count',
                popularity: np.ndarray | None = None,
                latents: torch.Tensor | None = None,
                semantic_codes: List[int] | None = None,
                ) -> Tuple[Dict[int, Tuple[int, ...]], Dict[Tuple[int, ...], List[int]], int]:
    """Append disambiguation suffix to colliding base SIDs.

    tie_break:
      'count'      -> first-come first-served
      'popularity' -> most popular item in cluster gets cnt=0, rest by descending popularity
      'centroid'   -> closest item to cluster centroid (in `latents` space) gets cnt=0,
                      others ordered by ascending distance
      'semantic'   -> use precomputed `semantic_codes[i]` as the disambig suffix
                      (semantic_codes is a global k-means assignment over residuals)
    Returns (item_sids, sid_to_item, max_dupe).
    """
    cluster: Dict[Tuple[int, ...], List[int]] = defaultdict(list)
    for i, s in enumerate(base):
        cluster[s].append(i)

    if tie_break == 'popularity':
        if popularity is None:
            raise ValueError("tie_break='popularity' requires popularity vector")
        for s, ids in cluster.items():
            ids.sort(key=lambda i: -popularity[i])
    elif tie_break == 'centroid':
        if latents is None:
            raise ValueError("tie_break='centroid' requires latents tensor")
        Z = F.normalize(latents, p=2, dim=1).cpu().numpy()
        for s, ids in cluster.items():
            if len(ids) == 1:
                continue
            sub = Z[ids]
            ctr = sub.mean(axis=0)
            ctr = ctr / (np.linalg.norm(ctr) + 1e-12)
            d = 1.0 - sub @ ctr
            order = np.argsort(d)
            cluster[s] = [ids[j] for j in order]
    elif tie_break == 'semantic':
        if semantic_codes is None:
            raise ValueError("tie_break='semantic' requires semantic_codes")

    full: Dict[int, Tuple[int, ...]] = {}
    if tie_break == 'semantic':
        used: Dict[Tuple[int, ...], set] = defaultdict(set)
        for s, ids in cluster.items():
            for i in ids:
                code = int(semantic_codes[i])
                while code in used[s]:
                    code += 1
                used[s].add(code)
                full[i] = s + (code,)
    else:
        for s, ids in cluster.items():
            for cnt, i in enumerate(ids):
                full[i] = s + (cnt,)

    sid_to_item: Dict[Tuple[int, ...], List[int]] = defaultdict(list)
    for i, sid in full.items():
        sid_to_item[sid].append(i)

    if tie_break == 'semantic':
        max_dupe = 1 + max(sid[-1] for sid in full.values())
    else:
        max_dupe = max(len(ids) for ids in cluster.values())
    return full, sid_to_item, max_dupe


# Latent extraction and semantic disambig

@torch.no_grad()
def encode_latents_and_residuals(rqvae, embeds: torch.Tensor, device: str,
                                 batch_size: int = 1024
                                 ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run encoder, return (z, residual) for each item.

    z - normalized encoder output (used for centroid tie-break)
    residual - z minus the sum of all quantized layers
    """
    rqvae.eval()
    n = embeds.shape[0]
    Zs, Rs = [], []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        x = embeds[start:end].to(device)
        x_n = F.normalize(x, p=2, dim=1)
        z = rqvae.enc(x_n)
        r = z.clone()
        for cb in rqvae.codebooks:
            _, emb_st, _ = cb(r)
            r = r - emb_st.detach()
        Zs.append(z.cpu())
        Rs.append(r.cpu())
    return torch.cat(Zs, 0), torch.cat(Rs, 0)


def kmeans_residual_codes(residuals: torch.Tensor, k: int,
                          collision_mask: np.ndarray,
                          n_iter: int = 25, seed: int = 0) -> List[int]:
    """Global k-means over residual vectors of items that collide with someone.

    Items that have a unique base SID get code 0.
    Items in any collision cluster get a code in [0, k-1] from the global
    residual k-means so 'cnt=k' carries comparable meaning across clusters.
    """
    rng = np.random.RandomState(seed)
    R = residuals.cpu().numpy().astype(np.float64)
    R = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-12)
    coll_idx = np.where(collision_mask)[0]
    if len(coll_idx) == 0:
        return [0] * len(residuals)
    X = R[coll_idx]
    init = X[rng.choice(len(X), size=min(k, len(X)), replace=False)]
    C = init.copy()
    for _ in range(n_iter):
        d = 1.0 - X @ C.T
        a = d.argmin(axis=1)
        new_C = np.zeros_like(C)
        for j in range(C.shape[0]):
            m = a == j
            if m.any():
                v = X[m].mean(axis=0)
                new_C[j] = v / (np.linalg.norm(v) + 1e-12)
            else:
                new_C[j] = C[j]
        if np.allclose(new_C, C, atol=1e-6):
            C = new_C
            break
        C = new_C
    d = 1.0 - X @ C.T
    a = d.argmin(axis=1)
    codes = [0] * len(residuals)
    for idx, c in zip(coll_idx, a):
        codes[int(idx)] = int(c)
    return codes


def build_collision_mask(base: List[Tuple[int, ...]]) -> np.ndarray:
    counts = Counter(base)
    return np.array([counts[s] > 1 for s in base], dtype=bool)


def build_trie(sid_to_item: Dict[Tuple[int, ...], List[int]]) -> dict:
    """Nested dict trie. Leaves are item ids."""
    trie: dict = {}
    for sid, ids in sid_to_item.items():
        node = trie
        for c in sid[:-1]:
            node = node.setdefault(c, {})
        node[sid[-1]] = ids[0]
    return trie


# SID metrics

def per_level_entropy(base: List[Tuple[int, ...]], codebook_size: int) -> np.ndarray:
    """log2 entropy of the code distribution at each level. Max = log2(codebook_size)."""
    L = len(base[0])
    out = np.zeros(L)
    arr = np.array(base, dtype=np.int64)  # (n_items, L)
    for l in range(L):
        counts = np.bincount(arr[:, l], minlength=codebook_size).astype(np.float64)
        p = counts / counts.sum()
        p = p[p > 0]
        out[l] = -(p * np.log2(p)).sum()
    return out


def per_level_unique(base: List[Tuple[int, ...]]) -> np.ndarray:
    L = len(base[0])
    arr = np.array(base, dtype=np.int64)
    return np.array([len(np.unique(arr[:, l])) for l in range(L)])


def collision_stats(base: List[Tuple[int, ...]]) -> Dict[str, float]:
    counts = Counter(base)
    n = len(base)
    n_unique = len(counts)
    n_collisions = sum(c - 1 for c in counts.values() if c > 1)
    max_dupe = max(counts.values())
    return dict(
        n_unique_base=n_unique,
        n_collisions=n_collisions,
        max_dupe=max_dupe,
        cfr=(n - n_collisions) / n,
    )


def path_avg_similarity(base: List[Tuple[int, ...]], embeds: torch.Tensor) -> float:
    """Mean intra-cluster cosine similarity in embedding space. NaN if no collisions."""
    cluster = defaultdict(list)
    for i, s in enumerate(base):
        cluster[s].append(i)
    sims = []
    emb_n = F.normalize(embeds, p=2, dim=1)
    for ids in cluster.values():
        if len(ids) < 2:
            continue
        E = emb_n[ids]
        S = (E @ E.T).cpu().numpy()
        m = S[np.triu_indices(len(ids), k=1)]
        sims.extend(m.tolist())
    return float(np.mean(sims)) if sims else float('nan')


def behavioral_pas(base: List[Tuple[int, ...]], sequences: Sequence[Sequence[int]],
                   n_items: int) -> float:
    """Mean intra-cluster Jaccard over the sets of users who interacted with each item.

    Two items are 'behaviorally similar' if the same users tend to use both. A
    high value means colliding items really are interchangeable in user behaviour.
    """
    item_users: List[set] = [set() for _ in range(n_items)]
    for u, seq in enumerate(sequences):
        for it in seq:
            idx = it - 1
            if 0 <= idx < n_items:
                item_users[idx].add(u)

    cluster = defaultdict(list)
    for i, s in enumerate(base):
        cluster[s].append(i)
    sims = []
    for ids in cluster.values():
        if len(ids) < 2:
            continue
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                A, B = item_users[ids[a]], item_users[ids[b]]
                if not A and not B:
                    continue
                sims.append(len(A & B) / max(1, len(A | B)))
    return float(np.mean(sims)) if sims else float('nan')


def zipf_alpha(base: List[Tuple[int, ...]]) -> float:
    """Slope of log-rank vs log-frequency for full SIDs.

    Very flat (<0.5) = noisy, low signal. Very steep (>1.5) = collapse.
    """
    counts = sorted(Counter(base).values(), reverse=True)
    if len(counts) < 10:
        return float('nan')
    ranks = np.arange(1, len(counts) + 1)
    x, y = np.log(ranks), np.log(counts)
    return float(-np.polyfit(x, y, 1)[0])


def codebook_utilization_ratio(base: List[Tuple[int, ...]], codebook_size: int) -> float:
    """|unique base SIDs| / codebook_size^L. Tiny for L=4, K=256, n=12k items."""
    L = len(base[0])
    return len(set(base)) / (codebook_size ** L)


def compute_sid_metrics(rqvae, embeds: torch.Tensor, device: str,
                        codebook_size: int,
                        popularity: np.ndarray | None = None,
                        sequences: Sequence[Sequence[int]] | None = None,
                        ) -> Dict[str, float]:
    """One-shot SID descriptor for a given RQ-VAE checkpoint."""
    base = encode_base_sids(rqvae, embeds, device)
    L = len(base[0])

    ent = per_level_entropy(base, codebook_size)
    uniq = per_level_unique(base)
    cs = collision_stats(base)

    out = {
        'n_layers': L,
        'codebook_size': codebook_size,
        **{f'entropy_l{l}': float(ent[l]) for l in range(L)},
        **{f'unique_l{l}': int(uniq[l]) for l in range(L)},
        'entropy_mean': float(ent.mean()),
        'entropy_min': float(ent.min()),
        'unique_total_base': cs['n_unique_base'],
        'n_collisions': cs['n_collisions'],
        'max_dupe': cs['max_dupe'],
        'cfr': cs['cfr'],
        'pas_emb': path_avg_similarity(base, embeds),
        'cur_total': codebook_utilization_ratio(base, codebook_size),
        'zipf_alpha_full': zipf_alpha(base),
    }
    if sequences is not None:
        out['pas_behavioral'] = behavioral_pas(base, sequences, embeds.shape[0])
    return out


# Spearman correlation table

def spearman_table(metric_rows: List[Dict[str, float]],
                   target_keys: Sequence[str]) -> 'pd.DataFrame':
    """Spearman corr between every numeric SID metric and each target metric.

    metric_rows: list of dicts, one per checkpoint. Each dict must contain BOTH
                 the SID metrics (e.g. cfr, entropy_l0, pas_emb) and the target
                 metrics (e.g. Recall@10, NDCG@10).
    """
    import pandas as pd
    from scipy.stats import spearmanr

    df = pd.DataFrame(metric_rows)
    sid_cols = [c for c in df.columns
                if c not in target_keys and pd.api.types.is_numeric_dtype(df[c])]
    rows = []
    for sid_col in sid_cols:
        row = {'sid_metric': sid_col}
        for tgt in target_keys:
            if df[sid_col].nunique() < 2 or df[tgt].nunique() < 2:
                row[tgt] = float('nan')
                continue
            rho, p = spearmanr(df[sid_col], df[tgt])
            row[tgt] = round(float(rho), 3)
            row[f'{tgt}_p'] = round(float(p), 3)
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.set_index('sid_metric')
