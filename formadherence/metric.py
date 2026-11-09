"""The three-axis Form-Adherence metric.

Inputs
------
embeddings  : (T, d) per-frame embeddings for one song.
target_form : str, e.g. "ABACA" -- the requested letter sequence.
boundaries  : optional interior frame indices. If omitted, discovered with Foote
              novelty, asking for len(target_form) - 1 boundaries.

Each segment is treated as a SET of frame embeddings and segments are compared
with optimal transport (ot.segment_distance). OT ignores frame order.

Two scoring modes (cfg.score_mode)
----------------------------------
"relative" (default). Every comparison is made INSIDE one song, so the song's
overall distance scale doesn't matter (what made the per-song AUC work on real
audio). m = the song's distance unit: midpoint of its median same-letter and
median different-letter distance (song_scale).

  Axis 1  Distinctness, per letter PAIR (A-B, A-C, ...). Each different-letter
          section pair is compared with the repeats of the letter whose
          repeats are TIGHTER: mean_k sigmoid((d_ij - s_k) / (kappa * m)).
          So "B is really an A" or "the bridge is really an A" fails that
          letter pair, and A's drifting repeats are not blamed on A-B.
  Axis 2  Similarity, per LETTER: each repeat pair vs the song's
          different-letter pairs, mean_k sigmoid((d_k - d_same) / (kappa * m)).
  Axis 3  Variation (copy-paste guard), per letter:
          q = matching-halves distance of the two repeats / their mean
          inside-change (first half vs second half). Exact copy -> q ~ 0.
          0 at q <= copy_floor, 1 at q >= copy_floor + copy_ramp.
          "Drifted" repeats are Axis 2's job, not this one.

  No in-song reference? Forms with no repeated letter (AB, ABC) have nothing to
  compare Axis 1 against, and single-letter forms (AAA) nothing for Axis 2.
  There a calibrated, unit-free anchor is used: matching-halves distance /
  inside-change vs repeat_ratio (fitted by calibrate.py between human repeat
  and different-letter ratios). Without it, "ABC" or "AAAAA" accepted any song.

  Reported scale: comparisons are 0.5 at chance; reported per-pair/per-letter
  scores map 0.5 + chance_margin -> 0 and 1 -> 1, then combine with a
  geometric mean (aggregate="geometric": every requested contrast/sameness
  must hold, so a merged letter pair or dropped bridge is flagged instead of
  averaged away), each term floored at comparison_floor so one noisy
  comparison can't zero a song by itself.
  For RL, result.raw_axes holds graded versions (no chance clipping, no
  copy-floor dead zone): the reward uses those so weak rollouts don't tie.
  contrast_temperature (kappa), repeat_ratio, copy_floor: fit with calibrate.py.

"absolute" (legacy). The original formulation: absolute distances through one
global temperature and a fixed (variation_floor, tau) band. Kept only so old
numbers can be reproduced.

Also reported (relative mode): `contrast` = soft AUC of all diff vs same pairs,
and `auc` = the hard per-song AUC used in the preliminary experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
import numpy as np

from .config import FormAdherenceConfig
from .segmentation import (Segment, boundaries_to_segments, form_aware_boundaries,
                           segment_boundaries)
from .ot import segment_distance


@dataclass
class FormAdherenceResult:
    overall: float
    distinctness: float          # Axis 1
    similarity: float            # Axis 2
    variation: float             # Axis 3
    target_form: str
    segments: list[Segment]
    score_mode: str = "relative"
    contrast: float | None = None        # soft AUC (diff vs same), relative mode
    auc: float | None = None             # hard per-song AUC
    segmentation_fallback: bool = False  # True if discovery failed -> uniform cuts
    # Diagnostics for interpretability / reward-hacking audits:
    per_letter_similarity: dict[str, float] = field(default_factory=dict)
    per_letter_variation: dict[str, float] = field(default_factory=dict)
    cross_pair_distances: dict[str, float] = field(default_factory=dict)
    same_pair_distances: dict[str, float] = field(default_factory=dict)
    within_segment_distances: dict[str, float] = field(default_factory=dict)
    copy_ratios: dict[str, float] = field(default_factory=dict)  # Axis 3 q per pair
    per_pair_distinctness: dict[str, float] = field(default_factory=dict)
    per_letter_pair_distinctness: dict[str, float] = field(default_factory=dict)
    variation_method: str = "embedding"          # which Axis 3 ran
    note_change: dict[str, float] = field(default_factory=dict)  # chroma_dtw: f per pair
    # relative mode: graded versions for the RL reward -- distinctness and
    # similarity before chance_to_zero (0.5 = chance, never clipped) and
    # variation without the dead zone below copy_floor
    raw_axes: tuple | None = None

    def as_reward_vector(self) -> np.ndarray:
        """The (distinctness, similarity, variation) vector for RL shaping."""
        return np.array([self.distinctness, self.similarity, self.variation],
                        dtype=np.float64)


# ---------------------------------------------------------------------------
# small helpers (also imported by scripts)
# ---------------------------------------------------------------------------

def _segment_distribution(seg: Segment, embeddings: np.ndarray,
                          max_frames: int, rng: np.random.Generator) -> np.ndarray:
    """Frames of a segment, uniformly subsampled to a cap (order preserved)."""
    frames = seg.frames(embeddings)
    if len(frames) == 0:
        raise ValueError(f"empty segment {seg}")
    if len(frames) > max_frames:
        idx = rng.choice(len(frames), size=max_frames, replace=False)
        idx.sort()
        frames = frames[idx]
    return frames


_OT_CACHE: dict | None = None


def enable_ot_cache(on: bool = True) -> None:
    """Memoize segment distances by frame content. Scoring the SAME song
    against many forms or settings (validity checks) then costs one OT pass.
    Call clear_ot_cache() between songs to bound memory."""
    global _OT_CACHE
    _OT_CACHE = {} if on else None


def clear_ot_cache() -> None:
    if _OT_CACHE is not None:
        _OT_CACHE.clear()


def _ot(a: np.ndarray, b: np.ndarray, cfg: FormAdherenceConfig) -> float:
    if _OT_CACHE is None:
        return segment_distance(a, b, cfg, metric="cosine")
    key = (a.shape, b.shape, hash(a.tobytes()), hash(b.tobytes()),
           cfg.ot_backend, cfg.sinkhorn_eps, cfg.sinkhorn_debias)
    if key not in _OT_CACHE:
        _OT_CACHE[key] = segment_distance(a, b, cfg, metric="cosine")
    return _OT_CACHE[key]


def _sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(x, dtype=np.float64)))


def hard_auc(same, diff) -> float | None:
    """P(random diff distance > random same distance), ties = 0.5.
    0.5 = chance, 1.0 = perfect. None if either list is empty."""
    same = np.sort(np.asarray(same, dtype=float))
    diff = np.asarray(diff, dtype=float)
    if len(same) == 0 or len(diff) == 0:
        return None
    lo = np.searchsorted(same, diff, side="left")
    hi = np.searchsorted(same, diff, side="right")
    return float((lo.sum() + 0.5 * (hi - lo).sum()) / (len(same) * len(diff)))


def soft_auc(same, diff, scale: float, kappa: float) -> float | None:
    """Smooth AUC: mean sigmoid((d_diff - d_same) / (kappa * scale))."""
    same = np.asarray(same, dtype=float)
    diff = np.asarray(diff, dtype=float)
    if len(same) == 0 or len(diff) == 0:
        return None
    t = max(kappa * scale, 1e-12)
    return float(_sigmoid((diff[None, :] - same[:, None]) / t).mean())


def copy_score(q: float, floor: float, ramp: float) -> float:
    """0 at/below floor, linear up to 1 at floor + ramp."""
    if q <= floor:
        return 0.0
    return float(np.clip((q - floor) / max(ramp, 1e-9), 0.0, 1.0))


def _chance_to_zero(x: float, margin: float = 0.0) -> float:
    """0.5 + margin -> 0, 1 -> 1, anything below clipped to 0. margin > 0
    means a comparison must beat chance by that much before it counts."""
    lo = 0.5 + margin
    return float(np.clip((x - lo) / max(1.0 - lo, 1e-9), 0.0, 1.0))


def _copy_ratio(i, j, d, halves, within, m, cfg) -> float:
    """Axis 3 statistic for a same-letter pair (i, j).

    q = (distance between matching halves of i and j)
        / (mean inside-change of i and j)

    An exact copy gives q ~ 0 (each frame has an identical twin). A genuine
    repeat -- even of the same material -- differs at least as much as a
    section differs from itself, so q is ~1 or more. Halves are used on both
    sides so the sample sizes match. Falls back to d / m (song median) when a
    segment is too short to split.
    """
    if halves[i] is not None and halves[j] is not None and i in within and j in within:
        num = 0.5 * (_ot(halves[i][0], halves[j][0], cfg) +
                     _ot(halves[i][1], halves[j][1], cfg))
        den = 0.5 * (within[i] + within[j])
        if den > 1e-12:
            return float(num / den)
        return 0.0 if num <= 1e-12 else float("inf")
    return float(d / m) if m > 1e-9 else 0.0


def _halves(frames: np.ndarray):
    """First and second temporal half of a segment; None if too short."""
    n = len(frames)
    if n < 4:
        return None
    k = (n + 1) // 2
    return frames[:k], frames[k:]


# ---------------------------------------------------------------------------
# legacy absolute-mode pieces
# ---------------------------------------------------------------------------

def _sim_from_dist(dist: float, cfg: FormAdherenceConfig) -> float:
    """Absolute mode: OT distance -> [0, 1] similarity via global temperature."""
    return float(np.exp(-dist / max(cfg.similarity_temperature, 1e-8)))


def _variation_score(dist: float, cfg: FormAdherenceConfig) -> float:
    """Absolute mode: 1 inside (variation_floor, tau), smooth ramps outside."""
    floor = cfg.variation_floor
    tau = cfg.tau
    ramp = max(cfg.variation_softness * tau, 1e-6)
    if dist <= floor:
        return 0.0
    if dist < floor + ramp:
        return float(np.clip((dist - floor) / ramp, 0.0, 1.0))
    if dist >= tau:
        return float(np.clip((tau + ramp - dist) / ramp, 0.0, 1.0))
    return 1.0


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------

def _resolve_boundaries(E, letters, boundaries, cfg):
    T = len(E)
    n_segments = len(letters)
    if boundaries is not None:
        b = segment_boundaries(E, boundaries=boundaries)
        if len(b) != n_segments - 1:
            raise ValueError(
                f"supplied {len(boundaries)} boundaries -> {len(b)} valid inside "
                f"(0, {T}); target form {''.join(letters)!r} needs "
                f"{n_segments - 1}")
        return b, False
    if cfg.segmentation_method == "form_aware":
        b = form_aware_boundaries(
            E, letters, kernel_size=cfg.form_aware_kernel,
            novelty_weight=cfg.form_aware_novelty_weight,
            min_len=max(1, int(round(cfg.form_aware_min_len_s * cfg.frame_rate))))
        return b, len(b) != n_segments - 1
    elif cfg.segmentation_method != "foote":
        raise ValueError(f"unknown segmentation_method {cfg.segmentation_method!r}")
    n_b = cfg.n_boundaries if cfg.n_boundaries is not None else n_segments - 1
    b = segment_boundaries(E, boundaries=None, n_boundaries=n_b,
                           window=cfg.mp_window,
                           exclusion_zone=cfg.mp_exclusion_zone)
    if len(b) != n_segments - 1:
        # discovery failed -> uniform cuts, and SAY SO in the result
        return [round(T * k / n_segments) for k in range(1, n_segments)], True
    return b, False


def _combine(axes, cfg):
    w = cfg.normalized_weights()
    axes = np.asarray(axes, dtype=float)
    if cfg.combine == "geometric":
        return float(np.exp(np.sum(np.array(w) * np.log(axes + 1e-12))))
    return float(np.dot(w, axes))


def score_form_adherence(
    embeddings: np.ndarray,
    target_form,
    boundaries: list[int] | None = None,
    config: FormAdherenceConfig | None = None,
    seed: int = 0,
    audio=None,
    sample_rate: int | None = None,
    chroma=None,
) -> FormAdherenceResult:
    """Score one song. Pass `audio` + `sample_rate` (or precomputed `chroma` =
    (C, fps) from variation.chroma_features) to measure Axis 3 on musical
    content (cfg.variation_method="chroma_dtw"); without them Axis 3 uses the
    embedding copy ratio."""
    result = _score_core(embeddings, target_form, boundaries, config, seed)
    cfg = config or FormAdherenceConfig()
    if (cfg.score_mode == "relative" and cfg.variation_method == "chroma_dtw"
            and (chroma is not None or audio is not None)):
        result = _apply_chroma_variation(result, cfg, audio, sample_rate, chroma)
    return result


def _apply_chroma_variation(result, cfg, audio, sample_rate, chroma):
    from dataclasses import replace as _replace
    from .variation import chroma_features, variation_from_chroma
    if chroma is None:
        if sample_rate is None:
            raise ValueError("audio given without sample_rate")
        chroma = chroma_features(audio, sample_rate, fps=cfg.chroma_fps,
                                 method=cfg.chroma_method)
    C, fps = chroma
    letters = [s.label for s in result.segments]
    seconds = [(s.start / cfg.frame_rate, s.end / cfg.frame_rate) for s in result.segments]
    v = variation_from_chroma(C, fps, seconds, letters, cfg)
    if v["per_letter"]:
        variation = _aggregate(v["per_letter"].values(), cfg.aggregate)
        soft = _aggregate(v["per_letter_soft"].values(), cfg.aggregate)
    else:
        variation, soft = 1.0, 1.0                 # no repeated letter: vacuous
    raw = result.raw_axes
    return _replace(
        result, variation=variation, per_letter_variation=v["per_letter"],
        overall=_combine([result.distinctness, result.similarity, variation], cfg),
        raw_axes=(raw[0], raw[1], soft) if raw is not None else None,
        variation_method="chroma_dtw", note_change=v["pairs"])


def _score_core(embeddings, target_form, boundaries, config, seed) -> FormAdherenceResult:
    cfg = config or FormAdherenceConfig()
    rng = np.random.default_rng(seed)
    E = np.asarray(embeddings, dtype=np.float64)
    T = len(E)

    # target_form may be a string ("ABACA") or a list of labels (manifests can
    # have multi-character labels like "X26" when a song has > 26 families).
    letters = [str(x) for x in target_form]
    target_form = "".join(letters) if all(len(x) == 1 for x in letters) \
        else "|".join(letters)
    n = len(letters)
    if n < 1:
        raise ValueError("target_form must contain at least one letter")
    if T < n:
        raise ValueError(f"{T} frames cannot hold {n} segments")

    b, fallback = _resolve_boundaries(E, letters, boundaries, cfg)
    segs = boundaries_to_segments(b, T, labels=letters)
    dists = [_segment_distribution(s, E, cfg.max_frames_per_segment, rng)
             for s in segs]

    # all pairwise segment distances, computed once
    D: dict[tuple[int, int], float] = {}
    for i, j in combinations(range(n), 2):
        D[(i, j)] = _ot(dists[i], dists[j], cfg)
    same_keys = [(i, j) for (i, j) in D if letters[i] == letters[j]]
    diff_keys = [(i, j) for (i, j) in D if letters[i] != letters[j]]
    cross_pairs = {f"{letters[i]}{i}-{letters[j]}{j}": D[(i, j)] for i, j in diff_keys}
    same_pairs = {f"{letters[i]}{i}-{letters[j]}{j}": D[(i, j)] for i, j in same_keys}

    groups: dict[str, list[int]] = {}
    for k, lab in enumerate(letters):
        groups.setdefault(lab, []).append(k)

    if cfg.score_mode == "absolute":
        return _score_absolute(cfg, letters, segs, D, same_keys, diff_keys,
                               groups, cross_pairs, same_pairs, fallback,
                               target_form)
    if cfg.score_mode != "relative":
        raise ValueError(f"unknown score_mode {cfg.score_mode!r}")

    return _score_relative(cfg, letters, segs, dists, D, same_keys, diff_keys,
                           groups, cross_pairs, same_pairs, fallback, target_form)


def _aggregate(values, how: str) -> float:
    """Combine per-letter / per-letter-pair scores. geometric (default): every
    requested sameness/contrast must hold -- one failure pulls the axis down,
    so a dropped bridge is flagged instead of averaged away. mean: lenient."""
    v = np.asarray(list(values), dtype=float)
    if how == "mean":
        return float(v.mean())
    if v.min() <= 1e-12:
        return 0.0
    return float(np.exp(np.mean(np.log(v))))


def song_scale(same_d, diff_d, all_d) -> float:
    """The song's distance unit for soft comparisons: midpoint of its typical
    same-letter and different-letter distance (median of all pairs if one kind
    is missing). Unlike the median of all pairs, it doesn't shift with the
    form's mix of same/different pairs (AABAA vs ABCDEA)."""
    if len(same_d) and len(diff_d):
        return 0.5 * (float(np.median(same_d)) + float(np.median(diff_d)))
    return float(np.median(all_d)) if len(all_d) else 0.0


def _score_relative(cfg, letters, segs, dists, D, same_keys, diff_keys, groups,
                    cross_pairs, same_pairs, fallback, target_form):
    kappa = cfg.contrast_temperature
    all_d = np.array(list(D.values()), dtype=float)
    m = song_scale([D[k] for k in same_keys], [D[k] for k in diff_keys], all_d)
    m_safe = max(m, 1e-9)
    t = max(kappa * m_safe, 1e-12)
    diff_d = np.array([D[k] for k in diff_keys], dtype=float)

    # inside-change of each section (first vs second half), and matching-halves
    # distances between sections -- the song's own unit for the anchor fallback
    halves = [_halves(x) for x in dists]
    within: dict[int, float] = {}
    for i, h in enumerate(halves):
        if h is not None:
            within[i] = _ot(h[0], h[1], cfg)
    _hcache: dict[tuple[int, int], float] = {}

    def half_dist(i, j):
        if (i, j) not in _hcache:
            _hcache[(i, j)] = 0.5 * (_ot(halves[i][0], halves[j][0], cfg) +
                                     _ot(halves[i][1], halves[j][1], cfg))
        return _hcache[(i, j)]

    def ratio(i, j):
        """matching-halves distance in units of the two sections' inside-change;
        None if a section is too short to split."""
        if halves[i] is None or halves[j] is None:
            return None
        den = 0.5 * (within[i] + within[j])
        return half_dist(i, j) / max(den, 1e-12)

    def anchor_score(r, farther_is_better: bool):
        """sigmoid on log(ratio / repeat_ratio): above repeat_ratio looks like
        different material, below looks like a repeat (calibrated threshold)."""
        if r is None:
            return 0.5
        z = (np.log(max(r, 1e-12)) - np.log(cfg.repeat_ratio)) / max(cfg.ratio_temperature, 1e-6)
        return float(_sigmoid(z if farther_is_better else -z))

    same_by_letter: dict[str, list[float]] = {}
    for i, j in same_keys:
        same_by_letter.setdefault(letters[i], []).append(D[(i, j)])
    all_same = [d for v in same_by_letter.values() for d in v]

    # ---- Axis 1: distinctness, per letter PAIR -----------------------------
    # Different letters must be farther apart than repeats. Reference for pair
    # (i, j): the TIGHTER repeat scale of letters i and j (min of their median
    # repeat distances), so A's drifting repeats are not blamed on A-B; if
    # neither repeats, the song's repeats; if the song has no repeats at all,
    # the calibrated anchor (ratio to inside-change vs repeat_ratio).
    pair_scores: dict[tuple[str, str], list[float]] = {}
    per_pair: dict[str, float] = {}
    for i, j in diff_keys:
        if all_same:
            own = [L for L in (letters[i], letters[j]) if L in same_by_letter]
            if own:
                # the letter whose repeats are TIGHTER sets the bar
                L = min(own, key=lambda x: float(np.median(same_by_letter[x])))
                ref = np.asarray(same_by_letter[L])
            else:
                ref = np.asarray(all_same)
            # compare with every reference repeat (smooth AUC), not just their
            # median: more comparisons -> less noise at chance level
            sc = float(_sigmoid((D[(i, j)] - ref) / t).mean())
        else:
            sc = anchor_score(ratio(i, j), farther_is_better=True)
        key = tuple(sorted((letters[i], letters[j])))
        pair_scores.setdefault(key, []).append(sc)
        per_pair[f"{letters[i]}{i}-{letters[j]}{j}"] = sc
    letter_pair_raw = {f"{a}-{b}": float(np.mean(v)) for (a, b), v in pair_scores.items()}

    # ---- Axes 2 & 3, per LETTER ---------------------------------------------
    # Similarity: repeats must be closer than the song's different-letter pairs;
    # if the form has no different letters (AAAA), use the calibrated anchor.
    sim_raw: dict[str, float] = {}
    var_: dict[str, float] = {}
    var_soft: dict[str, float] = {}
    copy_ratios: dict[str, float] = {}
    for lab, members in groups.items():
        if len(members) < 2:
            continue
        sims, vars_, vars_soft = [], [], []
        for i, j in combinations(members, 2):
            d = D[(i, j)]
            if len(diff_d):
                sims.append(float(_sigmoid((diff_d - d) / t).mean()))
            else:
                sims.append(anchor_score(ratio(i, j), farther_is_better=False))
            q = _copy_ratio(i, j, d, halves, within, m, cfg)
            copy_ratios[f"{lab}{i}-{lab}{j}"] = q
            vars_.append(copy_score(q, cfg.copy_floor, cfg.copy_ramp))
            # graded version for the reward: no dead zone below copy_floor, so
            # a policy stuck near copy-paste still sees which way is better
            vars_soft.append(float(np.clip(q / max(cfg.copy_floor + cfg.copy_ramp, 1e-9),
                                           0.01, 1.0)))
        sim_raw[lab] = float(np.mean(sims))
        var_[lab] = float(np.mean(vars_))
        var_soft[lab] = float(np.mean(vars_soft))

    agg = cfg.aggregate
    distinct_raw = _aggregate(letter_pair_raw.values(), agg) if letter_pair_raw else 1.0
    similarity_raw = _aggregate(sim_raw.values(), agg) if sim_raw else 1.0
    variation = _aggregate(var_.values(), agg) if var_ else 1.0

    # Comparison scores are 0.5 at chance. Reported axes map chance -> 0,
    # perfect -> 1 (cfg.chance_to_zero), so a failed requirement reads ~0.
    # raw_axes keep the unclipped values: graded everywhere, which the RL
    # reward needs (clipping makes everything below chance tie at 0).
    f = (lambda x: _chance_to_zero(x, cfg.chance_margin)) if cfg.chance_to_zero \
        else (lambda x: x)
    letter_pair = {k: f(v) for k, v in letter_pair_raw.items()}
    per_letter_sim = {k: f(v) for k, v in sim_raw.items()}
    # floor before the geometric mean: a requirement sitting exactly at chance
    # pulls the axis down hard, but one noisy comparison landing on 0 must not
    # zero the whole song (that made identical failures score 0 OR 0.75).
    fl = cfg.comparison_floor
    distinctness = _aggregate([max(v, fl) for v in letter_pair.values()], agg) \
        if letter_pair else 1.0
    similarity = _aggregate([max(v, fl) for v in per_letter_sim.values()], agg) \
        if per_letter_sim else 1.0

    same_d = [D[k] for k in same_keys]
    return FormAdherenceResult(
        overall=_combine([distinctness, similarity, variation], cfg),
        distinctness=distinctness, similarity=similarity, variation=variation,
        target_form=target_form, segments=segs, score_mode="relative",
        contrast=soft_auc(same_d, diff_d, m_safe, kappa),
        auc=hard_auc(same_d, diff_d), segmentation_fallback=fallback,
        per_letter_similarity=per_letter_sim, per_letter_variation=var_,
        cross_pair_distances=cross_pairs, same_pair_distances=same_pairs,
        within_segment_distances={f"{letters[i]}{i}": v for i, v in within.items()},
        copy_ratios=copy_ratios,
        per_pair_distinctness={k: f(v) for k, v in per_pair.items()},
        per_letter_pair_distinctness=letter_pair,
        raw_axes=(distinct_raw, similarity_raw,
                  _aggregate(var_soft.values(), agg) if var_soft else 1.0),
    )


def _score_absolute(cfg, letters, segs, D, same_keys, diff_keys, groups,
                    cross_pairs, same_pairs, fallback, target_form):
    """Legacy formulation, numerically identical to the original code."""
    if diff_keys:
        mean_cross = float(np.mean([D[k] for k in diff_keys]))
        distinctness = 1.0 - _sim_from_dist(mean_cross, cfg)
    else:
        distinctness = 1.0
    per_letter_sim, per_letter_var = {}, {}
    for lab, members in groups.items():
        if len(members) < 2:
            continue
        ds = [D[(i, j)] for i, j in combinations(members, 2)]
        per_letter_sim[lab] = float(np.mean([_sim_from_dist(d, cfg) for d in ds]))
        per_letter_var[lab] = float(np.mean([_variation_score(d, cfg) for d in ds]))
    similarity = float(np.mean(list(per_letter_sim.values()))) if per_letter_sim else 1.0
    variation = float(np.mean(list(per_letter_var.values()))) if per_letter_var else 1.0
    return FormAdherenceResult(
        overall=_combine([distinctness, similarity, variation], cfg),
        distinctness=distinctness, similarity=similarity, variation=variation,
        target_form=target_form, segments=segs, score_mode="absolute",
        contrast=None, auc=hard_auc([D[k] for k in same_keys], [D[k] for k in diff_keys]),
        segmentation_fallback=fallback,
        per_letter_similarity=per_letter_sim, per_letter_variation=per_letter_var,
        cross_pair_distances=cross_pairs, same_pair_distances=same_pairs,
    )


def song_ratios(embeddings, boundaries, labels, config=None, seed: int = 0):
    """For calibration: matching-halves distance / inside-change for every
    same-letter pair and every different-letter pair of an annotated song.
    Returns (same_ratios, diff_ratios)."""
    cfg = config or FormAdherenceConfig()
    rng = np.random.default_rng(seed)
    E = np.asarray(embeddings, dtype=np.float64)
    letters = [str(x) for x in labels]
    b = segment_boundaries(E, boundaries=boundaries)
    segs = boundaries_to_segments(b, len(E), labels=letters[:len(b) + 1])
    halves = [_halves(_segment_distribution(s, E, cfg.max_frames_per_segment, rng))
              for s in segs]
    within = {i: _ot(h[0], h[1], cfg) for i, h in enumerate(halves) if h is not None}
    same, diff = [], []
    for i, j in combinations(range(len(segs)), 2):
        if i not in within or j not in within:
            continue
        num = 0.5 * (_ot(halves[i][0], halves[j][0], cfg) + _ot(halves[i][1], halves[j][1], cfg))
        r = num / max(0.5 * (within[i] + within[j]), 1e-12)
        (same if letters[i] == letters[j] else diff).append(r)
    return same, diff
