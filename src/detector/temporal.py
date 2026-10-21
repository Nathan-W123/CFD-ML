"""Alarm rules over a camera's per-minute window scores, and per-fire / per-camera-hour metrics.

A camera scores a window ending every minute (src/detector/dense.py). Instead of alarming on every window above
a threshold, a rule looks at the recent scores:

* "single" (R0): alarm when the current window's P(smoke) >= T;
* "persist" (R1): alarm when at least k of the last m windows (the current one included; missing windows
  count as below) have P(smoke) >= T, or the current one has P(smoke) >= T2 (a bypass for strong evidence);
* "smooth" (R2): alarm when the mean logit of the last m windows (at least half of them present, the
  current one included) is >= logit(T).

Scores are given on a minute grid per sequence (NaN where no window could be scored). A fog-vetoed window
should be passed with P(smoke) = 0. Metrics: for a fire sequence, the delay from the first visible plume to
the first window end (at or after the plume) on which the rule fires; for smoke-free footage, alarm events
(firings grouped with a hold-off: a firing within `holdoff` minutes of the current event's start belongs to
it) per hour of scored window ends.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class Rule:
    kind: str                      # "single", "persist" or "smooth"
    T: float                       # threshold on P(smoke) (persist: T1; smooth: on the mean logit's sigmoid)
    k: int = 1
    m: int = 1
    T2: Optional[float] = None     # persist: bypass threshold for the current window

    def name(self) -> str:
        if self.kind == "single":
            return f"single T={self.T:.3f}"
        if self.kind == "persist":
            return f"{self.k} of {self.m} >= {self.T:.3f}" + (f", or one >= {self.T2:.3f}" if self.T2 else "")
        return f"mean logit of {self.m} >= logit({self.T:.3f})"


def _logit(p):
    p = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _trailing(x: np.ndarray, m: int) -> np.ndarray:
    """Sum over the last m entries (the current one included), x NaN-free."""
    c = np.concatenate([[0.0], np.cumsum(x)])
    i = np.arange(len(x))
    return c[i + 1] - c[np.maximum(i + 1 - m, 0)]


def fires(rule: Rule, p: np.ndarray) -> np.ndarray:
    """Boolean per minute: the rule fires on the window ending at that minute (p: P(smoke) per minute, NaN where
    no window was scored)."""
    p = np.asarray(p, np.float64)
    present = ~np.isnan(p)
    q = np.where(present, p, -1.0)
    if rule.kind == "single":
        return q >= rule.T
    if rule.kind == "persist":
        out = present & (_trailing((q >= rule.T).astype(np.float64), rule.m) >= rule.k)
        if rule.T2 is not None:
            out |= q >= rule.T2
        return out
    if rule.kind == "smooth":
        lg = np.where(present, _logit(np.where(present, p, 0.5)), 0.0)
        n = _trailing(present.astype(np.float64), rule.m)
        mean = _trailing(lg, rule.m) / np.maximum(n, 1)
        return present & (n >= math.ceil(rule.m / 2)) & (mean >= _logit(rule.T))
    raise ValueError(f"unknown rule kind {rule.kind!r}")


def events(fire_minutes: Sequence[int], holdoff: int = 30) -> int:
    """Number of alarm events among firing minutes (sorted ascending) with a hold-off in minutes."""
    n, start = 0, None
    for t in fire_minutes:
        if start is None or t - start >= holdoff:
            n += 1
            start = t
    return n


def evaluate(rule: Rule, seqs: List[Dict], holdoff: int = 30, horizons=(5, 10, 15, 20, 30)) -> Dict:
    """Per-fire detection and false-alarm events of a rule over sequences.

    Each sequence: {"p": P(smoke) per minute (NaN: not scored), "off": minutes from the first visible plume per
    minute (fire sequences; NaN for ordinary footage), "kind": "fire" or "ordinary"}. A fire sequence's minutes
    with off < 0 are smoke-free footage; its fire is detected at the first minute with off >= 0 on which the
    rule fires (delay = off). Ordinary sequences are smoke-free throughout.
    """
    delays, n_fire = [], 0
    fa = {"fire_pre": [0, 0, 0], "ordinary": [0, 0, 0]}      # events, scored minutes, firing minutes
    for s in seqs:
        f = fires(rule, s["p"])
        scored = ~np.isnan(np.asarray(s["p"], np.float64))
        minutes = np.arange(len(f))
        if s["kind"] == "fire":
            off = np.asarray(s["off"], np.float64)
            n_fire += 1
            post = f & (off >= 0)
            delays.append(float(off[post][0]) if post.any() else math.inf)
            pre = scored & (off < 0)
            key = "fire_pre"
        else:
            pre = scored
            key = "ordinary"
        fa[key][0] += events(minutes[f & pre].tolist(), holdoff)
        fa[key][1] += int(pre.sum())
        fa[key][2] += int((f & pre).sum())
    d = np.array(delays)
    out = {"rule": rule.name(), "fires": n_fire}
    for h in horizons:
        out[f"fires_within_{h}"] = float(np.mean(d <= h)) if n_fire else None
    found = d[np.isfinite(d)]
    out["median_delay_detected"] = float(np.median(found)) if len(found) else None
    out["fires_detected"] = float(np.mean(np.isfinite(d))) if n_fire else None
    for key, (ev, minutes, firing) in fa.items():
        out[f"{key}_hours"] = minutes / 60.0
        out[f"{key}_events"] = ev
        out[f"{key}_events_per_hour"] = ev / (minutes / 60.0) if minutes else None
        out[f"{key}_window_share"] = firing / minutes if minutes else None
    tot_min = fa["fire_pre"][1] + fa["ordinary"][1]
    out["events_per_hour"] = (fa["fire_pre"][0] + fa["ordinary"][0]) / (tot_min / 60.0) if tot_min else None
    return out
