"""Alarm rules over a camera's per-minute window scores, and per-fire / per-camera-hour metrics.

A camera scores a window ending every minute (src/detector/dense.py). Instead of alarming on every window above
a threshold, a rule looks at the recent scores:

* "single" (R0): alarm when the current window's P(smoke) >= T;
* "persist" (R1): alarm when at least k of the last m windows (the current one included; missing windows
  count as below) have P(smoke) >= T, or the current one has P(smoke) >= T2 (a bypass for strong evidence);
* "smooth" (R2): alarm when the mean logit of the last m windows (at least half of them present, the
  current one included) is >= logit(T);
* a change gate G (R3, with "single" or "persist"): a window counts only if its change score (the detector's
  reference tile head logit: each tile against the same camera 14 minutes before the window) is also >= G,
  so a look-alike that was already there 14 minutes earlier does not alarm however smoky it looks.

Scores are given on a minute grid per sequence (NaN where no window could be scored). A fog-vetoed window
should be passed with P(smoke) = 0. Firings become alarm events with a hold-off per camera: a firing within
`holdoff` minutes of the current event's start belongs to it (the operator has reviewed that camera). Metrics:

* a fire is detected at the start of the first alarm event at or after the first visible plume (delay = that
  window's end minus the plume time). An event that started before the plume is a false alarm, dismissed,
  and its hold-off also holds back an alarm on the fire: false alarms cost detection time. (Counting any
  firing after the plume instead would credit look-alikes that were already alarming before the plume;
  reported as fires_firing_within_*);
* false-alarm events (events starting before the plume, and on ordinary footage) per hour of scored
  window ends, and the share of smoke-free windows on which the rule fires.
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
    G: Optional[float] = None      # change gate on the second score (single, persist)

    def name(self) -> str:
        gate = f" with change >= {self.G:.2f}" if self.G is not None else ""
        if self.kind == "single":
            return f"single T={self.T:.3f}{gate}"
        if self.kind == "persist":
            return (f"{self.k} of {self.m} >= {self.T:.3f}{gate}"
                    + (f", or one >= {self.T2:.3f}" if self.T2 else ""))
        return f"mean logit of {self.m} >= logit({self.T:.3f})"


def _logit(p):
    p = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _trailing(x: np.ndarray, m: int) -> np.ndarray:
    """Sum over the last m entries (the current one included), x NaN-free."""
    c = np.concatenate([[0.0], np.cumsum(x)])
    i = np.arange(len(x))
    return c[i + 1] - c[np.maximum(i + 1 - m, 0)]


def fires(rule: Rule, p: np.ndarray, g: Optional[np.ndarray] = None) -> np.ndarray:
    """Boolean per minute: the rule fires on the window ending at that minute (p: P(smoke) per minute, NaN where
    no window was scored; g: the change score per minute, needed with a change gate)."""
    p = np.asarray(p, np.float64)
    present = ~np.isnan(p)
    q = np.where(present, p, -1.0)
    above = q >= rule.T
    if rule.G is not None:
        if g is None:
            raise ValueError("a change-gated rule needs the change scores")
        above &= np.nan_to_num(np.asarray(g, np.float64), nan=-np.inf) >= rule.G
    if rule.kind == "single":
        return above
    if rule.kind == "persist":
        out = present & (_trailing(above.astype(np.float64), rule.m) >= rule.k)
        if rule.T2 is not None:
            out |= q >= rule.T2
        return out
    if rule.kind == "smooth":
        lg = np.where(present, _logit(np.where(present, p, 0.5)), 0.0)
        n = _trailing(present.astype(np.float64), rule.m)
        mean = _trailing(lg, rule.m) / np.maximum(n, 1)
        return present & (n >= math.ceil(rule.m / 2)) & (mean >= _logit(rule.T))
    raise ValueError(f"unknown rule kind {rule.kind!r}")


def event_starts(fire_minutes: Sequence[int], holdoff: int = 30) -> List[int]:
    """Start minutes of the alarm events among firing minutes (sorted ascending) with a hold-off in minutes."""
    starts = []
    for t in fire_minutes:
        if not starts or t - starts[-1] >= holdoff:
            starts.append(int(t))
    return starts


def events(fire_minutes: Sequence[int], holdoff: int = 30) -> int:
    """Number of alarm events among firing minutes (sorted ascending) with a hold-off in minutes."""
    return len(event_starts(fire_minutes, holdoff))


def evaluate(rule: Rule, seqs: List[Dict], holdoff: int = 30, horizons=(5, 10, 15, 20, 30)) -> Dict:
    """Per-fire detection and false-alarm events of a rule over sequences.

    Each sequence: {"p": P(smoke) per minute (NaN: not scored), "off": minutes from the first visible plume per
    minute (fire sequences; NaN for ordinary footage), "kind": "fire" or "ordinary", and for change-gated rules
    "g": the change score per minute}. A fire sequence's minutes
    with off < 0 are smoke-free footage; its fire is detected at the first alarm event starting at a minute
    with off >= 0 (delay = off; see the module docstring). Ordinary sequences are smoke-free throughout.
    """
    delays, firing_delays, n_fire = [], [], 0
    fa = {"fire_pre": [0, 0, 0], "ordinary": [0, 0, 0]}      # events, scored minutes, firing minutes
    for s in seqs:
        f = fires(rule, s["p"], s.get("g"))
        scored = ~np.isnan(np.asarray(s["p"], np.float64))
        minutes = np.arange(len(f))
        starts = np.array(event_starts(minutes[f].tolist(), holdoff), int)
        if s["kind"] == "fire":
            off = np.asarray(s["off"], np.float64)
            n_fire += 1
            post = f & (off >= 0)
            firing_delays.append(float(off[post][0]) if post.any() else math.inf)
            post_starts = starts[off[starts] >= 0] if len(starts) else starts
            delays.append(float(off[post_starts[0]]) if len(post_starts) else math.inf)
            pre = scored & (off < 0)
            n_false = int(np.sum(off[starts] < 0)) if len(starts) else 0
            key = "fire_pre"
        else:
            pre = scored
            n_false = len(starts)
            key = "ordinary"
        fa[key][0] += n_false
        fa[key][1] += int(pre.sum())
        fa[key][2] += int((f & pre).sum())
    d, fd = np.array(delays), np.array(firing_delays)
    out = {"rule": rule.name(), "fires": n_fire, "holdoff": holdoff}
    for h in horizons:
        out[f"fires_within_{h}"] = float(np.mean(d <= h)) if n_fire else None
        out[f"fires_firing_within_{h}"] = float(np.mean(fd <= h)) if n_fire else None
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
