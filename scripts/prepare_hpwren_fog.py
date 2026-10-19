"""Build a fog-labelled real far-field set from the HPWREN camera archive (one frame per minute).

    # 1. candidate windows: METAR fog / clear evidence at airports inside each camera's view
    python scripts/prepare_hpwren_fog.py select --out data/real/hpwren_fog --start 2026-07-20 --end 2026-10-18
    # 2. download 12 frames per candidate, write blind contact sheets for labelling
    python scripts/prepare_hpwren_fog.py download --out data/real/hpwren_fog
    #    (re)write the contact sheets alone:  ... sheets --out data/real/hpwren_fog --force
    # 3. label the sheets (votes CSV), consolidate the votes, then the extractor manifest
    python scripts/prepare_hpwren_fog.py consolidate --out data/real/hpwren_fog \
        --votes configs/real/hpwren_fog_votes.csv --labels configs/real/hpwren_fog_labels.csv
    python scripts/prepare_hpwren_fog.py manifest --out data/real/hpwren_fog \\
        --labels configs/real/hpwren_fog_labels.csv
    # 4. optional: one manifest with FIgLib (smoke labels) and these windows (fog labels)
    python scripts/prepare_hpwren_fog.py merge --out data/real/far_combined.csv \\
        data/real/figlib/real.csv data/real/hpwren_fog/real.csv

Why: no public real footage at the far-field timing carries fog labels. FIgLib
(scripts/prepare_figlib.py) labels smoke only. The HPWREN cameras record one
frame per minute and the archive (https://cdn.hpwren.ucsd.edu/, about the
last three months) keeps every frame; airports near the cameras report fog in
their METARs. Imagery courtesy of HPWREN, the High Performance Wireless
Research and Education Network, UC San Diego; credit
https://www.hpwren.ucsd.edu/ whenever it is used (written to <out>/CREDIT.txt).

select
  Cameras come from the HPWREN camera catalogue (sites.js): active colour
  cameras (mobo-c) with their position, azimuth and horizontal field of view.
  Stations are the active ASOS stations of the IEM CA_ASOS network; a station
  is "in view" of a camera when it is within --max-km and inside the camera's
  horizontal field of view. Hourly and special METARs (IEM asos.py) for the
  date range are cached in <out>/_cache/. Per camera and local day
  (America/Los_Angeles):
  * fog evidence: an in-view station reports, between 03:00 and 10:00 local,
    fog (FG, incl. BCFG/MIFG/PRFG/FZFG), visibility <= 1 mile, mist (BR) with
    visibility <= 3 miles, or a ceiling (BKN/OVC/VV) at or below 1000 ft
    (low stratus, which a mountain-top camera sees as a fog deck);
  * clear evidence: every in-view station has at least 4 reports between
    03:00 and 11:00 local and all of them have visibility >= 10 miles, no
    weather code and no BKN/OVC/VV layer below 5000 ft.
  Window starts are morning slots (06:30-10:00 local every 30 min) at which the
  sun is at least 5 degrees up; a candidate takes one of the first two such
  slots of its day (chosen by a hash). Fog days come first by evidence
  strength; clear windows are drawn from the cameras that have fog candidates,
  at the same slots, so time of day and camera do not separate the classes.
  At most --max-per-camera windows of each kind per camera, round robin over
  cameras in a hashed order. Written: <out>/candidates.csv (window_id, camera,
  day, slot, kind, evidence).

  METAR evidence only proposes windows; it is NOT a label: an airport's fog
  may be outside the picture and clear airports do not rule out valley fog.
  The labels come from looking at the frames.

download
  For each candidate: the archive's image lists of the camera and day
  (https://cdn.hpwren.ucsd.edu/hpwren-cameras/<cam>/<yyyy>/<yyyymmdd>/
  <yyyymmdd>_<cam>_Q<q>.txt, Q1 = 00-03 local ... Q8 = 21-24 local), 12
  consecutive images from the slot start whose times fit a 60 s grid within
  --max-residual-s (default 8 s, below the far-field extractor's 10 s), each
  downloaded from https://cdn.hpwren.ucsd.edu/MTA/<cam>/large/<yyyymmdd>/Q<q>/
  <unix>.jpg, checked to decode, downscaled to at most --max-width pixels
  (default 1024, as FIgLib) and re-encoded as JPEG quality 90 into
  <out>/<window_id>/<unix>.jpg, with timestamps.csv (filename,timestamp_s
  relative to the first frame) and window.json (camera, times, grid start,
  evidence, credit) written last: a window with both is done and skipped on a
  rerun. Then a contact sheet per window, frames 0, 4, 8 and 11 in a 2 x 2
  grid, goes to <out>/_sheets/<sheet_id>.jpg under an opaque id, with the
  camera's burned-in banner (camera name, date, time) blanked: the sheet
  shows neither the camera nor the time nor the candidate kind, so labelling
  is blind to the METAR evidence (<out>/sheets.csv maps sheet ids to
  windows). `sheets --force` rewrites every sheet.

manifest
  Reads the labels (window_id or sheet_id, fog, smoke as 0 / 1 / empty =
  unknown, optional quality) and writes <out>/real.csv for
  scripts/extract_features.py --real-manifest: one row per labelled window
  with a known fog label and usable quality, video_path the window folder,
  start_s the fitted grid start, max_windows 1, camera_id, camera_mount and
  the split from prepare_figlib.split_for (crc32 of the camera mount), so a
  camera lands in the same split as its FIgLib sequences and a combined
  manifest stays camera-disjoint.

consolidate
  Votes (one row per sheet, round and labeller: fog, smoke as 0 / 1 / empty =
  unknown, fog_type, fog_extent, quality, confidence, note) become one label
  per sheet: fog is the value given by a majority of the votes (two votes
  must agree; a third vote, cast only where the first two disagree or one is
  unknown, decides with two of three), else unknown; smoke is 0 only when
  every vote says 0, 1 with a majority of 1, else unknown; quality is the
  majority value. Writes the labels CSV and, next to it, .stats.json with the
  label counts and the round-1 / round-2 fog agreement and Cohen's kappa.

merge
  Concatenates manifests (union of columns, in order) into one CSV and refuses
  duplicate window ids.
"""

import _bootstrap  # noqa: F401

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import math
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np

from src.data.feature_spec import N_STEPS
from src.utils.io import write_json

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_figlib import camera_mount, split_for  # noqa: E402

TZ = ZoneInfo("America/Los_Angeles")
SITES_URL = "https://www.hpwren.ucsd.edu/cameras/sites.js"
LIST_URL = "https://cdn.hpwren.ucsd.edu/hpwren-cameras/{cam}/{yyyy}/{day}/{day}_{cam}_Q{q}.txt"
IMAGE_URL = "https://cdn.hpwren.ucsd.edu/MTA/{cam}/large/{day}/Q{q}/{name}"
STATIONS_URL = "https://mesonet.agron.iastate.edu/geojson/network/CA_ASOS.geojson"
METAR_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
INCIDENTS_URL = "https://incidents.fire.ca.gov/umbraco/api/IncidentApi/List?inactive=true&year={year}"
CREDIT = ("HPWREN camera archive: imagery courtesy of HPWREN, the High Performance Wireless Research and "
          "Education Network, University of California San Diego, https://www.hpwren.ucsd.edu/. Weather "
          "evidence: METARs via the Iowa Environmental Mesonet ASOS archive (mesonet.agron.iastate.edu).")
USER_AGENT = "cfd-ml-prepare-hpwren-fog/1.0 (research; credit HPWREN)"
INTERVAL_S = 60.0
JPEG_QUALITY = 90
SHEET_FRAMES = (0, 4, 8, 11)
SHEET_TILE = (768, 512)
SHEET_BANNER_PX = 20                # top rows of a sheet tile blanked (HPWREN banner: camera name, date, time)
SHEET_SALT = "hpwren-fog-v1"
FOG_HOURS = (3, 10)                 # local hours [start, end) for fog evidence
CLEAR_HOURS = (3, 11)               # local hours for clear evidence
SLOTS = [(h, m) for h in range(6, 11) for m in (0, 30) if (6, 30) <= (h, m) <= (10, 0)]
MIN_SUN_DEG = 5.0
FOG_CODES = ("FG",)                 # matches FG, BCFG, MIFG, PRFG, FZFG
CANDIDATE_FIELDS = ["window_id", "camera", "site", "day", "slot", "start_unix_target", "kind", "split",
                    "stations", "evidence", "zoom_x"]


# =============================================================================
# Small helpers
# =============================================================================


def _get(url: str, timeout: float = 60.0, retries: int = 3) -> bytes:
    """GET with a fixed user agent and a few retries (exponential backoff) on network errors and 5xx."""
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code < 500:
                raise
            last = exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last = exc
        time.sleep(2.0 ** attempt)
    raise last


def _unit_hash(text: str) -> float:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:12], 16) / 16.0 ** 12


def solar_elevation_deg(lat: float, lon: float, t_unix: float) -> float:
    """Approximate solar elevation (degrees, about +-0.5) at a place and UTC time (low-precision ephemeris)."""
    d = t_unix / 86400.0 + 2440587.5 - 2451545.0
    g = math.radians((357.529 + 0.98560028 * d) % 360.0)
    q = (280.459 + 0.98564736 * d) % 360.0
    lam = math.radians((q + 1.915 * math.sin(g) + 0.020 * math.sin(2 * g)) % 360.0)
    eps = math.radians(23.439 - 0.00000036 * d)
    ra = math.atan2(math.cos(eps) * math.sin(lam), math.cos(lam))
    dec = math.asin(math.sin(eps) * math.sin(lam))
    gmst_h = (18.697374558 + 24.06570982441908 * d) % 24.0
    ha = math.radians(gmst_h * 15.0 + lon) - ra
    la = math.radians(lat)
    s = math.sin(la) * math.sin(dec) + math.cos(la) * math.cos(dec) * math.cos(ha)
    return math.degrees(math.asin(max(-1.0, min(1.0, s))))


def distance_bearing(lat1, lon1, lat2, lon2):
    """Great-circle distance (km) and initial bearing (degrees from north) from point 1 to point 2."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    dist = 2 * 6371.0 * math.asin(math.sqrt(a))
    brg = math.degrees(math.atan2(math.sin(dl) * math.cos(p2),
                                  math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)))
    return dist, brg % 360.0


def quarter(local: dt.datetime) -> int:
    """Archive quarter of a local time: Q1 = 00-03 ... Q8 = 21-24."""
    return local.hour // 3 + 1


# =============================================================================
# Cameras and stations
# =============================================================================


def parse_sites(text: str) -> list:
    """Active colour cameras from the HPWREN sites.js catalogue: id, site, lat, lon, elev, azimuth, hfov."""
    body = json.loads(text[text.index("{"): text.rindex("}") + 1])
    cams = []
    for site_id, site in body.items():
        for cam_id, cam in (site.get("cams") or {}).items():
            if cam.get("active") != "y" or not cam_id.endswith("-mobo-c") or cam.get("experimental") == "y":
                continue
            cams.append({"camera": cam_id, "site": site_id, "site_name": site.get("name", ""),
                         "lat": float(site["lat"]), "lon": float(site["long"]), "elev_m": float(site.get("elev", 0)),
                         "azimuth": float(cam.get("azimuth", 0)), "hfov": float(cam.get("horizontalView", 90))})
    return sorted(cams, key=lambda c: c["camera"])


def parse_stations(geojson: dict) -> list:
    """Active stations (no archive_end) of an IEM network GeoJSON: id, name, lat, lon."""
    out = []
    for f in geojson.get("features", []):
        p = f.get("properties") or {}
        if p.get("archive_end"):
            continue
        lon, lat = f["geometry"]["coordinates"][:2]
        out.append({"station": p.get("sid") or f.get("id"), "name": p.get("sname", ""),
                    "lat": float(lat), "lon": float(lon)})
    return sorted(out, key=lambda s: s["station"])


def stations_in_view(cam: dict, stations: list, max_km: float = 40.0, min_km: float = 2.0) -> list:
    """Stations within max_km of a camera and inside its horizontal field of view, nearest first."""
    hits = []
    for s in stations:
        d, b = distance_bearing(cam["lat"], cam["lon"], s["lat"], s["lon"])
        off = (b - cam["azimuth"] + 180.0) % 360.0 - 180.0
        if min_km <= d <= max_km and abs(off) <= cam["hfov"] / 2.0:
            hits.append((d, s["station"]))
    return [st for _, st in sorted(hits)]


# =============================================================================
# METAR evidence
# =============================================================================


def _float(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def parse_metar_csv(text: str) -> dict:
    """IEM asos.py 'onlycomma' rows -> {station: [report dict sorted by time]} (times UTC, unix seconds)."""
    out = defaultdict(list)
    for r in csv.DictReader(io.StringIO(text)):
        if not r.get("station") or not r.get("valid"):
            continue
        t = dt.datetime.strptime(r["valid"], "%Y-%m-%d %H:%M").replace(tzinfo=dt.timezone.utc)
        wx = (r.get("wxcodes") or "").strip()
        layers = []
        for i in range(1, 5):
            cover, base = (r.get(f"skyc{i}") or "").strip(), _float(r.get(f"skyl{i}"))
            if cover and cover != "M":
                layers.append((cover, base))
        out[r["station"]].append({"t": t.timestamp(), "vsby": _float(r.get("vsby")),
                                  "wx": "" if wx in ("", "M") else wx, "layers": layers})
    for v in out.values():
        v.sort(key=lambda x: x["t"])
    return dict(out)


def ceiling_ft(report: dict):
    """Lowest BKN / OVC / VV base (ft), or None."""
    bases = [b for c, b in report["layers"] if c in ("BKN", "OVC", "VV") and b is not None]
    return min(bases) if bases else None


def is_fog_report(r: dict) -> bool:
    toks = r["wx"].replace("-", " ").replace("+", " ").split()
    fog = any(any(code in tok for code in FOG_CODES) for tok in toks)
    mist = any("BR" in tok for tok in toks)
    v, ceil = r["vsby"], ceiling_ft(r)
    return bool(fog or (v is not None and v <= 1.0) or (mist and v is not None and v <= 3.0)
                or (ceil is not None and ceil <= 1000.0))


def is_clear_report(r: dict) -> bool:
    ceil = ceiling_ft(r)
    return bool(r["vsby"] is not None and r["vsby"] >= 10.0 and not r["wx"]
                and (ceil is None or ceil >= 5000.0))


def _local_bounds(day: dt.date, hours) -> tuple:
    a = dt.datetime(day.year, day.month, day.day, hours[0], tzinfo=TZ).timestamp()
    b = dt.datetime(day.year, day.month, day.day, hours[1], tzinfo=TZ).timestamp()
    return a, b


def day_evidence(stations: list, metar: dict, day: dt.date):
    """('fog', evidence) / ('clear', evidence) / (None, evidence) for one camera-day from its in-view stations."""
    a, b = _local_bounds(day, FOG_HOURS)
    fog_hits = []
    for st in stations:
        for r in metar.get(st, []):
            if a <= r["t"] < b and is_fog_report(r):
                fog_hits.append({"station": st, "t": r["t"], "vsby": r["vsby"], "wx": r["wx"],
                                 "ceiling_ft": ceiling_ft(r)})
    if fog_hits:
        return "fog", fog_hits
    a, b = _local_bounds(day, CLEAR_HOURS)
    counts = {}
    for st in stations:
        reps = [r for r in metar.get(st, []) if a <= r["t"] < b]
        if len(reps) < 4 or not all(is_clear_report(r) for r in reps):
            return None, []
        counts[st] = len(reps)
    return ("clear", [{"station": st, "reports": n} for st, n in counts.items()]) if counts else (None, [])


def fog_strength(evidence: list) -> float:
    """Ranking score of fog evidence: reports, plus weight for real fog codes and low visibility."""
    s = 0.0
    for e in evidence:
        s += 1.0
        if "FG" in (e.get("wx") or ""):
            s += 1.0
        if e.get("vsby") is not None and e["vsby"] <= 0.5:
            s += 0.5
    return s


def daylight_slots(cam: dict, day: dt.date) -> list:
    """Morning slot start times (unix) of a day at which the sun is at least MIN_SUN_DEG above the horizon."""
    out = []
    for h, m in SLOTS:
        t = dt.datetime(day.year, day.month, day.day, h, m, tzinfo=TZ).timestamp()
        if solar_elevation_deg(cam["lat"], cam["lon"], t) >= MIN_SUN_DEG:
            out.append(t)
    return out


def window_id_for(camera: str, t_unix: float) -> str:
    local = dt.datetime.fromtimestamp(t_unix, TZ)
    return f"hpwren-{camera}-{local:%Y%m%d-%H%M}"


def select_candidates(cams: list, stations: list, metar: dict, days: list, n_fog: int, n_clear: int,
                      max_per_camera: int = 4, max_km: float = 40.0, seed: str = "") -> list:
    """Fog and clear candidate windows (see the module docstring), as candidate rows."""
    fog, clear = defaultdict(list), defaultdict(list)
    for cam in cams:
        sts = stations_in_view(cam, stations, max_km=max_km)
        if not sts:
            continue
        for day in days:
            kind, ev = day_evidence(sts, metar, day)
            if kind is None:
                continue
            slots = daylight_slots(cam, day)[:2]
            if not slots:
                continue
            t = slots[int(_unit_hash(f"{seed}{cam['camera']}{day}") * len(slots))]
            row = {"window_id": window_id_for(cam["camera"], t), "camera": cam["camera"], "site": cam["site"],
                   "day": day.isoformat(), "slot": dt.datetime.fromtimestamp(t, TZ).strftime("%H:%M"),
                   "start_unix_target": int(t), "kind": kind, "split": split_for(cam["camera"], (0.5, 0.15, 0.15, 0.2)),
                   "stations": " ".join(sts), "evidence": json.dumps(ev, separators=(",", ":"))}
            (fog if kind == "fog" else clear)[cam["camera"]].append((fog_strength(ev) if kind == "fog" else 0.0, row))
    for d in fog.values():
        d.sort(key=lambda sr: (-sr[0], sr[1]["day"]))
    for cam_id, d in clear.items():
        d.sort(key=lambda sr: _unit_hash(f"{seed}clear{cam_id}{sr[1]['day']}"))

    def round_robin(pool: dict, n: int, cameras: list) -> list:
        order = sorted(cameras, key=lambda c: _unit_hash(f"{seed}order{c}"))
        out, k = [], 0
        while len(out) < n and k < max_per_camera:
            added = False
            for c in order:
                if k < len(pool.get(c, [])) and len(out) < n:
                    out.append(pool[c][k][1])
                    added = True
            if not added:
                break
            k += 1
        return out

    fog_rows = round_robin(fog, n_fog, list(fog))
    fog_cams = sorted({r["camera"] for r in fog_rows})
    clear_rows = round_robin(clear, n_clear, fog_cams)
    if len(clear_rows) < n_clear:      # not enough clear days on the fog cameras: top up from the others
        rest = {c: v for c, v in clear.items() if c not in fog_cams}
        clear_rows += round_robin(rest, n_clear - len(clear_rows), list(rest))
    return fog_rows + clear_rows


def parse_incidents(items: list) -> list:
    """CAL FIRE incidents with a start time and a location: name, id, start (unix), acres, lat, lon, county."""
    out = []
    for i in items:
        try:
            t = dt.datetime.fromisoformat(str(i["Started"]).replace("Z", "+00:00"))
            lat, lon = float(i["Latitude"]), float(i["Longitude"])
        except (KeyError, TypeError, ValueError):
            continue
        if t.tzinfo is None:
            t = t.replace(tzinfo=dt.timezone.utc)
        if not (math.isfinite(lat) and math.isfinite(lon)) or (lat == 0 and lon == 0):
            continue
        out.append({"name": str(i.get("Name", "")).strip(), "id": str(i.get("UniqueId", "")),
                    "start": t.timestamp(), "acres": float(i.get("AcresBurned") or 0.0), "lat": lat, "lon": lon,
                    "county": i.get("County", "")})
    return sorted(out, key=lambda r: r["start"])


def select_fire_candidates(cams: list, incidents: list, start: dt.date, end: dt.date, max_km: float = 50.0,
                           fov_margin_deg: float = 5.0, min_acres: float = 5.0, max_cameras: int = 4,
                           offsets_min=(-90, 10, 30)) -> list:
    """Windows around reported fire starts (see the module docstring, select-fires), as candidate rows."""
    a = dt.datetime(start.year, start.month, start.day, tzinfo=TZ).timestamp()
    b = (dt.datetime(end.year, end.month, end.day, tzinfo=TZ) + dt.timedelta(days=1)).timestamp()
    rows, seen = [], set()
    for inc in incidents:
        if not (a <= inc["start"] < b) or inc["acres"] < min_acres:
            continue
        hits = []
        for cam in cams:
            d, brg = distance_bearing(cam["lat"], cam["lon"], inc["lat"], inc["lon"])
            off = (brg - cam["azimuth"] + 180.0) % 360.0 - 180.0
            if 1.0 <= d <= max_km and abs(off) <= cam["hfov"] / 2.0 - fov_margin_deg:
                hits.append((d, cam["camera"], cam, off))
        for d, _, cam, off in sorted(hits, key=lambda h: (h[0], h[1]))[:max_cameras]:
            for o in offsets_min:
                t = inc["start"] + 60.0 * o
                if min(solar_elevation_deg(cam["lat"], cam["lon"], t),
                       solar_elevation_deg(cam["lat"], cam["lon"], t + INTERVAL_S * N_STEPS)) < MIN_SUN_DEG:
                    continue
                local = dt.datetime.fromtimestamp(t, TZ)
                wid = f"hpwrenfire-{cam['camera']}-{local:%Y%m%d-%H%M}"
                if wid in seen:
                    continue
                seen.add(wid)
                ev = {"incident": inc["name"], "incident_id": inc["id"], "acres": inc["acres"],
                      "started_unix": int(inc["start"]), "offset_min": o, "distance_km": round(d, 1),
                      "bearing_offset_deg": round(off, 1), "county": inc["county"]}
                rows.append({"window_id": wid, "camera": cam["camera"], "site": cam["site"],
                             "day": local.date().isoformat(), "slot": local.strftime("%H:%M"),
                             "start_unix_target": int(t), "kind": "fire_pre" if o < 0 else "fire_post",
                             "split": split_for(cam["camera"], (0.5, 0.15, 0.15, 0.2)), "stations": "",
                             "evidence": json.dumps(ev, separators=(",", ":")),
                             "zoom_x": f"{min(max(0.5 + off / cam['hfov'], 0.0), 1.0):.3f}"})
    return rows


# =============================================================================
# Frames
# =============================================================================


def fit_grid(times) -> tuple:
    """Least-squares start of a 60 s grid through consecutive frame times and the largest residual (s)."""
    t = np.asarray(times, dtype=np.float64)
    k = np.arange(len(t), dtype=np.float64)
    start = float(np.mean(t - INTERVAL_S * k))
    return start, float(np.max(np.abs(t - (start + INTERVAL_S * k))))


def choose_frames(names: list, target: float, n: int = N_STEPS, max_residual_s: float = 8.0,
                  max_start_offset_s: float = 45.0):
    """n consecutive archive images from the one nearest `target` whose times fit a 60 s grid, or None."""
    times = np.array([int(Path(nm).stem) for nm in names], dtype=np.float64)
    order = np.argsort(times)
    times, names = times[order], [names[i] for i in order]
    if len(times) < n:
        return None
    i0 = int(np.argmin(np.abs(times - target)))
    if abs(times[i0] - target) > max_start_offset_s or i0 + n > len(times):
        return None
    sel = times[i0:i0 + n]
    start, resid = fit_grid(sel)
    if resid > max_residual_s or np.any(np.diff(sel) < 45.0) or np.any(np.diff(sel) > 75.0):
        return None
    return {"names": names[i0:i0 + n], "times": sel.tolist(), "grid_start": start, "residual_s": resid}


def image_list(out: Path, cam: str, day: str, q: int) -> list:
    """The archive's image names for one camera, local day (yyyymmdd) and quarter, cached under _cache/lists."""
    cache = out / "_cache" / "lists" / f"{cam}_{day}_Q{q}.txt"
    if cache.is_file():
        text = cache.read_text()
    else:
        try:
            text = _get(LIST_URL.format(cam=cam, yyyy=day[:4], day=day, q=q)).decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            if exc.code in (403, 404):
                text = ""
            else:
                raise
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(text)
    return [ln.strip() for ln in text.split() if ln.strip().lower().endswith(".jpg")]


def _downscale(img: np.ndarray, max_width: int) -> np.ndarray:
    if img.shape[1] <= max_width:
        return img
    h = int(round(img.shape[0] * max_width / img.shape[1]))
    return cv2.resize(img, (max_width, h), interpolation=cv2.INTER_AREA)


def sheet_id_for(window_id: str) -> str:
    return hashlib.sha1(f"{SHEET_SALT}:{window_id}".encode("utf-8")).hexdigest()[:12]


def _zoom_crop(img: np.ndarray, zoom_x: float) -> np.ndarray:
    """A crop of aspect SHEET_TILE around x = zoom_x (fraction of the width), 0.45 of the height, centred at 0.55."""
    h, w = img.shape[:2]
    ch = int(round(0.45 * h))
    cw = min(w, int(round(ch * SHEET_TILE[0] / SHEET_TILE[1])))
    x0 = int(round(min(max(zoom_x * w - cw / 2, 0), w - cw)))
    y0 = int(round(min(max(0.55 * h - ch / 2, 0), h - ch)))
    return img[y0:y0 + ch, x0:x0 + cw]


def write_sheet(folder: Path, files: list, path: Path, zoom_x=None) -> None:
    """Contact sheet with only minute indices (blind to camera, time, kind): frames SHEET_FRAMES in a 2 x 2
    grid, or, with zoom_x (a region of interest, fraction of the width), frames 0 and 11 above and a zoom
    of the same frames around zoom_x below."""
    if zoom_x in ("", None):
        panels = [(k, None) for k in SHEET_FRAMES]
    else:
        panels = [(0, None), (N_STEPS - 1, None), (0, float(zoom_x)), (N_STEPS - 1, float(zoom_x))]
    tiles = []
    for k, zx in panels:
        img = cv2.imread(str(folder / files[k]), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"cannot read {folder / files[k]}")
        img[:int(round(img.shape[0] * SHEET_BANNER_PX / SHEET_TILE[1]))] = 0   # banner: camera name and time
        if zx is not None:
            img = _zoom_crop(img, zx)
        img = cv2.resize(img, SHEET_TILE, interpolation=cv2.INTER_AREA if zx is None else cv2.INTER_CUBIC)
        img[:SHEET_BANNER_PX] = 0 if zx is None else img[:SHEET_BANNER_PX]
        label = f"t = {k} min" if zx is None else f"zoom t = {k}"
        cv2.rectangle(img, (0, 0), (170, 34), (0, 0, 0), -1)
        cv2.putText(img, label, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        tiles.append(img)
    grid = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), grid, [cv2.IMWRITE_JPEG_QUALITY, 85])


def download_window(out: Path, cand: dict, max_width: int, max_residual_s: float) -> dict:
    """Download one candidate window (idempotent); returns its window.json record or raises ValueError."""
    folder = out / cand["window_id"]
    meta_path = folder / "window.json"
    if meta_path.is_file() and (folder / "timestamps.csv").is_file():
        return json.loads(meta_path.read_text())
    cam, target = cand["camera"], float(cand["start_unix_target"])
    local = dt.datetime.fromtimestamp(target, TZ)
    day = local.strftime("%Y%m%d")
    end_q = quarter(dt.datetime.fromtimestamp(target + INTERVAL_S * N_STEPS + 60, TZ))
    names, where = [], {}
    for q in range(quarter(local), end_q + 1):
        for nm in image_list(out, cam, day, q):
            names.append(nm)
            where[nm] = q
    if not names:
        raise ValueError("no archive images for this camera and day (camera not archived, or outside retention)")
    pick = choose_frames(names, target, max_residual_s=max_residual_s)
    if pick is None:
        raise ValueError("no 12 consecutive images on a 60 s grid near the slot")
    folder.mkdir(parents=True, exist_ok=True)
    files = []
    for nm in pick["names"]:
        dst = folder / nm
        if not dst.is_file():
            data = _get(IMAGE_URL.format(cam=cam, day=day, q=where[nm], name=nm))
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"image {nm} does not decode (archive returned {len(data)} bytes)")
            ok, enc = cv2.imencode(".jpg", _downscale(img, max_width), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if not ok:
                raise ValueError(f"cannot re-encode {nm}")
            tmp = dst.with_suffix(".part")
            tmp.write_bytes(enc.tobytes())
            tmp.replace(dst)
        files.append(nm)
    t0 = pick["times"][0]
    with open(folder / "timestamps.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["filename", "timestamp_s"])
        for nm, t in zip(files, pick["times"]):
            w.writerow([nm, f"{t - t0:.1f}"])
    meta = {"window_id": cand["window_id"], "camera": cam, "camera_mount": camera_mount(cam), "site": cand["site"],
            "day": cand["day"], "slot": cand["slot"], "unix_times": [int(t) for t in pick["times"]],
            "grid_start_s": round(pick["grid_start"] - t0, 2), "grid_residual_s": round(pick["residual_s"], 2),
            "kind_candidate": cand["kind"], "stations": cand["stations"],
            "evidence": json.loads(cand["evidence"]) if cand.get("evidence") else [],
            "sheet_id": sheet_id_for(cand["window_id"]), "files": files, "credit": CREDIT,
            "zoom_x": float(cand["zoom_x"]) if cand.get("zoom_x") not in (None, "") else None,
            "source": IMAGE_URL.format(cam=cam, day=day, q="<q>", name="<unix>.jpg")}
    write_json(meta_path, meta)
    return meta


# =============================================================================
# Stages
# =============================================================================


def _read_csv(path: Path) -> list:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _write_csv(path: Path, rows: list, fields: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def fetch_metar(out: Path, stations: list, start: dt.date, end: dt.date) -> str:
    """METAR CSV for the stations and dates (UTC, one day of margin), cached in <out>/_cache/metar.csv."""
    cache = out / "_cache" / "metar.csv"
    if cache.is_file():
        return cache.read_text()
    a, b = start - dt.timedelta(days=1), end + dt.timedelta(days=2)
    params = [("station", s) for s in stations] + [("data", k) for k in (
        "vsby", "wxcodes", "skyc1", "skyc2", "skyc3", "skyc4", "skyl1", "skyl2", "skyl3", "skyl4")]
    params += [("year1", a.year), ("month1", a.month), ("day1", a.day), ("year2", b.year), ("month2", b.month),
               ("day2", b.day), ("tz", "UTC"), ("format", "onlycomma"), ("latlon", "no"), ("missing", "M"),
               ("trace", "T"), ("direct", "no"), ("report_type", 3), ("report_type", 4)]
    text = _get(METAR_URL + "?" + urllib.parse.urlencode(params), timeout=300).decode("utf-8", "replace")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(text)
    return text


def cmd_select(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "CREDIT.txt").write_text(CREDIT + "\n")
    sites_cache = out / "_cache" / "sites.js"
    if not sites_cache.is_file():
        sites_cache.parent.mkdir(parents=True, exist_ok=True)
        sites_cache.write_bytes(_get(SITES_URL))
    st_cache = out / "_cache" / "stations.geojson"
    if not st_cache.is_file():
        st_cache.write_bytes(_get(STATIONS_URL))
    cams = parse_sites(sites_cache.read_text())
    stations = parse_stations(json.loads(st_cache.read_text()))
    in_view = {c["camera"]: stations_in_view(c, stations, max_km=args.max_km) for c in cams}
    used = sorted({s for v in in_view.values() for s in v})
    start, end = dt.date.fromisoformat(args.start), dt.date.fromisoformat(args.end)
    metar = parse_metar_csv(fetch_metar(out, used, start, end))
    days = [start + dt.timedelta(days=i) for i in range((end - start).days + 1)]
    rows = select_candidates(cams, stations, metar, days, args.n_fog, args.n_clear,
                             max_per_camera=args.max_per_camera, max_km=args.max_km, seed=args.seed)
    _write_csv(out / "candidates.csv", rows, CANDIDATE_FIELDS)
    kinds = defaultdict(int)
    for r in rows:
        kinds[r["kind"]] += 1
    print(json.dumps({"cameras": len(cams), "cameras_with_stations_in_view": sum(bool(v) for v in in_view.values()),
                      "stations": len(used), "candidates": dict(kinds),
                      "candidate_cameras": len({r["camera"] for r in rows})}))
    return 0


def _download_one(out: Path, c: dict, max_width: int, max_residual_s: float):
    try:
        meta = download_window(out, c, max_width, max_residual_s)
        sheet = out / "_sheets" / f"{meta['sheet_id']}.jpg"
        if not sheet.is_file():
            write_sheet(out / meta["window_id"], meta["files"], sheet, meta.get("zoom_x"))
        return meta, None
    except (ValueError, urllib.error.URLError, OSError) as exc:
        return None, {"window_id": c["window_id"], "reason": str(exc)}


def cmd_select_fires(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "CREDIT.txt").write_text(CREDIT + " Fire starts: CAL FIRE incident list (incidents.fire.ca.gov).\n")
    sites_cache = out / "_cache" / "sites.js"
    if not sites_cache.is_file():
        sites_cache.parent.mkdir(parents=True, exist_ok=True)
        sites_cache.write_bytes(_get(SITES_URL))
    start, end = dt.date.fromisoformat(args.start), dt.date.fromisoformat(args.end)
    items = []
    for year in range(start.year, end.year + 1):
        cache = out / "_cache" / f"incidents_{year}.json"
        if not cache.is_file():
            cache.write_bytes(_get(INCIDENTS_URL.format(year=year), timeout=120))
        items += json.loads(cache.read_text())
    cams = parse_sites(sites_cache.read_text())
    rows = select_fire_candidates(cams, parse_incidents(items), start, end, max_km=args.max_km,
                                  min_acres=args.min_acres, max_cameras=args.max_cameras,
                                  offsets_min=tuple(args.offsets_min))
    _write_csv(out / "candidates.csv", rows, CANDIDATE_FIELDS)
    kinds = defaultdict(int)
    for r in rows:
        kinds[r["kind"]] += 1
    print(json.dumps({"incidents": len({json.loads(r['evidence'])['incident_id'] for r in rows}),
                      "candidates": dict(kinds), "cameras": len({r["camera"] for r in rows})}))
    return 0


def cmd_download(args) -> int:
    from concurrent.futures import ThreadPoolExecutor

    out = Path(args.out)
    cands = _read_csv(out / "candidates.csv")
    done, skipped, sheets = [], [], []
    # windows in parallel (each window's frames in order); a few connections only, the archive is shared
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for i, (meta, skip) in enumerate(pool.map(lambda c: _download_one(out, c, args.max_width,
                                                                           args.max_residual_s), cands)):
            if skip is not None:
                skipped.append(skip)
                continue
            sheets.append({"sheet_id": meta["sheet_id"], "window_id": meta["window_id"]})
            done.append(meta["window_id"])
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(cands)} candidates, {len(done)} windows", file=sys.stderr)
    _write_csv(out / "sheets.csv", sorted(sheets, key=lambda r: r["sheet_id"]), ["sheet_id", "window_id"])
    write_json(out / "download_skipped.json", skipped)
    print(json.dumps({"windows": len(done), "skipped": len(skipped)}))
    return 0


def cmd_sheets(args) -> int:
    """(Re)write the contact sheets and sheets.csv of every downloaded window."""
    out = Path(args.out)
    sheets = []
    for meta_path in sorted(out.glob("hpwren*-*/window.json")):
        meta = json.loads(meta_path.read_text())
        sheet = out / "_sheets" / f"{meta['sheet_id']}.jpg"
        if args.force or not sheet.is_file():
            write_sheet(meta_path.parent, meta["files"], sheet, meta.get("zoom_x"))
        sheets.append({"sheet_id": meta["sheet_id"], "window_id": meta["window_id"]})
    _write_csv(out / "sheets.csv", sorted(sheets, key=lambda r: r["sheet_id"]), ["sheet_id", "window_id"])
    print(json.dumps({"sheets": len(sheets)}))
    return 0


VOTE_FIELDS = ["sheet_id", "round", "labeler", "fog", "fog_type", "fog_extent", "smoke", "quality", "confidence",
               "note"]
LABEL_FIELDS = ["sheet_id", "window_id", "fog", "smoke", "quality", "fog_type", "fog_extent", "n_votes",
                "fog_votes", "smoke_votes", "quality_votes"]


def _majority(values, min_count: int):
    """The value given by at least min_count votes (ties impossible for min_count > len / 2), else None."""
    for v in sorted(set(values)):
        if values.count(v) >= min_count:
            return v
    return None


def consolidate_votes(votes: list, primary: str = "fog") -> list:
    """Final labels per sheet from independent votes (see the module docstring, consolidate)."""
    by_sheet = defaultdict(list)
    for v in votes:
        by_sheet[v["sheet_id"]].append(v)
    out = []
    for sid, vs in sorted(by_sheet.items()):
        fogs = [parse_label(v.get("fog")) for v in vs]
        smokes = [parse_label(v.get("smoke")) for v in vs]
        quals = [(v.get("quality") or "ok").strip().lower() for v in vs]
        need = len(vs) // 2 + 1
        def majority(vals):                           # 2 votes: both agree; 3 votes: 2 of 3 ("" = unknown)
            v = _majority(vals, need)
            return v if v in ("0", "1") else ""

        def conservative(vals):                       # 0 only when every vote says 0
            return "0" if vals and all(x == "0" for x in vals) else ("1" if vals.count("1") >= need else "")

        fog = majority(fogs) if primary == "fog" else conservative(fogs)
        smoke = majority(smokes) if primary == "smoke" else conservative(smokes)
        quality = _majority(quals, need) or "disputed"
        types = [v.get("fog_type") or "" for v, f in zip(vs, fogs) if f == "1" and v.get("fog_type")]
        extents = [v.get("fog_extent") or "" for v, f in zip(vs, fogs) if f == "1" and v.get("fog_extent")]
        out.append({"sheet_id": sid, "fog": fog, "smoke": smoke, "quality": quality,
                    "fog_type": max(set(types), key=types.count) if fog == "1" and types else "",
                    "fog_extent": max(set(extents), key=extents.count) if fog == "1" and extents else "",
                    "n_votes": len(vs), "fog_votes": " ".join(f or "?" for f in fogs),
                    "smoke_votes": " ".join(s or "?" for s in smokes), "quality_votes": " ".join(quals)})
    return out


def needs_third_vote(votes_two: list, primary: str = "fog") -> bool:
    """A sheet needs a third vote when its two votes do not agree on a definite primary label and quality."""
    labs = [parse_label(v.get(primary)) for v in votes_two]
    quals = [(v.get("quality") or "ok").strip().lower() for v in votes_two]
    return not (len(labs) == 2 and labs[0] == labs[1] != "" and quals[0] == quals[1])


def cohen_kappa(a: list, b: list):
    """Cohen's kappa of two raters' labels over items both labelled (None when undefined)."""
    pairs = [(x, y) for x, y in zip(a, b) if x != "" and y != ""]
    if not pairs:
        return None
    cats = sorted({c for p in pairs for c in p})
    n = len(pairs)
    po = sum(x == y for x, y in pairs) / n
    pe = sum((sum(x == c for x, _ in pairs) / n) * (sum(y == c for _, y in pairs) / n) for c in cats)
    return None if pe >= 1.0 else (po - pe) / (1.0 - pe)


def cmd_consolidate(args) -> int:
    out = Path(args.out)
    votes = _read_csv(Path(args.votes))
    by_sheet = {r["sheet_id"]: r["window_id"] for r in _read_csv(out / "sheets.csv")}
    labels = consolidate_votes(votes, args.primary)
    for lab in labels:
        lab["window_id"] = by_sheet.get(lab["sheet_id"], "")
    _write_csv(Path(args.labels), labels, LABEL_FIELDS)
    r1 = {v["sheet_id"]: parse_label(v[args.primary]) for v in votes if str(v.get("round")) == "1"}
    r2 = {v["sheet_id"]: parse_label(v[args.primary]) for v in votes if str(v.get("round")) == "2"}
    common = sorted(set(r1) & set(r2))
    both = [(r1[s], r2[s]) for s in common if r1[s] != "" and r2[s] != ""]
    stats = {"sheets": len(labels), "fog": {k: sum(lab["fog"] == k for lab in labels) for k in ("1", "0", "")},
             "smoke": {k: sum(lab["smoke"] == k for lab in labels) for k in ("1", "0", "")},
             "quality_ok": sum(lab["quality"] == "ok" for lab in labels),
             "rounds_1_2": {"label": args.primary, "sheets": len(common), "both_definite": len(both),
                            f"{args.primary}_agreement": (sum(x == y for x, y in both) / len(both)) if both else None,
                            f"{args.primary}_cohen_kappa": cohen_kappa([r1[s] for s in common],
                                                                       [r2[s] for s in common])}}
    write_json(Path(args.labels).with_suffix(".stats.json"), stats)
    print(json.dumps(stats))
    return 0


def parse_label(v):
    v = (v or "").strip().lower()
    if v in ("", "unknown", "nan", "?"):
        return ""
    if v in ("0", "1"):
        return v
    raise ValueError(f"label must be 0, 1 or empty, got {v!r}")


def cmd_manifest(args) -> int:
    out = Path(args.out)
    labels = _read_csv(Path(args.labels))
    by_sheet = {r["sheet_id"]: r["window_id"] for r in _read_csv(out / "sheets.csv")}
    rows, dropped = [], defaultdict(int)
    for lab in labels:
        wid = lab.get("window_id") or by_sheet.get(lab.get("sheet_id", ""))
        if not wid:
            dropped["unknown_window"] += 1
            continue
        meta_path = out / wid / "window.json"
        if not meta_path.is_file():
            dropped["not_downloaded"] += 1
            continue
        fog, smoke = parse_label(lab.get("fog")), parse_label(lab.get("smoke"))
        if (lab.get("quality") or "ok").strip().lower() not in ("ok", ""):
            dropped["quality"] += 1
            continue
        if (fog if args.require == "fog" else smoke) == "":
            dropped[f"{args.require}_unknown"] += 1
            continue
        meta = json.loads(meta_path.read_text())
        cam = meta["camera"]
        rows.append({"video_path": str(out / wid), "camera_id": cam, "video_id": wid, "smoke": smoke, "fog": fog,
                     "split": split_for(cam, tuple(args.fractions)), "start_s": f"{meta['grid_start_s']:.2f}",
                     "max_windows": 1, "window_id": wid, "camera_mount": camera_mount(cam),
                     "source": args.source, "kind_candidate": meta.get("kind_candidate", "")})
    fields = ["video_path", "camera_id", "video_id", "smoke", "fog", "split", "start_s", "max_windows",
              "window_id", "camera_mount", "source", "kind_candidate"]
    _write_csv(out / "real.csv", rows, fields)
    per = defaultdict(int)
    for r in rows:
        per[(r["split"], r["fog"], r["smoke"] or "?")] += 1
    print(json.dumps({"rows": len(rows), "dropped": dict(dropped),
                      "split_fog_smoke": {f"{s}/fog{f}/smoke{m}": n for (s, f, m), n in sorted(per.items())}}))
    return 0


def cmd_merge(args) -> int:
    rows, fields, seen = [], [], set()
    for path in args.manifests:
        for r in _read_csv(Path(path)):
            wid = r.get("window_id") or r.get("video_id")
            if wid in seen:
                raise SystemExit(f"duplicate window id {wid!r} in {path}")
            seen.add(wid)
            rows.append(r)
            for k in r:
                if k not in fields:
                    fields.append(k)
    _write_csv(Path(args.out), rows, fields)
    print(json.dumps({"rows": len(rows), "manifests": len(args.manifests)}))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select", help="candidate windows from METAR evidence")
    s.add_argument("--out", required=True)
    s.add_argument("--start", required=True, help="first local day, YYYY-MM-DD (inside the archive's retention)")
    s.add_argument("--end", required=True, help="last local day, YYYY-MM-DD")
    s.add_argument("--n-fog", type=int, default=200)
    s.add_argument("--n-clear", type=int, default=200)
    s.add_argument("--max-per-camera", type=int, default=4)
    s.add_argument("--max-km", type=float, default=40.0)
    s.add_argument("--seed", default="")
    f = sub.add_parser("select-fires", help="candidate windows around CAL FIRE incident starts")
    f.add_argument("--out", required=True)
    f.add_argument("--start", required=True)
    f.add_argument("--end", required=True)
    f.add_argument("--max-km", type=float, default=50.0)
    f.add_argument("--min-acres", type=float, default=5.0)
    f.add_argument("--max-cameras", type=int, default=4)
    f.add_argument("--offsets-min", type=int, nargs="+", default=[-90, 10, 30],
                   help="window starts relative to the reported start (minutes); negative = before (smoke-free)")
    d = sub.add_parser("download", help="download candidate windows and write contact sheets")
    d.add_argument("--out", required=True)
    d.add_argument("--max-width", type=int, default=1024)
    d.add_argument("--max-residual-s", type=float, default=8.0)
    d.add_argument("--workers", type=int, default=4, help="windows downloaded in parallel")
    t = sub.add_parser("sheets", help="(re)write the contact sheets of the downloaded windows")
    t.add_argument("--out", required=True)
    t.add_argument("--force", action="store_true", help="rewrite existing sheets")
    m = sub.add_parser("manifest", help="labels -> real.csv for extract_features.py --real-manifest")
    m.add_argument("--out", required=True)
    m.add_argument("--labels", required=True)
    m.add_argument("--fractions", type=float, nargs=4, default=[0.5, 0.15, 0.15, 0.2],
                   metavar=("TRAIN", "VAL_SELECT", "VAL_CALIB", "TEST_IID"),
                   help="must match prepare_figlib.py --fractions so a camera gets the same split")
    m.add_argument("--require", choices=("fog", "smoke"), default="fog", help="label a row must have")
    m.add_argument("--source", default="hpwren_fog")
    c = sub.add_parser("consolidate", help="votes -> one label per sheet (+ agreement statistics)")
    c.add_argument("--out", required=True)
    c.add_argument("--votes", required=True)
    c.add_argument("--labels", required=True)
    c.add_argument("--primary", choices=("fog", "smoke"), default="fog",
                   help="label decided by majority (the other: 0 only when every vote says 0)")
    g = sub.add_parser("merge", help="concatenate manifests")
    g.add_argument("--out", required=True)
    g.add_argument("manifests", nargs="+")
    args = ap.parse_args(argv)
    return {"select": cmd_select, "select-fires": cmd_select_fires, "download": cmd_download, "sheets": cmd_sheets,
            "consolidate": cmd_consolidate, "manifest": cmd_manifest, "merge": cmd_merge}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
