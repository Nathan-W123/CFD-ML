"""HPWREN fog set preparation (scripts/prepare_hpwren_fog.py), without the network.

Camera catalogue parsing and the in-view station geometry; METAR fog / clear
rules; sun elevation; candidate selection (one window per camera-day,
daylight slots only, clear windows on the fog cameras); 12 consecutive
archive frames on a 60 s grid; blind contact sheets; the manifest (labels by
window or sheet id, unknown fog dropped, split equal to prepare_figlib's for
the same camera) and the merge (duplicate ids refused).
"""

import csv
import datetime as dt
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import prepare_figlib as F  # noqa: E402
import prepare_hpwren_fog as H  # noqa: E402

SITES_JS = """var sites = {
  "om": {"name": "Otay Mountain", "lat": 32.6, "long": -116.85, "elev": 1087, "cams": {
    "om-w-mobo-c": {"imager": "color", "azimuth": 270, "horizontalView": 90, "active": "y", "experimental": "n"},
    "om-w-mobo-m": {"imager": "monochrome", "azimuth": 270, "horizontalView": 90, "active": "y"},
    "om-e-mobo-c": {"imager": "color", "azimuth": 90, "horizontalView": 90, "active": "y"},
    "om-n-mobo-c": {"imager": "color", "azimuth": 0, "horizontalView": 90, "active": "n"}}}
};"""
STATIONS = [{"station": "WST", "name": "west", "lat": 32.6, "lon": -117.0},     # ~14 km west
            {"station": "EST", "name": "east", "lat": 32.6, "lon": -116.7},     # ~14 km east
            {"station": "FAR", "name": "far", "lat": 32.6, "lon": -117.6}]      # ~70 km west


def _metar(rows):
    head = "station,valid,vsby,wxcodes,skyc1,skyc2,skyc3,skyc4,skyl1,skyl2,skyl3,skyl4\n"
    return head + "\n".join(rows) + "\n"


def _utc(day, hh, mm=53):
    """UTC 'YYYY-MM-DD HH:MM' of a local (PDT, UTC-7) hour on `day`."""
    t = dt.datetime.combine(day, dt.time(hh, mm), tzinfo=H.TZ).astimezone(dt.timezone.utc)
    return t.strftime("%Y-%m-%d %H:%M")


def test_parse_sites_keeps_active_colour_cameras_only():
    cams = H.parse_sites(SITES_JS)
    assert [c["camera"] for c in cams] == ["om-e-mobo-c", "om-w-mobo-c"]
    w = cams[1]
    assert w["azimuth"] == 270 and w["hfov"] == 90 and w["lat"] == 32.6 and w["lon"] == -116.85


def test_stations_in_view_uses_distance_and_field_of_view():
    cams = {c["camera"]: c for c in H.parse_sites(SITES_JS)}
    assert H.stations_in_view(cams["om-w-mobo-c"], STATIONS, max_km=40) == ["WST"]
    assert H.stations_in_view(cams["om-e-mobo-c"], STATIONS, max_km=40) == ["EST"]
    assert H.stations_in_view(cams["om-w-mobo-c"], STATIONS, max_km=100) == ["WST", "FAR"]
    d, b = H.distance_bearing(32.6, -116.85, 32.6, -117.0)
    assert 13 < d < 15 and abs(b - 270) < 1


def test_metar_fog_and_clear_rules():
    day = dt.date(2026, 8, 24)
    text = _metar([
        f"WST,{_utc(day, 5)},0.25,FG,VV,M,M,M,200,M,M,M",          # fog code + low visibility
        f"WST,{_utc(day, 11)},0.25,FG,VV,M,M,M,200,M,M,M",         # outside the 03-10 local fog window
        f"EST,{_utc(day, 4)},10.00,M,CLR,M,M,M,M,M,M,M",
        f"EST,{_utc(day, 5)},10.00,M,FEW,M,M,M,9000,M,M,M",
        f"EST,{_utc(day, 7)},10.00,M,SCT,BKN,M,M,3000,8000,M,M",
        f"EST,{_utc(day, 9)},10.00,M,CLR,M,M,M,M,M,M,M",
    ])
    m = H.parse_metar_csv(text)
    assert H.is_fog_report(m["WST"][0]) and H.ceiling_ft(m["WST"][0]) == 200
    kind, ev = H.day_evidence(["WST"], m, day)
    assert kind == "fog" and len(ev) == 1 and ev[0]["wx"] == "FG"
    assert H.day_evidence(["EST"], m, day) == ("clear", [{"station": "EST", "reports": 4}])
    # one report with a ceiling below 5000 ft makes the day not clear; fewer than 4 reports is not clear either
    m2 = H.parse_metar_csv(text.replace("SCT,BKN,M,M,3000,8000", "BKN,M,M,M,3000,M"))
    assert H.day_evidence(["EST"], m2, day) == (None, [])
    m3 = H.parse_metar_csv(_metar([f"EST,{_utc(day, 4)},10.00,M,CLR,M,M,M,M,M,M,M"]))
    assert H.day_evidence(["EST"], m3, day) == (None, [])
    # mist only with visibility <= 3 mi; low stratus (OVC 800 ft) counts as fog evidence
    r = {"vsby": 5.0, "wx": "BR", "layers": []}
    assert not H.is_fog_report(r) and H.is_fog_report({**r, "vsby": 2.5})
    assert H.is_fog_report({"vsby": 10.0, "wx": "", "layers": [("OVC", 800.0)]})
    assert not H.is_fog_report({"vsby": 10.0, "wx": "", "layers": [("OVC", 1500.0)]})
    assert H.is_fog_report({"vsby": 6.0, "wx": "BCFG", "layers": []})


def test_solar_elevation_and_daylight_slots():
    noon = dt.datetime(2026, 8, 24, 12, 45, tzinfo=H.TZ).timestamp()
    midnight = dt.datetime(2026, 8, 24, 0, 0, tzinfo=H.TZ).timestamp()
    assert 60 < H.solar_elevation_deg(32.7, -117.0, noon) < 72          # late-August noon in San Diego
    assert H.solar_elevation_deg(32.7, -117.0, midnight) < -30
    cam = {"lat": 32.7, "lon": -117.0}
    aug = [dt.datetime.fromtimestamp(t, H.TZ).strftime("%H:%M") for t in H.daylight_slots(cam, dt.date(2026, 8, 24))]
    octo = [dt.datetime.fromtimestamp(t, H.TZ).strftime("%H:%M") for t in H.daylight_slots(cam, dt.date(2026, 10, 18))]
    assert aug[0] in ("06:30", "07:00") and octo[0] in ("07:30", "08:00") and aug[-1] == octo[-1] == "10:00"


def test_select_candidates_matches_clear_windows_to_fog_cameras():
    cams = H.parse_sites(SITES_JS)
    days = [dt.date(2026, 8, 20) + dt.timedelta(days=i) for i in range(6)]
    rows = []
    for i, day in enumerate(days):
        if i < 3:   # fog mornings at the west station, clear days at the east one
            rows.append(f"WST,{_utc(day, 6)},0.50,FG,VV,M,M,M,200,M,M,M")
        else:
            rows += [f"WST,{_utc(day, h)},10.00,M,CLR,M,M,M,M,M,M,M" for h in (4, 6, 8, 10)]
        rows += [f"EST,{_utc(day, h)},10.00,M,CLR,M,M,M,M,M,M,M" for h in (4, 6, 8, 10)]
    metar = H.parse_metar_csv(_metar(rows))
    out = H.select_candidates(cams, STATIONS, metar, days, n_fog=10, n_clear=10, max_per_camera=2)
    fog = [r for r in out if r["kind"] == "fog"]
    clear = [r for r in out if r["kind"] == "clear"]
    assert len(fog) == 2 and {r["camera"] for r in fog} == {"om-w-mobo-c"}            # capped per camera
    assert [r["camera"] for r in clear[:2]] == ["om-w-mobo-c", "om-w-mobo-c"]          # fog camera first
    assert len({(r["camera"], r["day"]) for r in out}) == len(out)                     # one window per camera-day
    for r in out:
        t = float(r["start_unix_target"])
        assert H.solar_elevation_deg(32.6, -116.85, t) >= H.MIN_SUN_DEG
        assert r["window_id"] == H.window_id_for(r["camera"], t)
        assert r["split"] == F.split_for(r["camera"], (0.5, 0.15, 0.15, 0.2))
    assert H.select_candidates(cams, STATIONS, metar, days, 10, 10, max_per_camera=2) == out   # deterministic


def test_choose_frames_fits_a_60s_grid():
    t0 = 1_790_000_000
    jit = [0, 2, -1, 3, 0, -2, 1, 0, 2, -3, 1, 0, 2, 0]
    names = [f"{t0 + 60 * k + j}.jpg" for k, j in enumerate(jit)][::-1]          # any order
    pick = H.choose_frames(names, t0 + 5)
    assert pick is not None and len(pick["names"]) == 12 and pick["names"][0] == f"{t0}.jpg"
    assert pick["residual_s"] <= 4 and abs(pick["grid_start"] - t0) < 2
    # a gap (a missing frame) or a slot far from every frame gives nothing
    gap = [n for i, n in enumerate(sorted(names)) if i != 5]
    assert H.choose_frames(gap, t0) is None
    assert H.choose_frames(names, t0 - 600) is None
    # drift beyond the residual limit is refused
    drift = [f"{t0 + 63 * k}.jpg" for k in range(14)]
    assert H.choose_frames(drift, t0, max_residual_s=8) is None
    start, resid = H.fit_grid([t0 + 60 * k for k in range(12)])
    assert start == t0 and resid == 0


def test_contact_sheet_is_blind_and_ids_are_stable(tmp_path):
    folder = tmp_path / "w"
    folder.mkdir()
    files = []
    for k in range(12):
        img = np.full((200, 300, 3), 20 * k, np.uint8)
        f = f"{1_790_000_000 + 60 * k}.jpg"
        cv2.imwrite(str(folder / f), img)
        files.append(f)
    H.write_sheet(folder, files, tmp_path / "s.jpg")
    sheet = cv2.imread(str(tmp_path / "s.jpg"))
    assert sheet.shape[:2] == (2 * H.SHEET_TILE[1], 2 * H.SHEET_TILE[0])
    # the banner rows of every tile are blank (beyond the minute label box)
    for r0 in (0, H.SHEET_TILE[1]):
        for c0 in (0, H.SHEET_TILE[0]):
            assert sheet[r0:r0 + H.SHEET_BANNER_PX - 8, c0 + 180:c0 + H.SHEET_TILE[0]].max() < 8   # clear of the
            # JPEG block that straddles the blank banner and the image
    # with a region of interest: frames 0 and 11 above, zooms of them around zoom_x below
    H.write_sheet(folder, files, tmp_path / "z.jpg", zoom_x=0.9)
    z = cv2.imread(str(tmp_path / "z.jpg"))
    assert z.shape == sheet.shape
    top, bottom = z[100:400, 200:700], z[H.SHEET_TILE[1] + 100:H.SHEET_TILE[1] + 400, 200:700]
    assert abs(float(top.mean()) - float(bottom.mean())) < 4          # same frame (flat test images)
    assert abs(float(z[100:400, H.SHEET_TILE[0] + 200:].mean()) - 220) < 6   # frame 11 on the right
    img = np.zeros((100, 400, 3), np.uint8)
    img[:, 300:] = 255
    crop = H._zoom_crop(img, 0.95)
    assert crop.shape[0] == 45 and crop.shape[1] == round(45 * 1.5) and crop[:, -1].min() == 255
    sid = H.sheet_id_for("hpwren-om-w-mobo-c-20260824-0700")
    assert sid == H.sheet_id_for("hpwren-om-w-mobo-c-20260824-0700") and len(sid) == 12
    assert "om-w" not in sid and sid != H.sheet_id_for("hpwren-om-w-mobo-c-20260825-0700")


def _fake_window(out: Path, wid: str, cam: str, kind: str):
    folder = out / wid
    folder.mkdir(parents=True)
    meta = {"window_id": wid, "camera": cam, "grid_start_s": -0.5, "kind_candidate": kind,
            "sheet_id": H.sheet_id_for(wid), "files": []}
    (folder / "window.json").write_text(json.dumps(meta))
    (folder / "timestamps.csv").write_text("filename,timestamp_s\n")
    return meta


def test_manifest_and_merge(tmp_path):
    out = tmp_path / "hf"
    metas = [_fake_window(out, f"hpwren-om-w-mobo-c-2026082{i}-0700", "om-w-mobo-c", "fog") for i in range(4)]
    with open(out / "sheets.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sheet_id", "window_id"])
        for m in metas:
            w.writerow([m["sheet_id"], m["window_id"]])
    labels = tmp_path / "labels.csv"
    labels.write_text("sheet_id,window_id,fog,smoke,quality\n"
                      f"{metas[0]['sheet_id']},,1,0,ok\n"          # by sheet id
                      f",{metas[1]['window_id']},0,0,ok\n"         # by window id
                      f",{metas[2]['window_id']},,0,ok\n"          # fog unknown: dropped
                      f",{metas[3]['window_id']},1,,dark\n")       # unusable quality: dropped
    assert H.main(["manifest", "--out", str(out), "--labels", str(labels)]) == 0
    with open(out / "real.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert [(r["window_id"], r["fog"], r["smoke"]) for r in rows] == [
        (metas[0]["window_id"], "1", "0"), (metas[1]["window_id"], "0", "0")]
    for r in rows:
        assert r["split"] == F.split_for("om-w-mobo-c", (0.5, 0.15, 0.15, 0.2))
        assert r["camera_mount"] == "om-w" and r["start_s"] == "-0.50" and r["max_windows"] == "1"
        assert Path(r["video_path"]).name == r["window_id"]
    # merge with a FIgLib-style manifest: union of columns, duplicate ids refused
    fig = tmp_path / "fig.csv"
    fig.write_text("video_path,camera_id,video_id,smoke,fog,split,start_s,max_windows,window_id,"
                   "plume_offset_first_s\nx,om-w-mobo-c,v,1,,train,0,1,v-post0,60\n")
    merged = tmp_path / "all.csv"
    assert H.main(["merge", "--out", str(merged), str(fig), str(out / "real.csv")]) == 0
    with open(merged, newline="") as fh:
        rd = csv.DictReader(fh)
        all_rows = list(rd)
        assert "plume_offset_first_s" in rd.fieldnames and "source" in rd.fieldnames
    assert len(all_rows) == 3
    with pytest.raises(SystemExit):
        H.main(["merge", "--out", str(tmp_path / "dup.csv"), str(fig), str(fig)])
    with pytest.raises(ValueError):
        H.parse_label("2")


def test_consolidate_votes_and_agreement(tmp_path):
    v = lambda sid, rnd, fog, smoke="0", q="ok", ft="": {"sheet_id": sid, "round": rnd, "labeler": f"L{rnd}",
                                                         "fog": fog, "smoke": smoke, "quality": q, "fog_type": ft}
    votes = [v("a", 1, "1", ft="below"), v("a", 2, "1", ft="below"),         # agree: fog
             v("b", 1, "0"), v("b", 2, "0"),                                 # agree: clear
             v("c", 1, "1"), v("c", 2, "0"), v("c", 3, "1", ft="bank"),       # third vote decides
             v("d", 1, "1"), v("d", 2, ""), v("d", 3, ""),                   # 2 of 3 unknown: unknown
             v("e", 1, "0", smoke=""), v("e", 2, "0"),                       # smoke not all 0: unknown
             v("f", 1, "0", q="dark"), v("f", 2, "0", q="dark")]             # agreed bad quality
    assert not H.needs_third_vote(votes[0:2]) and H.needs_third_vote(votes[4:6]) and H.needs_third_vote(votes[7:9])
    assert H.needs_third_vote([v("x", 1, "0"), v("x", 2, "0", q="glare")])
    lab = {r["sheet_id"]: r for r in H.consolidate_votes(votes)}
    assert (lab["a"]["fog"], lab["a"]["fog_type"], lab["a"]["smoke"]) == ("1", "below", "0")
    assert lab["b"]["fog"] == "0" and lab["c"]["fog"] == "1" and lab["c"]["n_votes"] == 3
    assert lab["d"]["fog"] == "" and lab["e"]["smoke"] == "" and lab["e"]["fog"] == "0"
    assert lab["f"]["quality"] == "dark" and lab["c"]["fog_votes"] == "1 0 1"
    assert H.cohen_kappa(["1", "0", "1", "0"], ["1", "0", "1", "0"]) == 1.0
    assert H.cohen_kappa(["1", "0"], ["0", "1"]) == -1.0 and H.cohen_kappa(["", ""], ["1", "0"]) is None
    # the command writes labels with window ids and the statistics next to them
    out = tmp_path / "hf"
    out.mkdir()
    (out / "sheets.csv").write_text("sheet_id,window_id\n" + "".join(f"{s},w{s}\n" for s in "abcdef"))
    vf = tmp_path / "votes.csv"
    with open(vf, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=H.VOTE_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(votes)
    assert H.main(["consolidate", "--out", str(out), "--votes", str(vf), "--labels", str(tmp_path / "lab.csv")]) == 0
    stats = json.loads((tmp_path / "lab.stats.json").read_text())
    assert stats["sheets"] == 6 and stats["fog"]["1"] == 2 and stats["rounds_1_2"]["sheets"] == 6
    with open(tmp_path / "lab.csv", newline="") as fh:
        assert {r["sheet_id"]: r["window_id"] for r in csv.DictReader(fh)}["a"] == "wa"
