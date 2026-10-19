"""FIgLib preparation (scripts/prepare_figlib.py), without the network.

Window planning on a FIgLib-style record (one frame per minute, offsets
relative to the first visible plume): windows never straddle offset 0 and are
labelled pre (smoke 0) / post (smoke 1); camera ids and mounts from sequence
names; tar members with absolute paths, ".." or links are refused; the
split-aware choice fills every split for a small N; a short download keeps
its .part and resumes; a truncated archive is deleted and listed as skipped.
"""

import csv
import io
import json
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import prepare_figlib as P  # noqa: E402

FRACTIONS = [0.5, 0.15, 0.15, 0.2]


def _record(offsets, jitter_s=0.0, seed=0, name="20220302-Jimfire-0921-stgo-s-mobo-c"):
    """sequence.json-style record: frames at the given plume offsets (s), times relative to the first frame."""
    rng = np.random.default_rng(seed)
    offsets = np.asarray(offsets, dtype=np.int64)
    t = (offsets - offsets[0]).astype(np.float64) + jitter_s * rng.uniform(-1, 1, len(offsets))
    t[0] = 0.0
    cam = P.camera_id(name)
    return {"name": name, "camera_id": cam, "mount": P.camera_mount(cam),
            "frames": [{"timestamp_s": float(ti), "offset_s": int(o)} for ti, o in zip(t, offsets)]}


def _args(**kw):
    base = dict(interval_s=60.0, max_offset_s=10.0, pre_windows=1, post_windows=1, fractions=FRACTIONS,
                zero_shot=False)
    base.update(kw)
    return SimpleNamespace(**base)


# ============================================================================= windows
@pytest.mark.parametrize("jitter_s, drop", [(0.0, ()), (4.0, ()), (0.0, (-300, 420))])
def test_plan_windows_never_straddle_the_plume_onset(jitter_s, drop):
    offsets = [o for o in range(-3000, 3001, 60) if o not in drop]
    rec = _record(offsets, jitter_s=jitter_s)
    off = np.array([f["offset_s"] for f in rec["frames"]])
    ts = np.array([f["timestamp_s"] for f in rec["frames"]])
    wins = P.plan_windows(rec, 60.0, 10.0, n_pre=3, n_post=3)
    pre = [w for w in wins if w[0] == "pre"]
    post = [w for w in wins if w[0] == "post"]
    assert len(pre) == 3 and len(post) == 3
    for kind, k, start, first, last in wins:
        win = P.select_window(ts, n=P.N_STEPS, interval_s=60.0, start=start, max_offset_s=10.0, flow_lag_s=None)
        sel = off[win["obs_idx"]]
        assert win["lag_idx"] is None and len(sel) == P.N_STEPS
        assert (first, last) == (sel[0], sel[-1])
        if kind == "pre":
            assert np.all(sel < 0), sel                     # wholly before the first visible plume
        else:
            assert np.all(sel >= 60), sel                   # starts at or after +60 s
    # latest pre window first, earliest post window first, none overlapping
    assert [w[1] for w in pre] == [0, 1, 2] and [w[1] for w in post] == [0, 1, 2]
    assert all(a[3] > b[4] for a, b in zip(pre, pre[1:]))
    assert all(b[3] > a[4] for a, b in zip(post, post[1:]))
    if not drop:
        assert (pre[0][4], post[0][3]) == (-60, 60)
    else:                                                    # windows over a missing frame are not listed
        assert (pre[0][4], post[0][3]) == (-360, 480)

    rows = P.manifest_rows(rec, Path("seq"), _args(pre_windows=3, post_windows=3))
    assert [r["smoke"] for r in rows] == [0, 0, 0, 1, 1, 1]
    assert all(r["fog"] == "" for r in rows)                 # fog unknown: FIgLib does not annotate it
    assert all((r["plume_offset_last_s"] < 0) if r["smoke"] == 0 else (r["plume_offset_first_s"] >= 60)
               for r in rows)
    assert len({r["window_id"] for r in rows}) == len(rows)
    assert {r["camera_id"] for r in rows} == {"stgo-s-mobo-c"} and {r["camera_mount"] for r in rows} == {"stgo-s"}
    assert {r["split"] for r in rows} == {P.split_for("stgo-s-mobo-c", FRACTIONS)}
    assert {r["split"] for r in P.manifest_rows(rec, Path("seq"), _args(zero_shot=True))} == {"test_iid"}


def test_plan_windows_skips_a_side_without_room():
    rec = _record(range(-300, 1201, 60))                     # only 5 frames before the plume
    wins = P.plan_windows(rec, 60.0, 10.0, n_pre=1, n_post=1)
    assert [w[0] for w in wins] == ["post"]


# ============================================================================= names, members, choice
def test_camera_id_and_mount():
    assert P.camera_id("20220302-Jimfire-0921-stgo-s-mobo-c") == "stgo-s-mobo-c"
    assert P.camera_id("20160604_FIRE_rm-n-mobo-c") == "rm-n-mobo-c"
    assert P.camera_id("20190610_FIRE_smer-tcs3-mobo-m") == "smer-tcs3-mobo-m"
    assert P.camera_id("20200822_BrattonFire_lp-e-iqeye") == "lp-e-iqeye"
    assert P.camera_mount("stgo-s-mobo-c") == P.camera_mount("stgo-s-mobo-m") == "stgo-s"
    assert P.split_for("stgo-s-mobo-c", FRACTIONS) == P.split_for("stgo-s-mobo-m", FRACTIONS)
    with pytest.raises(ValueError):
        P.camera_id("20220302_no_camera_here")


def test_safe_member_refuses_absolute_parent_and_link_paths():
    def info(name, kind=tarfile.REGTYPE):
        ti = tarfile.TarInfo(name)
        ti.type = kind
        return ti

    assert P._safe_member(info("seq/1646200000_-60.jpg")) is True
    for bad in ("/etc/passwd", "seq/../../outside.jpg", "../x.jpg", "C:/x.jpg"):
        with pytest.raises(ValueError):
            P._safe_member(info(bad))
    for kind in (tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.DIRTYPE, tarfile.CHRTYPE, tarfile.FIFOTYPE):
        assert P._safe_member(info("seq/1646200000_0.jpg", kind)) is False


SITES = ("rm", "sm", "lp", "hp", "mw", "om", "pi", "wc", "bl", "ml", "sp", "dl", "cp", "mg", "so", "stgo", "smer",
         "lo", "tp", "bh", "vo", "sy", "pv", "ow")


def _names(per_camera=2):
    """FIgLib-style names: <date>_FIRE_<site>-<direction>-<model>, two models per mount, oldest first."""
    names = []
    for s_i, site in enumerate(SITES):
        for d_i, direction in enumerate(("n", "e", "s", "w", "ne", "nw", "se", "sw")):
            for model in ("mobo-c", "mobo-m"):
                for k in range(per_camera):
                    names.append(f"20{16 + k:02d}{s_i % 12 + 1:02d}{d_i + 1:02d}_FIRE_{site}-{direction}-{model}")
    return names


def test_split_quotas():
    assert P.split_quotas(4, FRACTIONS) == [1, 1, 1, 1]
    assert P.split_quotas(10, FRACTIONS) == [5, 2, 1, 2]
    assert P.split_quotas(40, FRACTIONS) == [20, 6, 6, 8]
    assert sum(P.split_quotas(2, FRACTIONS)) == 2
    assert P.split_quotas(5, [1, 0, 0, 1]) == [3, 0, 0, 2]
    for n in range(0, 30):
        q = P.split_quotas(n, FRACTIONS)
        assert sum(q) == n and (n < 4 or min(q) >= 1)


@pytest.mark.parametrize("n", [4, 5, 8, 10])
def test_choose_fills_every_split_for_a_small_n(n):
    names = _names()
    picked = P.choose(names, n, FRACTIONS)
    assert len(picked) == len(set(picked)) == n
    splits = [P.split_for(P.camera_id(x), FRACTIONS) for x in picked]
    counts = {s: splits.count(s) for s in P.SPLIT_NAMES}
    assert counts == dict(zip(P.SPLIT_NAMES, P.split_quotas(n, FRACTIONS)))
    assert all(v >= 1 for v in counts.values())
    # within a split, spread over cameras before a second sequence of one camera
    for s in P.SPLIT_NAMES:
        cams = [P.camera_id(x) for x, sp in zip(picked, splits) if sp == s]
        assert len(set(cams)) == len(cams)
    assert P.choose(names, n, FRACTIONS) == picked                       # deterministic
    # zero-shot (or no fractions): the round robin over every camera
    rr = P.choose(names, n, FRACTIONS, zero_shot=True)
    assert rr == P.choose(names, n) and len(set(P.camera_id(x) for x in rr)) == n


def test_choose_hands_a_short_splits_share_to_the_others():
    names = [x for x in _names() if P.split_for(P.camera_id(x), FRACTIONS) != "val_calib"]
    picked = P.choose(names, 8, FRACTIONS)
    assert len(picked) == 8
    splits = {P.split_for(P.camera_id(x), FRACTIONS) for x in picked}
    assert splits == {"train", "val_select", "test_iid"}
    one = [x for x in _names() if P.split_for(P.camera_id(x), FRACTIONS) == "test_iid"][:2]
    assert sorted(P.choose(one, 4, FRACTIONS)) == sorted(one)          # fewer names than asked: all of them


# ============================================================================= download and archives
class _Resp:
    def __init__(self, status, body, headers):
        self.status, self._body, self.headers = status, io.BytesIO(body), headers

    def read(self, n=-1):
        return self._body.read(n)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_download_keeps_a_short_transfer_and_resumes(tmp_path, monkeypatch):
    data = bytes(range(256)) * 4                                    # 1024 bytes
    calls = []

    def serve(responses):
        def urlopen(req, timeout=None):
            calls.append(dict(req.header_items()))
            return responses.pop(0)
        return urlopen

    dest = tmp_path / "dl" / "seq.tgz"
    part = dest.with_name("seq.tgz.part")
    # 200 announcing 1024 bytes, connection closes after 600
    monkeypatch.setattr(P.urllib.request, "urlopen", serve([_Resp(200, data[:600], {"Content-Length": "1024"})]))
    with pytest.raises(OSError, match="incomplete download"):
        P.download("https://example.invalid/seq.tgz", dest)
    assert not dest.exists() and part.read_bytes() == data[:600]
    # 206 resume, short again: still incomplete, the .part grows
    monkeypatch.setattr(P.urllib.request, "urlopen", serve([
        _Resp(206, data[600:900], {"Content-Length": "424", "Content-Range": "bytes 600-1023/1024"})]))
    with pytest.raises(OSError, match="rerun to resume"):
        P.download("https://example.invalid/seq.tgz", dest)
    assert calls[-1].get("Range") == "bytes=600-" and part.stat().st_size == 900 and not dest.exists()
    # 206 with the rest: renamed
    monkeypatch.setattr(P.urllib.request, "urlopen", serve([
        _Resp(206, data[900:], {"Content-Length": "124", "Content-Range": "bytes 900-1023/1024"})]))
    assert P.download("https://example.invalid/seq.tgz", dest) == dest
    assert calls[-1].get("Range") == "bytes=900-"
    assert dest.read_bytes() == data and not part.exists()
    # a server that ignores the range (200) restarts the file; without a length nothing can be checked
    dest.unlink()
    part.write_bytes(data[:10])
    monkeypatch.setattr(P.urllib.request, "urlopen", serve([_Resp(200, data, {"Content-Length": "1024"})]))
    assert P.download("https://example.invalid/seq.tgz", dest).read_bytes() == data


def _jpeg(seed):
    img = np.random.default_rng(seed).integers(0, 255, (24, 32, 3), dtype=np.uint8)
    return cv2.imencode(".jpg", img)[1].tobytes()


def _tarball(path, name, offsets, t0=1646200000, extra=()):
    with tarfile.open(path, "w:gz") as tf:
        for k, off in enumerate(offsets):
            body = _jpeg(k)
            ti = tarfile.TarInfo(f"{name}/{t0 + off - offsets[0]}_{off}.jpg")
            ti.size = len(body)
            tf.addfile(ti, io.BytesIO(body))
        for ti in extra:
            tf.addfile(ti)
    return path


def test_prepare_skips_and_deletes_a_truncated_archive(tmp_path, monkeypatch):
    good, bad = "20220302-Jimfire-0921-stgo-s-mobo-c", "20220303-Otherfire-1000-rm-n-mobo-c"
    src = tmp_path / "src"
    src.mkdir()
    link = tarfile.TarInfo(f"{good}/1646200000_-9999.jpg")
    link.type, link.linkname = tarfile.SYMTYPE, "/etc/passwd"
    offsets = list(range(-720, 721, 60))
    _tarball(src / f"{good}.tgz", good, offsets, extra=[link])
    whole = _tarball(src / "whole.tgz", bad, offsets).read_bytes()
    (src / f"{bad}.tgz").write_bytes(whole[: len(whole) // 2])                # truncated gzip stream

    def fake_download(url, dest):
        name = url.rsplit("/", 1)[1]
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((src / name).read_bytes())
        return dest

    monkeypatch.setattr(P, "download", fake_download)
    out = tmp_path / "figlib"
    assert P.main(["--out", str(out), "--names", f"{good},{bad}", "--max-width", "16"]) == 0
    skipped = json.loads((out / "prepare_skipped.json").read_text())
    assert [s["name"] for s in skipped] == [bad] and "corrupt" in skipped[0]["reason"]
    assert not (out / "_downloads" / f"{bad}.tgz").exists()                   # deleted: a rerun downloads it
    assert not (out / bad).exists()
    assert not (out / "_downloads" / f"{good}.tgz").exists()                  # done: tarball removed
    rec = json.loads((out / good / "sequence.json").read_text())
    assert rec["n_frames"] == len(offsets) and rec["camera_id"] == "stgo-s-mobo-c"
    assert all(-9999 != f["offset_s"] for f in rec["frames"])                 # the link member was not used
    assert cv2.imread(str(out / good / rec["frames"][0]["filename"])).shape[1] == 16
    with open(out / "real.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["smoke"] for r in rows] == ["0", "1"] and {r["camera_mount"] for r in rows} == {"stgo-s"}
    assert int(rows[0]["plume_offset_last_s"]) < 0 <= 60 <= int(rows[1]["plume_offset_first_s"])
