import argparse
import io
import json
import math
import multiprocessing as mp
import os
import pickle
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

FRONTEND = {
    "sample_rate": 16000,
    "window_ms": 25,
    "hop_ms": 10,
    "n_fft": 512,
    "n_mels": 40,
    "fmin_hz": 20.0,
    "fmax_hz": 8000.0,
    "pre_emphasis": 0.97,
    "log_floor": 1e-6,
    "mel_scale": "htk",
    "window": "hann_periodic",
}

REPO = "shekar-ai/Neyshekar"
EXPECTED_ITEMS = {"real_train": 58167, "real_dev": 1884}
EXPECTED_FRAMES = {"real_train": 32831886, "real_dev": 1123843}

REF_MEAN = [-11.717080116271973, -9.643693923950195, -8.648884773254395, -7.017769813537598, -6.37713098526001, -6.8447041511535645, -6.445818901062012, -6.300551414489746, -6.802940845489502, -6.659536838531494, -6.671498775482178, -6.964646816253662, -7.1944098472595215, -7.210780620574951, -7.424978733062744, -7.57636022567749, -7.520925998687744, -7.496340274810791, -7.255692958831787, -7.12778377532959, -7.225313663482666, -7.334304332733154, -7.328255653381348, -7.114144802093506, -6.954550743103027, -6.80725622177124, -6.649163722991943, -6.663939952850342, -6.751320838928223, -6.718827724456787, -6.799285411834717, -6.899539470672607, -7.0068488121032715, -7.079463958740234, -7.154209613800049, -7.202225208282471, -7.092080593109131, -6.981790542602539, -7.018618106842041, -7.320793628692627]
REF_STD = [2.2621545791625977, 3.827039957046509, 4.26253604888916, 4.975212097167969, 5.2113237380981445, 5.014046669006348, 5.25247049331665, 5.35931396484375, 5.197705268859863, 5.267049312591553, 5.295414447784424, 5.216392517089844, 5.068057537078857, 5.0487380027771, 4.9764323234558105, 4.923926830291748, 4.915106773376465, 4.919919967651367, 4.9702630043029785, 5.001905918121338, 4.865936756134033, 4.755415916442871, 4.736501693725586, 4.823422431945801, 4.922173976898193, 4.940371036529541, 4.945075988769531, 4.9175519943237305, 4.892247200012207, 4.921128749847412, 4.8858561515808105, 4.892706871032715, 4.8902435302734375, 4.881634712219238, 4.89664363861084, 4.8725905418396, 4.976902961730957, 5.054525375366211, 5.070406913757324, 4.95303201675415]

_AR2FA = {
    "ي": "ی",
    "ك": "ک",
    "ة": "ه",
    "إ": "ا",
    "أ": "ا",
    "ؤ": "و",
    "ئ": "ی",
    "ٰ": "ا",
}
_DIACRITICS = "ًٌٍَُِّْٰٓٔ"
_DIGITS = {}
for _i in range(10):
    _DIGITS[chr(0x660 + _i)] = chr(0x6F0 + _i)
    _DIGITS[str(_i)] = chr(0x6F0 + _i)


def normalize_fa(text):
    _t = str(text)
    for _a, _b in _AR2FA.items():
        _t = _t.replace(_a, _b)
    for _d in _DIACRITICS:
        _t = _t.replace(_d, "")
    _t = _t.replace("ـ", "")
    for _a, _b in _DIGITS.items():
        _t = _t.replace(_a, _b)
    _t = _t.replace("«", "").replace("»", "")
    _t = re.sub(r"[^\w\s‌]", " ", _t)
    _t = re.sub(r"\s+", " ", _t).strip()
    return _t


LETTERS = list("اآبپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی")
VOCAB = (
    ["<blank>"] + LETTERS + [" ", "‌"] + [chr(0x6F0 + _i) for _i in range(10)] + ["<unk>"]
)
CHAR2ID = {c: i for i, c in enumerate(VOCAB)}
UNK_ID = len(VOCAB) - 1


def encode_text(text):
    _ids = []
    for _ch in normalize_fa(text):
        _ids.append(CHAR2ID.get(_ch, UNK_ID))
    return _ids


_F = FRONTEND
_sr = _F["sample_rate"]
_win = int(_sr * _F["window_ms"] / 1000)
_hop = int(_sr * _F["hop_ms"] / 1000)
_nfft = _F["n_fft"]
_nmel = _F["n_mels"]


def _mel_filterbank():
    _fmin, _fmax = _F["fmin_hz"], _F["fmax_hz"]
    _mmin = 2595 * np.log10(1 + _fmin / 700)
    _mmax = 2595 * np.log10(1 + _fmax / 700)
    _mpoints = np.linspace(_mmin, _mmax, _nmel + 2)
    _fpoints = 700 * (10 ** (_mpoints / 2595) - 1)
    _bins = np.floor((_nfft + 1) * _fpoints / _sr).astype(int)
    _fb = np.zeros((_nmel, _nfft // 2 + 1), dtype=np.float32)
    for _m in range(_nmel):
        _l, _c, _r = _bins[_m], _bins[_m + 1], _bins[_m + 2]
        for _k in range(_l, _c):
            _fb[_m, _k] = (_k - _l) / max(_c - _l, 1)
        for _k in range(_c, _r):
            _fb[_m, _k] = (_r - _k) / max(_r - _l, 1)
    return _fb


MELFB = _mel_filterbank()
_window_np = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(_win) / _win)).astype(np.float32)
_mel_t = torch.tensor(MELFB)
_win_t = torch.tensor(_window_np)


def extract_features_torch(wav_t):
    _x = wav_t.to(device="cpu", dtype=torch.float32).clone()
    if _x.numel() > 1:
        _x[1:] = _x[1:] - _F["pre_emphasis"] * _x[:-1]
        _x[0] = _x[0] * (1 - _F["pre_emphasis"])
    _frames = _x.unfold(0, _win, _hop)
    _frames = _frames * _win_t
    _spec = torch.fft.rfft(_frames, n=_nfft, dim=-1)
    _pow = _spec.real**2 + _spec.imag**2
    _e = _pow @ _mel_t.T
    return torch.log(torch.clamp(_e, min=_F["log_floor"]))


def stage_a(work, keep_parquet, max_shards):
    from huggingface_hub import hf_hub_download, list_repo_files

    import pyarrow.parquet as pq

    _t0 = time.time()
    _data = work / "data"
    _ndir = _data / "neyshekar"
    _pqdir = _ndir / "parquet"
    _parts = _data / "parts"
    _pqdir.mkdir(parents=True, exist_ok=True)
    _parts.mkdir(parents=True, exist_ok=True)
    _files = list_repo_files(REPO, repo_type="dataset")
    for _split, _mname in [
        ("train", "neyshekar_train"),
        ("validation", "neyshekar_validation"),
    ]:
        _shards = sorted(
            _f
            for _f in _files
            if _f.startswith(f"data/{_split}-") and _f.endswith(".parquet")
        )
        if max_shards:
            _shards = _shards[:max_shards]
        if not _shards:
            raise RuntimeError(f"شارد پارکت برای {_split} پیدا نشد")
        _sdir = _ndir / _split
        _sdir.mkdir(parents=True, exist_ok=True)
        _rows_all = []
        for _idx, _shard in enumerate(_shards):
            _part = _parts / f"{_mname}_part{_idx:02d}.json"
            if _part.exists():
                _old = json.loads(_part.read_text())
                _rows_all.extend(_old)
                print(
                    f"{_split} شارد {_idx + 1}/{len(_shards)}: از قبل آماده ({len(_old)} ردیف)",
                    flush=True,
                )
                continue
            _st = time.time()
            _local = hf_hub_download(
                REPO, _shard, repo_type="dataset", local_dir=str(_pqdir)
            )
            _table = pq.read_table(_local)
            _rows = []
            _skipped = 0
            _base = len(_rows_all)
            for _rec in _table.to_pylist():
                try:
                    _aud = _rec.get("audio")
                    _text = _rec.get("text")
                    if not _text or not isinstance(_aud, dict) or not _aud.get("bytes"):
                        _skipped += 1
                        continue
                    _arr, _srx = sf.read(io.BytesIO(_aud["bytes"]), dtype="float32")
                    _arr = np.asarray(_arr, dtype=np.float32)
                    if _arr.ndim > 1:
                        _arr = _arr[:, 0]
                    _srx = int(_srx)
                    if _srx != 16000 and _arr.size:
                        _g = math.gcd(_srx, 16000)
                        _arr = resample_poly(
                            _arr, 16000 // _g, _srx // _g
                        ).astype(np.float32)
                        _srx = 16000
                    if not _arr.size:
                        _skipped += 1
                        continue
                    _p = _sdir / f"ney_{_base + len(_rows):06d}.wav"
                    sf.write(str(_p), _arr, _srx)
                    _rows.append({"path": str(_p), "text": str(_text)})
                except Exception:
                    _skipped += 1
                    continue
            _part.write_text(json.dumps(_rows, ensure_ascii=False))
            _rows_all.extend(_rows)
            print(
                f"{_split} شارد {_idx + 1}/{len(_shards)}: {len(_rows)} ردیف · {_skipped} رد شد · {(time.time() - _st):.0f} ثانیه",
                flush=True,
            )
            if not keep_parquet:
                Path(_local).unlink(missing_ok=True)
        _mani = _data / f"manifest_{_mname}.json"
        _mani.write_text(json.dumps(_rows_all, ensure_ascii=False))
        print(f"مانیفست {_mname}: {len(_rows_all)} ردیف", flush=True)
    print(f"مرحله A تمام شد · {(time.time() - _t0) / 60:.1f} دقیقه", flush=True)


def _load_wav16(path):
    _w, _srx = sf.read(path, dtype="float32")
    if _w.ndim > 1:
        _w = _w[:, 0]
    if _srx != 16000:
        _g = math.gcd(_srx, 16000)
        _w = resample_poly(_w, 16000 // _g, _srx // _g).astype(np.float32)
    return _w


_allowed = set("اآبپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی‌ ۰۱۲۳۴۵۶۷۸۹")


def _clean_tokens(text):
    _t = "".join(_c for _c in normalize_fa(text) if _c in _allowed)
    _t = " ".join(_t.split())
    return encode_text(_t)


def _proc_rows(_args):
    _name, _rows = _args
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    _out = []
    for _row in _rows:
        try:
            _w = _load_wav16(_row["path"])
        except Exception:
            continue
        if len(_w) < 4800 or len(_w) > 20.0 * 16000:
            continue
        _toks = _clean_tokens(_row["text"])
        if len(_toks) < 2:
            continue
        _feat = (
            extract_features_torch(torch.tensor(_w)).cpu().numpy().astype(np.float16)
        )
        _out.append({"feat": _feat, "tokens": _toks, "text": _row["text"], "src": _name})
    return _out


def _build(work, split_names, workers):
    _items = []
    for _name in split_names:
        _mani = work / "data" / f"manifest_{_name}.json"
        if not _mani.exists():
            continue
        _rows = json.loads(_mani.read_text())
        _chunks = [
            (_name, _rows[i : i + 1500]) for i in range(0, len(_rows), 1500)
        ]
        _kept = 0
        with mp.get_context("fork").Pool(workers) as _pool:
            for _out in _pool.imap(_proc_rows, _chunks):
                _items.extend(_out)
                _kept += len(_out)
        print(f"{_name}: {_kept} از {len(_rows)} نگه داشته شد", flush=True)
    return _items


def _load_pkl(path):
    with open(path, "rb") as _f:
        return pickle.load(_f)


def stage_c(work, workers, smoke):
    _t0 = time.time()
    _feat = work / "features"
    _feat.mkdir(parents=True, exist_ok=True)
    _sets = {}
    for _name, _splits in [
        ("real_train", ["neyshekar_train"]),
        ("real_dev", ["neyshekar_validation"]),
    ]:
        _pkl = _feat / f"{_name}.pkl"
        if _pkl.exists() and not smoke:
            _items = _load_pkl(_pkl)
            if len(_items) == EXPECTED_ITEMS[_name]:
                print(f"{_name}: از قبل آماده ({len(_items)} آیتم)، بازسازی نمی‌شود", flush=True)
                _sets[_name] = _items
                continue
            print(
                f"{_name}: تعداد آیتم‌ها {len(_items)} با مقدار موردانتظار {EXPECTED_ITEMS[_name]} فرق دارد، بازسازی می‌شود",
                flush=True,
            )
        _items = _build(work, _splits, workers)
        with open(_pkl, "wb") as _f:
            pickle.dump(_items, _f)
        _sets[_name] = _items
    _counts = {}
    for _name, _items in _sets.items():
        _frames = sum(int(_i["feat"].shape[0]) for _i in _items)
        _counts[_name] = {"items": len(_items), "frames": _frames}
        _hrs = _frames * 0.01 / 3600
        print(f"{_name}: {len(_items)} آیتم · ≈{_hrs:.1f} ساعت ویژگی", flush=True)
    _stats_path = _feat / "stats.json"
    if "real_train" in _sets and _sets["real_train"]:
        _sample = np.concatenate(
            [_i["feat"].astype(np.float32) for _i in _sets["real_train"][::7]], axis=0
        )
        _stats = {"mean": _sample.mean(axis=0).tolist(), "std": _sample.std(axis=0).tolist()}
        _stats_path.write_text(json.dumps(_stats))
        print("آمار استانداردسازی ذخیره شد", flush=True)
    (_feat / "counts.json").write_text(json.dumps(_counts))
    print(f"مرحله C تمام شد · {(time.time() - _t0) / 60:.1f} دقیقه", flush=True)
    return _sets


def verify(work, smoke):
    _feat = work / "features"
    _ok = True
    for _name in ["real_train", "real_dev"]:
        _pkl = _feat / f"{_name}.pkl"
        if not _pkl.exists():
            print(f"راستی‌آزمایی {_name}: فایل نیست — ناموفق", flush=True)
            _ok = False
            continue
        _items = _load_pkl(_pkl)
        _frames = sum(int(_i["feat"].shape[0]) for _i in _items)
        _n_ok = len(_items) == EXPECTED_ITEMS[_name]
        _f_ok = abs(_frames - EXPECTED_FRAMES[_name]) <= 0.005 * EXPECTED_FRAMES[_name]
        print(
            f"راستی‌آزمایی {_name}: {len(_items)} آیتم (انتظار {EXPECTED_ITEMS[_name]}) · {_frames} فریم (انتظار {EXPECTED_FRAMES[_name]}) — {'قبول' if _n_ok and _f_ok else 'ناموفق'}",
            flush=True,
        )
        if smoke:
            print("حالت دود: مقایسه با مقادیر مرجع انجام نمی‌شود", flush=True)
        elif not (_n_ok and _f_ok):
            _ok = False
    _stats_path = _feat / "stats.json"
    if _stats_path.exists() and REF_MEAN and not smoke:
        _stats = json.loads(_stats_path.read_text())
        _dm = max(abs(float(a) - float(b)) for a, b in zip(_stats["mean"], REF_MEAN))
        _ds = max(abs(float(a) - float(b)) for a, b in zip(_stats["std"], REF_STD))
        _s_ok = _dm < 0.05 and _ds < 0.05
        print(
            f"راستی‌آزمایی stats.json: بیشترین اختلاف میانگین {_dm:.4f} · انحراف معیار {_ds:.4f} (آستانه ۰٫۰۵) — {'قبول' if _s_ok else 'ناموفق'}",
            flush=True,
        )
        if not _s_ok:
            _ok = False
    print(f"راستی‌آزمایی نهایی: {'قبول' if _ok else 'ناموفق'}", flush=True)
    return _ok


def main():
    _ap = argparse.ArgumentParser()
    _ap.add_argument("--work", default="stt_v6_work")
    _ap.add_argument("--workers", type=int, default=4)
    _ap.add_argument("--keep-parquet", action="store_true")
    _ap.add_argument("--keep-wavs", action="store_true")
    _ap.add_argument("--skip-a", action="store_true")
    _ap.add_argument("--skip-c", action="store_true")
    _ap.add_argument("--verify-only", action="store_true")
    _ap.add_argument("--max-shards", type=int, default=0)
    _args = _ap.parse_args()
    _work = Path(_args.work)
    for _sub in ["data", "features", "ckpt", "logs"]:
        (_work / _sub).mkdir(parents=True, exist_ok=True)
    print(f"پوشه‌ی کاری: {_work.resolve()}", flush=True)
    _smoke = _args.max_shards > 0
    _t0 = time.time()
    if _args.verify_only:
        sys.exit(0 if verify(_work, False) else 1)
    if not _args.skip_a:
        stage_a(_work, _args.keep_parquet, _args.max_shards)
    if not _args.skip_c:
        stage_c(_work, _args.workers, _smoke)
    _ok = verify(_work, _smoke)
    if _ok and not _smoke:
        if not _args.keep_wavs:
            shutil.rmtree(_work / "data" / "neyshekar" / "train", ignore_errors=True)
            shutil.rmtree(_work / "data" / "neyshekar" / "validation", ignore_errors=True)
            print("فایل‌های wav موقت پاک شدند (مانیفست‌ها ماندند)", flush=True)
        if not _args.keep_parquet:
            shutil.rmtree(_work / "data" / "neyshekar" / "parquet", ignore_errors=True)
    print(f"مجموع زمان آماده‌سازی: {(time.time() - _t0) / 60:.1f} دقیقه", flush=True)
    sys.exit(0 if _ok else 1)


if __name__ == "__main__":
    main()
