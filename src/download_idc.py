"""Загрузка серий DICOM из открытого хранилища NCI Imaging Data Commons (IDC) — по сериям, без
скачивания коллекции целиком.

Список серий берется из индекса IDC (`idc_index.parquet` из пакета `idc-index-data`, релиз 25.0.5;
копия — `data/external/idc_index/`). Файлы серии лежат в открытом бакете Google Cloud Storage
`idc-open-data/<crdc_series_uuid>/`; каждый файл сверяется по md5 из хранилища.

Наборы (оба CC BY 4.0, только для проверки системы, не для обучения):
- `vs_mc_rc` — Vestibular-Schwannoma-MC-RC: все серии T1 (описание начинается с `_T1`):
  124 серии, 93 пациента, ~2,1 ГБ; аппараты GE, Siemens, Philips;
- `vs_seg` — Vestibular-Schwannoma-SEG: случайные N пациентов (seed 42) — серия T1 с контрастом
  и RTSTRUCT с контуром опухоли (по нему выбираются срезы без опухоли);
- `vs_seg_fresh` — свежая проверка круга 4 (решение команды 2026-11-14): все пациенты VS-SEG с серией T1,
  которых нет в `data/external/vs_seg/series.csv` и нет в коллекции VS-MC-RC (152 пациента, ~9,7 ГБ).

    python -m src.download_idc --preset vs_mc_rc --out data/external/vs_mc_rc
    python -m src.download_idc --preset vs_seg --n-patients 40 --out data/external/vs_seg
    python -m src.download_idc --preset vs_seg_fresh --out data/external/vs_seg_fresh
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

GCS_LIST = "https://storage.googleapis.com/storage/v1/b/idc-open-data/o"
GCS_GET = "https://storage.googleapis.com/idc-open-data/"
UA = {"User-Agent": "research-download-script"}
COLS = ["collection_id", "PatientID", "StudyInstanceUID", "SeriesInstanceUID", "Modality",
        "SeriesDescription", "Manufacturer", "ManufacturerModelName", "series_size_MB",
        "instanceCount", "crdc_series_uuid", "license_short_name"]


def get(url: str, tries: int = 8) -> bytes:
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300) as r:
                return r.read()
        except Exception as e:  # DNS на этой машине отвечает через раз
            if k == tries - 1:
                raise
            print(f"  повтор {k + 1}: {e}", flush=True)
            time.sleep(3 * (k + 1))
    return b""


def list_series_files(uuid: str) -> list[dict]:
    items, token = [], None
    while True:
        q = {"prefix": f"{uuid}/", "fields": "items(name,size,md5Hash),nextPageToken"}
        if token:
            q["pageToken"] = token
        d = json.loads(get(GCS_LIST + "?" + urllib.parse.urlencode(q)).decode("utf-8"))
        items += d.get("items", [])
        token = d.get("nextPageToken")
        if not token:
            return items


def md5_b64(data: bytes) -> str:
    return base64.b64encode(hashlib.md5(data).digest()).decode()


def fetch_file(it: dict, sdir: Path) -> None:
    dest = sdir / it["name"].rsplit("/", 1)[-1]
    if dest.exists() and md5_b64(dest.read_bytes()) == it["md5Hash"]:
        return
    data = get(GCS_GET + it["name"])
    if md5_b64(data) != it["md5Hash"]:
        raise IOError(f"не совпала контрольная сумма: {it['name']}")
    dest.write_bytes(data)


def select(idx: pd.DataFrame, preset: str, n_patients: int, seed: int, other: pd.DataFrame | None = None) -> pd.DataFrame:
    if preset == "vs_mc_rc":
        u = idx[idx.collection_id == "vestibular_schwannoma_mc_rc"]
        return u[(u.Modality == "MR") & u.SeriesDescription.str.startswith("_T1")]
    if preset == "vs_seg":
        u = idx[idx.collection_id == "vestibular_schwannoma_seg"]
        t1 = u[(u.Modality == "MR") & u.SeriesDescription.str.lower().str.startswith("t1")]
        patients = pd.Series(sorted(t1.PatientID.unique())).sample(n_patients, random_state=seed)
        keep = u.PatientID.isin(set(patients))
        return pd.concat([t1[t1.PatientID.isin(set(patients))], u[keep & (u.Modality == "RTSTRUCT")]])
    if preset == "vs_seg_fresh":
        u = idx[idx.collection_id == "vestibular_schwannoma_seg"]
        t1 = u[(u.Modality == "MR") & u.SeriesDescription.str.lower().str.startswith("t1")]
        used = set(pd.read_csv("data/external/vs_seg/series.csv").PatientID) | set(other.PatientID)
        fresh = set(t1.PatientID) - used
        return pd.concat([t1[t1.PatientID.isin(fresh)], u[u.PatientID.isin(fresh) & (u.Modality == "RTSTRUCT")]])
    raise ValueError(preset)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", default="data/external/idc_index/idc_index.parquet")
    ap.add_argument("--preset", required=True, choices=["vs_mc_rc", "vs_seg", "vs_seg_fresh"])
    ap.add_argument("--n-patients", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8, help="файлов одновременно")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # только нужная коллекция: весь индекс в памяти занимает ~1,5 ГБ
    coll = {"vs_mc_rc": "vestibular_schwannoma_mc_rc", "vs_seg": "vestibular_schwannoma_seg",
            "vs_seg_fresh": "vestibular_schwannoma_seg"}[args.preset]
    idx = pd.read_parquet(args.index, columns=COLS, filters=[("collection_id", "==", coll)])
    other = (pd.read_parquet(args.index, columns=["collection_id", "PatientID"],
                             filters=[("collection_id", "==", "vestibular_schwannoma_mc_rc")])
             if args.preset == "vs_seg_fresh" else None)
    sel = select(idx, args.preset, args.n_patients, args.seed, other).sort_values(["PatientID", "Modality"])
    print(f"серий {len(sel)}, пациентов {sel.PatientID.nunique()}, "
          f"{sel.series_size_MB.sum() / 1024:.2f} ГБ", flush=True)
    sel.to_csv(out / "series.csv", index=False)

    t0, done_mb = time.time(), 0.0
    for i, s in enumerate(sel.itertuples(index=False), 1):
        sdir = out / "dicom" / s.PatientID / f"{s.Modality}_{s.SeriesInstanceUID}"
        sdir.mkdir(parents=True, exist_ok=True)
        files = list_series_files(s.crdc_series_uuid)
        with ThreadPoolExecutor(args.workers) as pool:  # файлы мелкие: время уходит на ожидание ответа
            list(pool.map(lambda it: fetch_file(it, sdir), files))
        done_mb += s.series_size_MB
        if i % 5 == 0 or i == len(sel):
            print(f"{i}/{len(sel)} серий  {done_mb / 1024:.2f} ГБ  {time.time() - t0:.0f} с", flush=True)
    print("готово:", out, flush=True)


if __name__ == "__main__":
    main()
