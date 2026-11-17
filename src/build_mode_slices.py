"""Срезы других режимов МРТ (T1 без контраста, T2, FLAIR) у тех же пациентов и на тех же местах,
что срезы T1 с контрастом в `data/processed/external_slices.csv`.

Зачем: (1) замер системы по режимам (система обучена на T1 с контрастом; на других режимах она
выдает результат с предупреждением); (2) данные для определителя режима.
Срез одного и того же места головы во всех четырех режимах — значит, режим нельзя угадать по
месту среза, пациенту или источнику, только по виду ткани.

Наборы:
  - `upenn` — UPENN-GBM: T1w, T2w, FLAIR в той же сетке (атлас SRI24), что T1 с контрастом
    (проверено для всех 147 пациентов) — срез берется по тому же номеру; обводка та же;
  - `ms` — склероз (Любляна): T1W, T2W и FLAIR пересчитываются в сетку T1WKS (линейно) через файлы
    совмещения авторов (`*_to_FLAIR.txt`); по координатам аппарата нельзя — FLAIR повернут на ~11°
    и сдвинут на ~2 см (P11-6).
Обработка та же, что у T1 с контрастом (`src/build_external_slices.py`): пиксель 0,7 мм, окно
0–99,8-й перцентиль среза, все вне головы обнулено (голова — по самому снимку этого режима).

    python -m src.build_mode_slices
Результат: PNG `data/external/slices/<набор>/<пациент>_<плоскость><срез>_{t1,t2,flair}.png`,
таблица `data/processed/external_modes.csv` (те же колонки, `mode` = t1 / t2 / flair, `t1c_path` —
срез T1 с контрастом того же места), кэш `extm_256`, лист `reports/external/modes_examples.png`.
Эти срезы в обучение моделей опухоли не идут — отдельная таблица и отдельный кэш.
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np
import pandas as pd

from src.build_external_slices import EXT, head_thr, keep_head, load, ms_to_ref, take, to_px, to_u8
from src.dataset import DataConfig, PreprocCache, build_cache

SEED = 42
MODES = {  # режим → файл
    "upenn": {"t1": "upenn_gbm/nifti/{p}_T1w.nii.gz", "t2": "upenn_gbm/nifti/{p}_T2w.nii.gz",
              "flair": "upenn_gbm/nifti/{p}_FLAIR.nii.gz"},
    "ms": {"t1": "ms_ljubljana/{p}/T1W.nii.gz", "t2": "ms_ljubljana/{p}/T2W.nii.gz",
           "flair": "ms_ljubljana/{p}/FLAIR.nii.gz"},
}
REF = {"upenn": "upenn_gbm/nifti/{p}_ce-gd_T1w.nii.gz", "ms": "ms_ljubljana/{p}/T1WKS.nii.gz"}
MODE_RU = {"t1c": "T1 с контрастом", "t1": "T1", "t2": "T2", "flair": "FLAIR"}


def process(source: str, patient: str, rows: pd.DataFrame, out_dir: Path, px_mm: float) -> list[dict]:
    ref_path = EXT / REF[source].format(p=patient)
    ref_arr, ref_zooms = load(ref_path)
    ref_shape = ref_arr.shape
    del ref_arr
    out = []
    for mode, tmpl in MODES[source].items():
        path = EXT / tmpl.format(p=patient)
        if source == "ms":  # через FLAIR: файлы совмещения авторов
            xfm = {"t1": "T1W_to_FLAIR.txt", "t2": "T2W_to_FLAIR.txt", "flair": None}[mode]
            vol, zooms = load(ms_to_ref(path, ref_path, path.parent / xfm if xfm else None,
                                        path.parent / "T1WKS_to_FLAIR.txt", order=1))
        else:
            vol, zooms = load(path)
        if vol.shape != ref_shape:
            raise ValueError(f"{patient} {mode}: сетка {vol.shape} не совпадает с T1 с контрастом {ref_shape}")
        if not np.allclose(zooms, ref_zooms, atol=1e-3):
            raise ValueError(f"{patient} {mode}: шаг {zooms} не совпадает с {ref_zooms}")
        thr = head_thr(vol)
        for r in rows.itertuples(index=False):
            raw = to_px(*take(vol, zooms, r.plane, int(r.slice)), px_mm, nearest=False)
            img = keep_head(to_u8(raw), raw, thr)
            name = r.path.removesuffix("_t1c.png") + f"_{mode}.png"
            cv2.imwrite(str(out_dir / name), img)
            row = r._asdict()
            row.update({"path": name, "mode": mode, "t1c_path": r.path, "flair": "", "label4": ""})
            out.append(row)
        del vol
        gc.collect()
    return out


def sheet(table: pd.DataFrame, cache: PreprocCache, t1c_cache: PreprocCache, out: Path, n: int = 4) -> None:
    """По n пациентов каждого набора: один срез во всех четырех режимах, как его видит модель."""
    tiles = []
    for src, g in table.groupby("source", sort=False):
        for p in g["patient"].drop_duplicates().sample(min(n, g["patient"].nunique()), random_state=SEED):
            r = g[(g["patient"] == p) & (g["mode"] == "t1")]
            x = r.iloc[len(r) // 2]
            row = []
            for mode in ("t1c", "t1", "t2", "flair"):
                im = t1c_cache.image(x["t1c_path"]) if mode == "t1c" else cache.image(x["t1c_path"].removesuffix("_t1c.png") + f"_{mode}.png")
                im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
                cv2.putText(im, f"{src} {p[-9:]} {x['plane']} {mode}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                row.append(im)
            tiles.append(np.hstack(row))
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack(tiles))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    ap.add_argument("--out", type=Path, default=EXT / "slices")
    ap.add_argument("--px-mm", type=float, default=0.7)
    ap.add_argument("--limit", type=int, default=0, help="только первые N пациентов каждого набора (проверка)")
    args = ap.parse_args()
    sys.stdout.reconfigure(errors="replace")
    cfg = DataConfig.from_yaml(args.config)
    processed = Path(cfg.processed_dir)

    t1c = pd.read_csv(processed / "external_slices.csv", keep_default_na=False)
    t1c = t1c[t1c["source"].isin(list(MODES))]
    for col in ("has_mask", "has_tumor", "ms_lesion"):
        t1c[col] = t1c[col].astype(str).str.lower().eq("true")
    pts = t1c[["source", "patient"]].drop_duplicates()
    if args.limit:
        pts = pts.groupby("source").head(args.limit)
    done_dir = args.out / "_rows_modes"
    done_dir.mkdir(parents=True, exist_ok=True)
    t0, parts = time.time(), []
    for i, (src, p) in enumerate(pts.itertuples(index=False), 1):
        f = done_dir / f"{src}__{p}.csv"
        if not f.exists():
            pd.DataFrame(process(src, p, t1c[(t1c["source"] == src) & (t1c["patient"] == p)], args.out, args.px_mm)
                         ).to_csv(f, index=False, encoding="utf-8")
        parts.append(f)
        if i % 25 == 0 or i == len(pts):
            print(f"{i}/{len(pts)} пациентов, {time.time() - t0:.0f} с", flush=True)
    table = pd.concat([pd.read_csv(f, keep_default_na=False) for f in parts], ignore_index=True)
    for col in ("has_mask", "has_tumor", "ms_lesion"):
        table[col] = table[col].astype(str).str.lower().eq("true")
    table.to_csv(processed / "external_modes.csv", index=False, encoding="utf-8")
    build_cache(table, None, cfg.cache_dir, "extm", cfg.size, roots={"ext": args.out})
    sheet(table, PreprocCache(cfg.cache_dir, "extm", cfg.size), PreprocCache(cfg.cache_dir, "ext", cfg.size),
          Path("reports/external/modes_examples.png"))
    print(pd.crosstab([table["source"], table["mode"]], table["split"]).to_string())
    print(f"срезов {len(table)}, {time.time() - t0:.0f} с")


if __name__ == "__main__":
    main()
