"""Кэш для модели ядра и отека (глиобластома, UPENN-GBM): FLAIR того же среза в геометрии T1 с контрастом
и маска всей опухоли.

Модель ядра и отека учится на срезах UPENN-GBM (`data/processed/external_slices.csv`, source == upenn):
  - вход — T1 с контрастом из кэша `ext_256` (как у основной системы) и, для сравнения, FLAIR того же среза;
  - выход — два канала: ядро (метки 1 + 3 — некроз и контрастируемая часть; это маска кэша `ext_256`)
    и вся опухоль (метки 1 + 2 + 3); отек = вся опухоль минус ядро.
FLAIR и обводка лежат в той же сетке, что T1 с контрастом (атлас SRI24), поэтому к ним применяется ровно та же
обрезка и масштаб, что препроцессинг выбрал для T1 с контрастом (геометрия из `ext_256.csv`); яркость FLAIR
нормализуется по его собственному 99,5-му перцентилю ткани. Иначе препроцессинг выбрал бы для FLAIR свою рамку,
и каналы разошлись бы на несколько пикселей.

    python -m src.build_core_cache
Результат: кэш `upennf_256` (снимки — FLAIR, маски — вся опухоль; `path` — путь FLAIR, `t1c_path` — путь
T1 с контрастом), все части UPENN (train / val / holdout); лист `reports/external/core_edema_examples.png`.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.dataset import CACHE_VERSION, GEOMETRY, DataConfig, PreprocCache, _cache_files
from src.io_utils import read_image
from src.preprocess import PreprocMeta, mask_to_model, to_gray

SEED = 42


def meta_of(row: pd.Series, size: int) -> PreprocMeta:
    names = {f.name for f in fields(PreprocMeta)}
    return PreprocMeta(**{k: row[k] for k in GEOMETRY if k in names}, size=size, bg=0.0, hi=255.0)


def align(image: np.ndarray, meta: PreprocMeta) -> np.ndarray:
    """Другой режим того же среза (та же сетка) → координаты модели по геометрии T1 с контрастом."""
    g = to_gray(image)
    if g.shape != (meta.orig_h, meta.orig_w):
        raise ValueError(f"размер {g.shape} не совпадает с T1 с контрастом {(meta.orig_h, meta.orig_w)}")
    crop = g[meta.y0 : meta.y1, meta.x0 : meta.x1]
    tissue = crop[crop > 0]  # вне головы обнулено при нарезке
    hi = float(np.percentile(tissue, 99.5)) if tissue.size else 1.0
    norm = np.clip(crop / max(hi, 1e-6), 0.0, 1.0)
    interp = cv2.INTER_AREA if meta.scale < 1 else cv2.INTER_LINEAR
    resized = np.clip(cv2.resize(norm, (meta.new_w, meta.new_h), interpolation=interp), 0.0, 1.0)
    out = np.zeros((meta.size, meta.size), np.float32)
    out[meta.top : meta.top + meta.new_h, meta.left : meta.left + meta.new_w] = resized
    return np.round(out * 255).astype(np.uint8)


def sheet(table: pd.DataFrame, t1c: PreprocCache, fl: PreprocCache, out: Path, n: int = 8) -> None:
    """Срезы train с опухолью: T1 с контрастом | FLAIR | ядро (красный) и отек (желтый) на T1 с контрастом."""
    x = table[(table["split"] == "train") & table["has_tumor"]].sample(n, random_state=SEED)
    rows = []
    for r in x.itertuples(index=False):
        a, b = t1c.image(r.t1c_path), fl.image(r.path)
        core, whole = t1c.mask(r.t1c_path) > 0, fl.mask(r.path) > 0
        ov = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
        for m, color in ((whole & ~core, (0, 220, 255)), (core, (0, 0, 255))):
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(ov, cnts, -1, color, 1)
        tiles = [cv2.cvtColor(a, cv2.COLOR_GRAY2BGR), cv2.cvtColor(b, cv2.COLOR_GRAY2BGR), ov]
        cv2.putText(tiles[0], f"{r.patient} {r.plane}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        rows.append(np.hstack(tiles))
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack(rows))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    ap.add_argument("--slices", type=Path, default=Path("data/external/slices"))
    args = ap.parse_args()
    sys.stdout.reconfigure(errors="replace")
    cfg = DataConfig.from_yaml(args.config)
    t1c = PreprocCache(cfg.cache_dir, "ext", cfg.size)
    tab = t1c.table[t1c.table["source"] == "upenn"].reset_index(drop=True)
    tab["has_tumor"] = tab["has_tumor"].astype(str).str.lower().eq("true")
    if (tab["flair"] == "").any() or tab["flair"].isna().any() or tab["label4"].isna().any():
        raise ValueError("у части срезов UPENN нет FLAIR или полной обводки")

    files = _cache_files(cfg.cache_dir, "upennf", cfg.size)
    images = np.lib.format.open_memmap(files["images"], mode="w+", dtype=np.uint8, shape=(len(tab), cfg.size, cfg.size))
    masks = np.lib.format.open_memmap(files["masks"], mode="w+", dtype=np.uint8, shape=(len(tab), cfg.size, cfg.size))
    bad = 0
    for i, r in enumerate(tqdm(tab.itertuples(index=False), total=len(tab), desc="кэш upennf")):
        meta = meta_of(pd.Series(r._asdict()), cfg.size)
        images[i] = align(read_image(args.slices / r.flair), meta)
        lab = read_image(args.slices / r.label4)
        lab = lab if lab.ndim == 2 else lab[..., 0]
        masks[i] = mask_to_model((lab > 0).astype(np.uint8), meta)
        core = t1c.mask(r.path) > 0
        bad += int((core & (masks[i] == 0)).sum())  # ядро должно лежать внутри всей опухоли
    images.flush()
    masks.flush()
    out = pd.DataFrame({"path": tab["flair"], "t1c_path": tab["path"], "split": tab["split"], "source": tab["source"],
                        "patient": tab["patient"], "group": tab["group"], "plane": tab["plane"], "has_tumor": tab["has_tumor"]})
    pd.concat([out, tab[GEOMETRY]], axis=1).to_csv(files["table"], index=False, encoding="utf-8")
    files["info"].write_text(json.dumps({"version": CACHE_VERSION, "size": cfg.size, "n": len(tab), "has_masks": True}), encoding="utf-8")

    fl = PreprocCache(cfg.cache_dir, "upennf", cfg.size)
    core_px = np.array([int((t1c.mask(p) > 0).sum()) for p in tab["path"]])
    whole_px = np.array([int((fl.mask(p) > 0).sum()) for p in tab["flair"]])
    sheet(fl.table.assign(has_tumor=tab["has_tumor"]), t1c, fl, Path("reports/external/core_edema_examples.png"))
    print(f"срезов {len(tab)}; пикселей ядра вне всей опухоли: {bad}")
    print(pd.crosstab(tab["split"], tab["has_tumor"]).to_string())
    t = tab["has_tumor"].to_numpy()
    print(f"с опухолью: медиана площади ядра {np.median(core_px[t]):.0f} пикс., всей опухоли {np.median(whole_px[t]):.0f}; "
          f"срезов с отеком {(whole_px[t] > core_px[t]).sum()} из {t.sum()}; без опухоли с непустой маской всей опухоли {(whole_px[~t] > 0).sum()}")


if __name__ == "__main__":
    main()
