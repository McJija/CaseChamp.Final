"""Тест с закрашиванием опухоли для всей системы финала (`run3`), а не только для модели типа.

Дополнение к src/eval_occlusion.py: этой проверки не было среди определений, записанных до обучения
(configs/eval_definitions.yaml), — она добавлена 2026-11-16 после того, как закрашивание показало, что модель типа
`run3` на опухоль не смотрит. Снимки, закрашивание и контрольный участок — те же (наш val, срезы с маской; src/stress.py).
Ответ «опухоль есть» — как в системе (src/system.py): область модели контура > 50 пикселей ИЛИ вероятность опухоли у
модели типа ≥ порога configs/presence_run3.yaml. Для каждой из трех копий считается, по снимкам с исходным ответом
«опухоль»: у какой доли ответ сменился на «нет опухоли» после закрашивания опухоли и после закрашивания контроля,
и через что система продолжает видеть опухоль — модель контура нашла область или сработал только порог модели типа.

    python -m src.eval_occlusion_system        # → reports/stress/run3/occlusion_system_val.md и .csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from src.config import load_config
from src.dataset import PreprocCache, to_tensor
from src.eval_occlusion import bootstrap, measures
from src.stress import DEFS, control_region, occlude, tumor_region
from src.system import SystemPredictor


def run(sp: SystemPredictor, images: np.ndarray, batch: int = 16) -> tuple[list[dict], np.ndarray]:
    rows, masks = [], []
    with torch.no_grad():
        for i in range(0, len(images), batch):
            r, m = sp.forward(torch.stack([to_tensor(im) for im in images[i:i + batch]]))
            rows += r
            masks.append(m)
    return rows, np.concatenate(masks)


def where(masks: np.ndarray, regions: np.ndarray, near_px: int = 16) -> dict:
    """Где лежит область модели контура после закрашивания опухоли: на закрашенном месте, рядом (≤ near_px), дальше.
    masks — N×256×256 одной копии (пустые, если область не найдена), regions — закрашенные участки."""
    k = np.ones((2 * near_px + 1, 2 * near_px + 1), np.uint8)
    inside = near = far = 0
    far_slices = found = 0
    for m, reg in zip(masks, regions):
        if not m.any():
            continue
        found += 1
        nb = cv2.dilate(reg.astype(np.uint8), k) > 0
        i, n, f = int((m & reg).sum()), int((m & nb & ~reg).sum()), int((m & ~nb).sum())
        inside, near, far = inside + i, near + n, far + f
        far_slices += f > 0.5 * (i + n + f)
    tot = max(1, inside + near + far)
    return {"found": found, "px_inside": inside / tot, "px_near": near / tot, "px_far": far / tot, "slices_mostly_far": far_slices}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--cache-dir", type=Path, default=Path("data/processed/cache"))
    ap.add_argument("--out-dir", type=Path, default=Path("reports/stress/run3"))
    args = ap.parse_args()

    defs_all = load_config(DEFS)
    defs, unc = defs_all["occlusion"], defs_all["uncertainty"]
    pool = pd.read_csv(args.processed / "pool.csv", keep_default_na=False)
    table = pool[(pool["split"] == "val") & (pool["has_mask"].astype(str).str.lower() == "true")].reset_index(drop=True)
    cache = PreprocCache(args.cache_dir, "pool", 256)
    base, occ, ctl, how, regions = [], [], [], [], []
    for r in table.itertuples():
        img, mask = cache.image(r.path), cache.mask(r.path)
        cand, name = control_region(img, mask, defs)
        region = tumor_region(mask, defs)
        regions.append(region > 0)
        base.append(img)
        occ.append(occlude(img, region, defs))
        ctl.append(occlude(img, cand, defs) if cand is not None else img)
        how.append(name)
    table["has_control"] = [h != "нет" for h in how]

    torch.set_num_threads(4)
    sp = SystemPredictor(core=False, cam=False)
    runs = {tag: run(sp, np.stack(arr)) for tag, arr in (("0", base), ("_occ", occ), ("_ctl", ctl))}
    out = {tag: r for tag, (r, _) in runs.items()}
    occ_masks, regions = runs["_occ"][1], np.stack(regions)
    frames, rows, loc = [], [], []
    for c in range(len(sp.copies)):
        loc.append(where(occ_masks[:, c], regions))
        f = table[["path", "cls", "group", "has_control"]].copy()
        for tag, res in out.items():
            f["tumor" + tag] = [x[f"present_{c}"] for x in res]
            f["found" + tag] = [x[f"found_{c}"] for x in res]
            f["p" + tag] = [x[f"p_tumor_{c}"] for x in res]
            f["thr_only" + tag] = f["tumor" + tag] & ~f["found" + tag]
        frames.append(f)
        a = f[f["tumor0"]]
        m = measures(f)
        rows.append({"copy": c, "n": len(f), "n_answer_tumor": len(a),
                     "still_tumor_after_occ": float(a["tumor_occ"].mean()),
                     "via_contour_after_occ": float(a["found_occ"].mean()),
                     "via_threshold_only_after_occ": float(a["thr_only_occ"].mean()),
                     "contour_found_before": float(a["found0"].mean()),
                     **m})
    res = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    res.round(4).to_csv(args.out_dir / "occlusion_system_val.csv", index=False, encoding="utf-8")
    pd.concat([f.assign(copy=c) for c, f in enumerate(frames)]).to_csv(args.out_dir / "occlusion_system_val_images.csv",
                                                                        index=False, encoding="utf-8")

    ci = {k: bootstrap(frames, k, unc["n_boot"], unc["seed"]) for k in ("flip_rate_minus_control", "flip_rate", "flip_rate_control")}
    mean = {k: float(np.mean([measures(f)[k] for f in frames])) for k in ci}
    pc = lambda x: f"{x * 100:.1f}%"  # noqa: E731
    L = ["# Закрашивание опухоли: вся система финала (`run3`)", "",
         "Дополнительная проверка (не из определений, записанных до обучения; добавлена 2026-11-16). Наш val: срезы Figshare "
         f"с обводкой — {len(table)}, контрольный участок найден у {int(table['has_control'].sum())}. Закрашивание и контроль — "
         "как в `src/eval_occlusion.py`. «Опухоль есть» — как в системе: область модели контура > 50 пикселей или вероятность "
         f"опухоли у модели типа ≥ {sp.thr:.6f}. `python -m src.eval_occlusion_system`.", "",
         "| копия | исходный ответ «опухоль» | после закрашивания опухоли ответ остался «опухоль» | из них модель контура нашла область | "
         "из них только порог модели типа | смена ответа: опухоль / контроль (снимки с контролем) |", "|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['copy'] + 1} | {r['n_answer_tumor']} из {r['n']} | {pc(r['still_tumor_after_occ'])} | {pc(r['via_contour_after_occ'])} | "
                 f"{pc(r['via_threshold_only_after_occ'])} | {pc(r['flip_rate'])} / {pc(r['flip_rate_control'])} |")
    L += ["", f"Среднее трех копий, снимки с контрольным участком: ответ сменился на «нет опухоли» у {pc(mean['flip_rate'])} "
          f"[{pc(ci['flip_rate'][0])}; {pc(ci['flip_rate'][1])}] при закрашенной опухоли и у {pc(mean['flip_rate_control'])} при "
          f"закрашенном контроле; разница {pc(mean['flip_rate_minus_control'])} [{pc(ci['flip_rate_minus_control'][0])}; "
          f"{pc(ci['flip_rate_minus_control'][1])}] (бутстреп по пациентам, {unc['n_boot']} повторов, seed {unc['seed']}).", "",
          "## Где модель контура находит область после закрашивания опухоли", "",
          "Разбивка добавлена после первого прогона, когда стало видно, что модель контура часто находит область и после "
          "закрашивания. Снимки, где после закрашивания найдена область; доли — от всех пикселей найденных областей.", "",
          "| копия | снимков с областью | пикселей на закрашенном месте | рядом (до 16 пикс.) | дальше | снимков, где область в основном дальше |",
          "|---|---|---|---|---|---|"]
    L += [f"| {c + 1} | {w['found']} из {len(table)} | {pc(w['px_inside'])} | {pc(w['px_near'])} | {pc(w['px_far'])} | {w['slices_mostly_far']} |"
          for c, w in enumerate(loc)]
    L += ["", "Как понимать: закрашивание заполняет место опухоли сглаженным пятном, а отек и смещение тканей вокруг (их нет в "
          "обводке Figshare) остаются. Если область лежит на месте опухоли или рядом, модель контура смотрит туда, где была "
          "опухоль; реагирует ли она на пятно или на оставшиеся изменения вокруг, этот тест не различает.", ""]
    (args.out_dir / "occlusion_system_val.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
