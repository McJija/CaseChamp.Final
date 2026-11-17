"""Выбор силы выравнивания резкости R для круга 4 — по правилу, записанному до обучения (2026-11-14).

Только обучающие снимки этапа 3 (наш train + внешний train), снимок в том виде, в каком он идет в модель
(кэш 256×256 → `src.models.harmonize_tensor`). Признаки — те же, что в `src.check_source_probe`.
Правило: из R ∈ {224, 192, 160, 128} берется наибольшее, при котором AUC «наш набор / внешние» и по резкости,
и по шуму не выше 0,65; если ни одно не проходит — 128. Для справки — та же проба «какой набор» (5 фолдов по пациентам).

    python -m src.choose_harmonize
Результат: reports/harmonize_choice.md (+ выбранное R в последней строке вывода).
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from src.check_source_probe import FEAT_RU, FEATS, cv_proba, features
from src.dataset import DataConfig, load_pool
from src.models import harmonize_tensor

CANDIDATES = [224, 192, 160, 128]
LIMIT = 0.65
FALLBACK = 128
REPORT = Path("reports/harmonize_choice.md")


def harmonized(img: np.ndarray, res: int) -> np.ndarray:
    x = torch.from_numpy(img.astype(np.float32))[None, None]
    return harmonize_tensor(x, res)[0, 0].round().clamp(0, 255).numpy().astype(np.uint8)


def main() -> None:
    sys.stdout.reconfigure(errors="replace")
    torch.set_num_threads(6)
    table, cache = load_pool(replace(DataConfig.from_yaml("configs/base.yaml"), table="poolx"))
    d = table[table["split"] == "train"].reset_index(drop=True)
    ds = np.where(d["origin"] == "ext", d["source"], "ours")
    groups = (d["origin"] + ":" + d["group"].astype(str)).to_numpy()
    imgs = [cache.image(p) for p in d["path"]]
    rows = []
    for res in [0] + CANDIDATES:
        F = pd.DataFrame([features(harmonized(im, res) if res else im) for im in imgs])
        single = {f: max(a, 1 - a) for f in FEATS for a in [roc_auc_score(ds == "ours", F[f].fillna(F[f].median()))]}
        pr = cv_proba(F, ds, groups)
        classes = sorted(set(ds))
        auc_ds = {c: roc_auc_score(ds == c, pr[:, classes.index(c)]) for c in ("ours", "upenn", "menrt")}
        med = F.assign(ds=ds).groupby("ds")[["sharp", "noise"]].median()
        rows.append({"R": res or "без выравнивания", "sharp": single["sharp"], "noise": single["noise"],
                     "auc_ours": auc_ds["ours"], "auc_upenn": auc_ds["upenn"], "auc_menrt": auc_ds["menrt"],
                     "top": max(single, key=single.get), "top_auc": max(single.values()),
                     **{f"sharp_{k}": med.loc[k, "sharp"] for k in med.index}, **{f"noise_{k}": med.loc[k, "noise"] for k in med.index}})
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
    R = pd.DataFrame(rows)
    ok = R[(R["R"] != "без выравнивания") & (R["sharp"] <= LIMIT) & (R["noise"] <= LIMIT)]
    chosen = int(ok["R"].astype(int).max()) if len(ok) else FALLBACK
    L = ["# Выбор силы выравнивания резкости (круг 4)", "",
         f"`python -m src.choose_harmonize`. Только обучающие снимки этапа 3 ({len(d)}). Правило (записано до обучения): наибольшее R из "
         f"{CANDIDATES}, при котором AUC «наш набор / внешние» и по резкости, и по шуму ≤ {LIMIT}; иначе {FALLBACK}.", "",
         "| R | AUC по резкости | AUC по шуму | медиана резкости: наш / UPENN / MEN-RT | медиана шума: наш / UPENN / MEN-RT | "
         "проба «какой набор», AUC наш / UPENN / MEN-RT | самый различающий признак |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['R']} | {r['sharp']:.3f} | {r['noise']:.3f} | {r['sharp_ours']:.0f} / {r['sharp_upenn']:.0f} / {r['sharp_menrt']:.0f} | "
                 f"{r['noise_ours']:.2f} / {r['noise_upenn']:.2f} / {r['noise_menrt']:.2f} | "
                 f"{r['auc_ours']:.3f} / {r['auc_upenn']:.3f} / {r['auc_menrt']:.3f} | {FEAT_RU[r['top']]} ({r['top_auc']:.3f}) |")
    L += ["", f"**Выбрано R = {chosen}**" + ("" if len(ok) else " (ни один кандидат не прошел порог — запасное значение по правилу)") + "."]
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"R={chosen}")


if __name__ == "__main__":
    main()
