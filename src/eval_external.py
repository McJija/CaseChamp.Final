"""Замер системы по правилам П1–П8 (`configs/eval_external.yaml`).

Три копии системы (классификатор + сегментатор на seed 42, 43, 44, порог «опухоль есть» из
`models/final/presence.yaml`) прогоняются на снимках из кэшей препроцессинга — ровно то, что видит модель:
  - `val` — наш val (pool_256, 525 снимков);
  - `ext` — внешние срезы T1 с контрастом (ext_256) частей val, holdout и check (train не открывается);
  - `modes` — те же места тех же пациентов в режимах T1, T2, FLAIR (extm_256), части val, holdout, check.
Копия 0 (seed 42) — основная система, она же models/final (сверяется по md5). Для каждого снимка:
ответы трех копий, площадь маски, Dice маски с обводкой (в координатах модели, 256×256) и попарный
Dice масок копий. По этой таблице считаются П1 (главная проверка), П3 (доля менингиом и гипофиза с типом
«глиома»), П5 (польза «проверьте»), результаты по плоскостям и режимам. Интервалы — бутстреп по
пациентам (`group`), 2000 повторов, seed 42. По check и holdout — только сводные числа.

    python -m src.eval_external --tag before
На машине с 7 ГБ памяти — по одному набору на процесс (долгий процесс с шестью моделями падал):
    python -m src.eval_external --tag before --sets val --only-predict   # затем ext, modes
    python -m src.eval_external --tag before --report-only
Кандидат на замену (файл системы: presence, copies, check_final: false, confidence — способ «проверьте»,
выбранный на val, `src/confidence.py`) — сначала только на val:
    python -m src.eval_external --tag run3_val --system configs/system_run3.yaml --splits val --sets val --only-predict
Набор `test` — старый test (pool, 582 снимка; просмотрен при разборе ошибок) — только «рядом», в правила не входит.
Результат: reports/eval_external/<tag>/{pred_<набор>.csv, predictions.csv, copies.json, metrics.json, report.md}.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.confidence import signals as conf_signals
from src.dataset import DataConfig, PreprocCache, to_tensor
from src.evaluate import boot_ci
from src.inference import FINAL, MODELS, _load
from src.labels import CLASSES, NO_TUMOR_IDX
from src.metrics import wilson

TUMOR_TYPES = CLASSES[:NO_TUMOR_IDX]
NC = 3  # копий
PLANE_RU = {"ax": "аксиальная", "cor": "корональная", "sag": "сагиттальная"}
MODE_RU = {"t1c": "T1 с контрастом", "t1": "T1", "t2": "T2", "flair": "FLAIR", "?": "неизвестен"}
SIG_RU = {"chk_presence": "расходятся «есть / нет»", "chk_type": "тип", "chk_contour": "контур",
          "chk_models": "модели внутри системы", "chk_type_prob": "тип неуверенный"}
SRC_RU = {"upenn": "UPENN-GBM (глиобластома)", "menrt": "BraTS-MEN-RT (менингиома)", "ms": "склероз (Любляна)",
          "vs": "шванномы (Лондон)"}


# ---------------------------------------------------------------- модели и данные


def md5(path: Path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def find_run(pattern: str) -> Path:
    runs = sorted(MODELS.glob(pattern))
    if not runs:
        raise FileNotFoundError(f"нет весов {MODELS / pattern}")
    return runs[-1] / "best.pt"


def load_copies(system: dict, device: torch.device) -> tuple[list, float, int, list[dict]]:
    pres = yaml.safe_load(Path(system["presence"]).read_text(encoding="utf-8"))
    man = json.loads((FINAL / "manifest.json").read_text(encoding="utf-8"))
    copies, info = [], []
    for i, c in enumerate(system["copies"]):
        pc, ps = find_run(c["classifier"]), find_run(c["segmenter"])
        if i == 0 and system.get("check_final", True) and (
                md5(pc) != man["files"]["classifier.pt"]["md5"] or md5(ps) != man["files"]["segmenter.pt"]["md5"]):
            raise ValueError("копия 0 не совпадает с models/final")
        copies.append((_load(pc, device)[0], _load(ps, device)[0]))
        info.append({"classifier": pc.parent.name, "segmenter": ps.parent.name})
    return copies, float(pres["cls_threshold"]), int(pres["min_area_px"]), info


def tables(cfg: DataConfig, sets: list[str], splits: tuple[str, ...] = ("val", "holdout", "check"),
           ext: tuple[str, str] = ("external_slices.csv", "ext")) -> list[tuple[pd.DataFrame, PreprocCache]]:
    """ext — таблица и кэш набора `ext` (свежая проверка круга 4: external_fresh.csv, extf)."""
    processed, out = Path(cfg.processed_dir), []
    for name in ("val", "test"):  # наш val и старый test (pool)
        if name in sets:
            v = pd.read_csv(processed / "pool.csv", keep_default_na=False)
            v = v[v["split"] == name].copy()
            v["has_mask"] = v["has_mask"].astype(str).str.lower().eq("true")
            v = v.assign(set=name, has_tumor=v["cls"] != "no_tumor", plane="", mode="", patient=v["group"], ms_lesion=False)
            out.append((v, PreprocCache(cfg.cache_dir, "pool", cfg.size)))
    for name, csv, cache in (("ext", *ext), ("modes", "external_modes.csv", "extm")):
        if name in sets:
            t = pd.read_csv(processed / csv, keep_default_na=False)
            t = t[t["split"].isin(splits) & (t["split"] != "train")].copy()  # train внешних наборов не открывается
            if name == "modes":  # T2 и FLAIR UPENN в корональной и сагиттальной плоскости — размытый пересчет (P11-7)
                t = t[t["plane"] == "ax"]
            for col in ("has_mask", "has_tumor", "ms_lesion"):
                t[col] = t[col].astype(str).str.lower().eq("true")
            t["set"] = name
            out.append((t, PreprocCache(cfg.cache_dir, cache, cfg.size)))
    return out


def typed(pred: pd.DataFrame) -> pd.DataFrame:
    """Таблица предсказаний из csv (keep_default_na=False) → логические и числовые колонки."""
    for col in [c for c in pred.columns if c.startswith(("present_", "found_")) or c in ("has_tumor", "has_mask", "ms_lesion")]:
        pred[col] = pred[col].astype(str).str.lower().eq("true")
    for col in [c for c in pred.columns if c.startswith(("dice_gt_", "p_tumor_", "p_", "pair_dice", "area_", "gt_area"))]:
        pred[col] = pd.to_numeric(pred[col], errors="coerce")
    return pred


def dice(a: np.ndarray, b: np.ndarray) -> float:
    s = int(a.sum() + b.sum())
    return 2 * int((a & b).sum()) / s if s else 1.0


@torch.no_grad()
def predict(table: pd.DataFrame, cache: PreprocCache, copies: list, thr: float, min_area: int, bs: int) -> pd.DataFrame:
    rows, paths, t0 = [], table["path"].tolist(), time.time()
    for i in range(0, len(paths), bs):
        chunk = paths[i:i + bs]
        x = torch.stack([to_tensor(cache.image(p)) for p in chunk])
        gts = [cache.mask(p) > 0 for p in chunk]
        out = []
        for cls, seg in copies:
            probs = torch.softmax(cls(x).float(), 1).numpy()
            m = (torch.sigmoid(seg(x).float())[:, 0] > 0.5).numpy()
            area = m.reshape(len(chunk), -1).sum(1)
            m[area <= min_area] = False  # площадь ≤ порога — область не найдена, маска пустая
            out.append((probs, m, area))
        for j, p in enumerate(chunk):
            r, masks = {"path": p, "gt_area": int(gts[j].sum())}, []
            for c, (probs, m, area) in enumerate(out):
                pr, found = probs[j], bool(area[j] > min_area)
                p_t = float(1 - pr[NO_TUMOR_IDX])
                present = found or p_t >= thr
                typ = TUMOR_TYPES[int(pr[:NO_TUMOR_IDX].argmax())]
                r.update({f"p_tumor_{c}": p_t, f"found_{c}": found, f"area_{c}": int(area[j]) if found else 0,
                          f"present_{c}": present, f"type_{c}": typ, f"answer_{c}": typ if present else "no_tumor",
                          f"cls_argmax_{c}": CLASSES[int(pr.argmax())],
                          f"dice_gt_{c}": dice(m[j], gts[j]) if gts[j].any() else np.nan})
                if c == 0:
                    r.update({f"p_{k}": float(pr[ki]) for ki, k in enumerate(CLASSES)})
                masks.append(m[j])
            r["pair_dice"] = float(np.mean([dice(masks[a], masks[b]) for a, b in ((0, 1), (0, 2), (1, 2))]))
            rows.append(r)
        if (i // bs) % 25 == 0:
            print(f"  {i + len(chunk)}/{len(paths)}  {time.time() - t0:.0f} с", flush=True)
    return table.reset_index(drop=True).merge(pd.DataFrame(rows), on="path", how="left", validate="one_to_one")


# ---------------------------------------------------------------- меры


def default_rule(c: dict) -> dict:
    """Прежнее определение «проверьте» (П5, записано 2026-11-12): копии расходятся в «есть / нет» или в типе,
    или средний попарный Dice масок копий < min_pair_dice."""
    return {"name": "A: прежний (есть/нет, тип, контур < 0,7)", "signals": ["presence", "type", "contour"],
            "pair_dice_min": c["min_pair_dice"]}


def add_flags(d: pd.DataFrame, rule: dict) -> pd.DataFrame:
    """Ошибка основной системы (копия 0) и «проверьте» по правилу rule (признаки — src/confidence.py)."""
    d = d.copy()
    sig = conf_signals(d, rule)
    for s in rule["signals"]:
        d[f"chk_{s}"] = sig[s].astype(bool)
    d["check"] = sig.any(axis=1).astype(bool)
    type_eval = d["has_tumor"] & d["present_0"] & d["cls"].isin(TUMOR_TYPES)
    d["err_presence"] = d["present_0"] != d["has_tumor"]
    d["err_type"] = type_eval & (d["type_0"] != d["cls"])
    d["error"] = d["err_presence"] | d["err_type"]
    return d


def prop(d: pd.DataFrame, col: str, boot: dict) -> dict:
    """Доля True в col: k из n, интервал — бутстреп по пациентам (и Уилсон для справки)."""
    n, k = len(d), int(d[col].sum()) if len(d) else 0
    lo, hi = boot_ci(d, lambda x: x[col].mean(), boot["n"], boot["seed"]) if n else (np.nan, np.nan)
    wl, wh = wilson(k, n) if n else (np.nan, np.nan)
    return {"k": k, "n": n, "rate": k / n if n else np.nan, "lo": lo, "hi": hi, "wilson_lo": wl, "wilson_hi": wh,
            "patients": int(d["group"].nunique()) if n else 0}


def per_patient_dice(d: pd.DataFrame, boot: dict, col: str = "dice_gt_0") -> dict:
    x = d[d[col].notna()]
    if x.empty:
        return {"n_slices": 0}
    pp = x.groupby("group")[col].mean().rename("v").reset_index()
    lo, hi = boot_ci(pp, lambda y: y["v"].mean(), boot["n"], boot["seed"])
    return {"slices_mean": float(x[col].mean()), "patients_mean": float(pp["v"].mean()), "lo": lo, "hi": hi,
            "n_slices": int(len(x)), "n_patients": int(len(pp))}


def confidence(d: pd.DataFrame, c: dict) -> dict:
    n, nf, ne = len(d), int(d["check"].sum()), int(d["error"].sum())
    e_f = d.loc[d["check"], "error"].mean() if nf else np.nan
    e_u = d.loc[~d["check"], "error"].mean() if n - nf else np.nan
    ratio = (e_f / e_u) if (e_u and e_u > 0) else (np.inf if e_f and e_f > 0 else np.nan)
    caught = int(d.loc[d["check"], "error"].sum()) / ne if ne else np.nan
    flagged = nf / n if n else np.nan
    ok = bool(ne and ratio >= c["min_error_ratio"] and caught >= c["min_errors_caught"] and flagged <= c["max_flagged"])
    return {"n": n, "errors": ne, "flagged": nf, "flagged_share": flagged, "err_rate_flagged": e_f, "err_rate_confident": e_u,
            "ratio": ratio, "errors_caught": caught, "helps": ok,
            "why_flagged": {k: int(d[k].sum()) for k in d.columns if k.startswith("chk_")}}


def glioma_share(v: pd.DataFrame) -> dict:
    """П3: доля менингиом и опухолей гипофиза нашего val с типом «глиома» (по каждой копии и среднее)."""
    x = v[v["cls"].isin(["meningioma", "pituitary"])]
    per = [float((x[f"answer_{c}"] == "glioma").mean()) for c in range(NC)]
    return {"n": int(len(x)), "per_copy": per, "mean": float(np.mean(per)), "k_copy0": int((x["answer_0"] == "glioma").sum())}


def compute(pred: pd.DataFrame, rules: dict, conf_rule: dict | None = None) -> dict:
    boot, c = rules["bootstrap"], rules["confidence"]
    rule = conf_rule or default_rule(c)
    d = add_flags(pred, rule)
    M: dict = {"confidence_rule": rule}
    for name in ("val", "test"):
        v = d[d["set"] == name]
        if v.empty:
            continue
        tu, nt = v[v["has_tumor"]], v[~v["has_tumor"]]
        tu = tu.assign(hit=tu["present_0"])
        nt = nt.assign(err=nt["present_0"])
        found = tu[tu["present_0"]]
        M[name] = {"recall": prop(tu, "hit", boot), "no_tumor_errors": prop(nt, "err", boot),
                   "type_correct": prop(found.assign(ok=found["type_0"] == found["cls"]), "ok", boot),
                   "recall_copies": [float(tu[f"present_{k}"].mean()) for k in range(NC)],
                   "dice": per_patient_dice(tu, boot), "glioma_share": glioma_share(v), "confidence": confidence(v, c),
                   "recall_by_source": {s: prop(g, "hit", boot) for s, g in tu.groupby("source")},
                   "no_tumor_errors_by_mode": ({m: prop(g, "err", boot) for m, g in nt.groupby("nt_mode")}
                                               if "nt_mode" in nt.columns else {})}
    e = d[d["set"] == "ext"]
    if len(e) and (e["split"] == "check").any():
        mc = rules["main_check"]
        main = e[(e["split"] == mc["split"]) & (~e["has_tumor"]) & (e["mode"] == mc["mode"])].assign(err=lambda x: x["present_0"])
        M["main_check"] = {"all": prop(main, "err", boot),
                           "by_source": {s: prop(g, "err", boot) for s, g in main.groupby("source")},
                           "by_plane": {p: prop(g, "err", boot) for p, g in main.groupby("plane")},
                           "ms_lesion": {str(k): prop(g, "err", boot) for k, g in main[main["source"] == "ms"].groupby("ms_lesion")},
                           "copies": [float(main[f"present_{k}"].mean()) for k in range(NC)],
                           "confidence": confidence(main, c)}
        unk = e[(e["split"] == "check") & (~e["has_tumor"]) & (e["mode"] != "t1c")].assign(err=lambda x: x["present_0"])
        M["vs_unknown_mode"] = prop(unk, "err", boot) if len(unk) else {}
        vs_t = e[(e["source"] == "vs") & e["has_tumor"]]
        M["schwannoma"] = {"found": prop(vs_t.assign(hit=vs_t["present_0"]), "hit", boot),
                           "types": vs_t.loc[vs_t["present_0"], "type_0"].value_counts().to_dict(),
                           "dice": per_patient_dice(vs_t, boot)}
    if len(e):
        for part in ("holdout", "val"):
            h = e[e["split"] == part]
            if h.empty:
                continue
            res = {}
            for s, g in h.groupby("source"):
                nt, tu = g[~g["has_tumor"]].assign(err=lambda x: x["present_0"]), g[g["has_tumor"]]
                found = tu[tu["present_0"]]
                res[s] = {"no_tumor_errors": prop(nt, "err", boot), "found": prop(tu.assign(hit=tu["present_0"]), "hit", boot),
                          "type_correct": prop(found.assign(ok=found["type_0"] == found["cls"]), "ok", boot),
                          "dice": per_patient_dice(tu, boot),
                          "by_plane": {p: {"no_tumor_errors": prop(gg[~gg["has_tumor"]].assign(err=lambda x: x["present_0"]), "err", boot),
                                           "found": prop(gg[gg["has_tumor"]].assign(hit=lambda x: x["present_0"]), "hit", boot)}
                                       for p, gg in g.groupby("plane")}}
            res["confidence"] = confidence(h, c)
            M[f"ext_{part}"] = res
    # режимы: T1 с контрастом из ext + T1/T2/FLAIR из modes, только UPENN и склероз и только аксиальные срезы —
    # T2 и FLAIR у UPENN сняты толстыми аксиальными срезами, их корональные и сагиттальные картинки — размытый пересчет (P11-7)
    em = d[(d["set"].isin(["ext", "modes"])) & d["source"].isin(["upenn", "ms"]) & (d["plane"] == "ax")]
    if len(em) and (d["set"] == "modes").any():
        res = {}
        for (s, m), g in em.groupby(["source", "mode"]):
            nt, tu = g[~g["has_tumor"]].assign(err=lambda x: x["present_0"]), g[g["has_tumor"]]
            found = tu[tu["present_0"]]
            res[f"{s}|{m}"] = {"no_tumor_errors": prop(nt, "err", boot),
                               "found": prop(tu.assign(hit=tu["present_0"]), "hit", boot) if len(tu) else {},
                               "type_correct": prop(found.assign(ok=found["type_0"] == found["cls"]), "ok", boot) if len(found) else {},
                               "dice": per_patient_dice(tu, boot) if len(tu) else {},
                               "flagged_share": float(g["check"].mean())}
        M["modes"] = res
        M["modes_parts"] = sorted(f"{s}:{sp}" for s, sp in em[["source", "split"]].drop_duplicates().itertuples(index=False))
    return M, d


# ---------------------------------------------------------------- отчет


def f(r: dict, pct: bool = True) -> str:
    if not r or not r.get("n"):
        return "—"
    return f"{r['k']} из {r['n']} = {r['rate']:.1%} [{r['lo']:.1%}; {r['hi']:.1%}]" if pct else f"{r['k']} из {r['n']}"


def fd(r: dict) -> str:
    if not r or not r.get("n_slices"):
        return "—"
    return f"{r['patients_mean']:.3f} [{r['lo']:.3f}; {r['hi']:.3f}] ({r['n_patients']} пац.; по срезам {r['slices_mean']:.3f})"


def fc(r: dict) -> str:
    if not r.get("n"):
        return "—"
    why = ", ".join(f"{SIG_RU.get(k, k)} {v}" for k, v in r["why_flagged"].items())
    return (f"«проверьте» на {r['flagged']} из {r['n']} ({r['flagged_share']:.1%}); ошибок {r['errors']}; доля ошибок среди «проверьте» "
            f"{r['err_rate_flagged']:.1%}, среди «уверенно» {r['err_rate_confident']:.1%} (в {r['ratio']:.1f} раза); "
            f"поймано ошибок {r['errors_caught']:.0%}; причины: {why} → "
            f"**{'помогает' if r['helps'] else 'не помогает'}** по П5")


def report(M: dict, tag: str, info: list[dict], path: Path, system_name: str = "`models/final`") -> None:
    L = [f"# Замер системы «{tag}» по правилам П1–П8", "",
         f"`python -m src.eval_external --tag {tag}`, {time.strftime('%Y-%m-%d %H:%M')}. Правила — `configs/eval_external.yaml` "
         "(закоммичены до замера). Основная система — копия 0 = " + system_name + ": " +
         "; ".join(f"копия {i}: {x['classifier']} + {x['segmenter']}" for i, x in enumerate(info)) + ". "
         "Интервалы — бутстреп по пациентам. Dice — в координатах модели (256×256). "
         f"Пометка «проверьте»: {M.get('confidence_rule', {}).get('name', 'A')}.", ""]
    if "main_check" in M:
        m = M["main_check"]
        L += ["## П1. Главная проверка: срезы T1 с контрастом без опухоли из независимых источников", "",
              f"Ошибка — ответ «опухоль есть». **Все: {f(m['all'])}**, пациентов {m['all']['patients']}.", ""]
        L += [f"- {SRC_RU.get(s, s)}: {f(r)}" for s, r in m["by_source"].items()]
        L += [f"- плоскость {PLANE_RU.get(p, p)}: {f(r)}" for p, r in m["by_plane"].items()]
        L += [f"- склероз, срез {'с очагом' if k == 'True' else 'без очага'}: {f(r)}" for k, r in m["ms_lesion"].items()]
        L += [f"- копии 0 / 1 / 2: " + " / ".join(f"{x:.1%}" for x in m["copies"])]
        if M.get("vs_unknown_mode"):
            L += [f"- шванномы, режим не установлен (в главную меру не входят): {f(M['vs_unknown_mode'])}"]
        s = M["schwannoma"]
        L += ["", f"Срезы со шванномой (незнакомая опухоль): опухоль найдена {f(s['found'])}; тип: {s['types']}; Dice {fd(s['dice'])}.", ""]
    for part, title in (("ext_holdout", "Отложенная часть UPENN-GBM и BraTS-MEN-RT (тот же источник, что будет в обучении)"),
                        ("ext_val", "Внешний val UPENN-GBM и BraTS-MEN-RT (для подбора)")):
        if part not in M:
            continue
        L += [f"## {title}", ""]
        for s, r in M[part].items():
            if s == "confidence":
                continue
            L += [f"**{SRC_RU.get(s, s)}**", "", f"- без опухоли, ошибок: {f(r['no_tumor_errors'])}",
                  f"- с опухолью, найдено: {f(r['found'])}; тип верен у найденных: {f(r['type_correct'])}",
                  f"- Dice ядра: {fd(r['dice'])}"]
            L += [f"- {PLANE_RU.get(p, p)}: без опухоли ошибок {f(x['no_tumor_errors'])}; найдено {f(x['found'])}" for p, x in r["by_plane"].items()]
            L += [""]
        L += [f"П5: {fc(M[part]['confidence'])}", ""]
    for name, title in (("val", "Наш val (525 снимков)"),
                        ("test", "Старый test (582 снимка; уже просмотрен при разборе ошибок — только «рядом»)")):
        if name not in M:
            continue
        v = M[name]
        g = v["glioma_share"]
        L += [f"## {title}", "", f"- найдено опухолей: {f(v['recall'])}; копии: " + " / ".join(f"{x:.2%}" for x in v["recall_copies"])]
        L += [f"  - источник {s}: {f(r)}" for s, r in v.get("recall_by_source", {}).items()]
        L += [f"- «нет опухоли», ошибок: {f(v['no_tumor_errors'])}"]
        L += [f"  - режим «{m}»: {f(r)}" for m, r in v.get("no_tumor_errors_by_mode", {}).items()]
        main = name == "val"
        L += [f"- тип верен у найденных: {f(v['type_correct'])}", f"- Dice: {fd(v['dice'])}",
              f"- {'**П3**' if main else 'П3 (справка)'}: менингиом и гипофиза с типом «глиома» — копия 0: {g['k_copy0']} из {g['n']}; "
              "по копиям " + " / ".join(f"{x:.2%}" for x in g["per_copy"]) + f"; среднее {g['mean']:.2%}",
              f"- {'**П5**' if main else 'П5 (справка)'}: {fc(v['confidence'])}", ""]
    if "main_check" in M:
        L += [f"П5 на главной проверке: {fc(M['main_check']['confidence'])}", ""]
    if "modes" in M:
        L += [f"## По режимам МРТ ({', '.join(M.get('modes_parts', []))}; одни и те же места срезов; только аксиальные — P11-7)", "",
              "| набор | режим | без опухоли: ошибок | с опухолью: найдено | тип верен | Dice ядра | «проверьте» |", "|---|---|---|---|---|---|---|"]
        for key, r in M["modes"].items():
            s, m = key.split("|")
            L += [f"| {SRC_RU.get(s, s)} | {MODE_RU.get(m, m)} | {f(r['no_tumor_errors'])} | {f(r['found'])} | {f(r['type_correct'])} | "
                  f"{fd(r['dice'])} | {r['flagged_share']:.1%} |"]
        L += [""]
    if "test" not in M:
        L += ["Старый test (просмотрен при разборе ошибок), `models/final`: найдено 528 из 533 опухолей, верно «нет опухоли» 48 из 49, "
              "тип верен у 443 из 528 — `reports/metrics.md`.", ""]
    path.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True, help="before / after / ... — папка результата")
    ap.add_argument("--rules", type=Path, default=Path("configs/eval_external.yaml"))
    ap.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    ap.add_argument("--sets", default="val,ext,modes")
    ap.add_argument("--system", type=Path, default=None,
                    help="файл системы (presence, copies, check_final); по умолчанию — раздел system правил (models/final)")
    ap.add_argument("--splits", default="val,holdout,check",
                    help="части внешних наборов; val — только промежуточная проверка (отложенная не открывается)")
    ap.add_argument("--ext-table", default="external_slices.csv",
                    help="таблица внешних срезов; свежая проверка круга 4 — external_fresh.csv (с --ext-cache extf)")
    ap.add_argument("--ext-cache", default="ext")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="первые N снимков каждого набора (проверка)")
    ap.add_argument("--report-only", action="store_true", help="пересчитать меры по сохраненным pred_<набор>.csv")
    ap.add_argument("--only-predict", action="store_true",
                    help="только посчитать и сохранить предсказания (на машине с 7 ГБ — по одному набору на процесс)")
    args = ap.parse_args()
    sys.stdout.reconfigure(errors="replace")
    torch.set_num_threads(6)
    rules = yaml.safe_load(args.rules.read_text(encoding="utf-8"))
    system = yaml.safe_load(args.system.read_text(encoding="utf-8")) if args.system else rules["system"]
    splits = tuple(args.splits.split(","))
    if args.system and {"holdout", "check"} & set(splits) and not args.report_only:
        print("ВНИМАНИЕ: открывается отложенная проверка для системы", args.system, "— запишите дату и версию в журнал проекта", flush=True)
    out = Path("reports/eval_external") / args.tag
    out.mkdir(parents=True, exist_ok=True)
    cfg = DataConfig.from_yaml(args.config)
    sets = args.sets.split(",")
    part = {s: out / f"pred_{s}.csv" for s in sets}  # каждый набор — свой файл: после сбоя считаются только недостающие
    info_path = out / "copies.json"
    if not args.report_only:
        if args.limit and any(part[s].exists() for s in sets):
            raise ValueError(f"в {out} уже есть предсказания — для проверки с --limit возьмите другой --tag")
        todo = [s for s in sets if args.limit or not part[s].exists()]
        if todo:
            copies, thr, min_area, info = load_copies(system, torch.device("cpu"))
            info_path.write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
            for table, cache in tables(cfg, todo, splits, (args.ext_table, args.ext_cache)):
                if args.limit:
                    table = table.groupby("source", group_keys=False).head(args.limit)
                print(f"{table['set'].iloc[0]}: {len(table)} снимков", flush=True)
                predict(table, cache, copies, thr, min_area, args.batch_size).to_csv(
                    part[table["set"].iloc[0]], index=False, encoding="utf-8")
                gc.collect()
        if args.only_predict:
            return
    missing = [s for s in sets if not part[s].exists()]
    if missing:
        raise FileNotFoundError(f"нет предсказаний для {missing} — запустите без --report-only")
    pred = typed(pd.concat([pd.read_csv(part[s], keep_default_na=False, low_memory=False) for s in sets], ignore_index=True))
    pred.to_csv(out / "predictions.csv", index=False, encoding="utf-8")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    conf_rule = (yaml.safe_load(Path(system["confidence"]).read_text(encoding="utf-8"))["rule"]
                 if system.get("confidence") else None)
    M, _ = compute(pred, rules, conf_rule)
    (out / "metrics.json").write_text(json.dumps({"tag": args.tag, "copies": info, **M}, ensure_ascii=False, indent=1, default=float),
                                      encoding="utf-8")
    report(M, args.tag, info, out / "report.md", system.get("name", "`models/final`"))


if __name__ == "__main__":
    main()
