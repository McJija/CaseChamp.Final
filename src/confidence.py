"""Пометка «проверьте» (оценка уверенности системы, П5) — варианты и выбор только на val.

Ошибка основной системы (копия 0, seed 42) — неверный ответ «есть / нет» или неверный тип найденной опухоли.
Признаки по трем копиям системы (seed 42, 43, 44) и основной копии:
  presence — копии расходятся в ответе «есть / нет»;
  type     — все три видят опухоль, но называют разный тип;
  contour  — хоть одна копия нашла область, а средний попарный Dice масок копий < pair_dice_min;
  models   — у основной копии модели расходятся: модель типа по самому вероятному классу и модель контура
             (площадь > 50 пикселей) отвечают по-разному (это прежний флаг «требует внимания врача»);
  type_prob — основная копия видит опухоль, а доля самого вероятного типа среди трех типов < type_prob_min.

Варианты перечислены в CANDIDATES до просмотра результатов на val. Выбор (записан до запуска, 2026-11-13):
вариант проходит, если на нашем val И на внешнем val одновременно выполнены условия П5 (configs/eval_external.yaml:
ошибок среди «проверьте» не меньше чем в 3 раза больше, чем среди «уверенно»; поймано ≥ 50% ошибок; помечено ≤ 25%).
Из прошедших — больше всего пойманных ошибок на обоих val вместе, при равенстве — меньше помеченных, затем меньше
признаков. Если не проходит ни один — сначала варианты, где помечено ≤ 25% и на нашем, и на внешнем val (нагрузка
на врача — жесткое ограничение), из них — больше пойманных ошибок на обоих val; если таких нет — вариант с наименьшей
долей помеченных (большей из двух val); в отчете и заключении пишется, что
П5 на val не выполнено. (Запасной порядок уточнен 2026-11-13 после просмотра таблицы для нынешней системы и до
запуска на новой: прежний «больше пойманных» выбирал вариант, помечающий 60% снимков.)
Отложенная проверка (holdout, check) здесь не используется.

    python -m src.confidence --pred reports/eval_external/run3_val/predictions.csv --out configs/confidence_run3.yaml
Результат: --out (выбранный вариант), reports/run3/confidence_val.md.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

TYPES = ["glioma", "meningioma", "pituitary"]
NC = 3

CANDIDATES = [
    {"name": "A: прежний (есть/нет, тип, контур < 0,7)", "signals": ["presence", "type", "contour"], "pair_dice_min": 0.7},
    {"name": "B: копии расходятся (есть/нет, тип)", "signals": ["presence", "type"]},
    {"name": "C: B + модели расходятся", "signals": ["presence", "type", "models"]},
    *[{"name": f"D{t}: B + тип неуверенный (< {t})", "signals": ["presence", "type", "type_prob"], "type_prob_min": t}
      for t in (0.6, 0.7, 0.8, 0.9)],
    *[{"name": f"E{t}: C + тип неуверенный (< {t})", "signals": ["presence", "type", "models", "type_prob"], "type_prob_min": t}
      for t in (0.6, 0.7, 0.8, 0.9)],
    {"name": "F: B + контур < 0,5", "signals": ["presence", "type", "contour"], "pair_dice_min": 0.5},
]


def _bool(s: pd.Series) -> pd.Series:
    return s if s.dtype == bool else s.astype(str).str.lower().eq("true")


def signals(d: pd.DataFrame, rule: dict) -> pd.DataFrame:
    """Каждый признак — столбец True/False."""
    pres = pd.concat([_bool(d[f"present_{c}"]) for c in range(NC)], axis=1)
    types = d[[f"type_{c}" for c in range(NC)]]
    found = pd.concat([_bool(d[f"found_{c}"]) for c in range(NC)], axis=1)
    out = pd.DataFrame(index=d.index)
    for s in rule["signals"]:
        if s == "presence":
            out[s] = pres.nunique(axis=1) > 1
        elif s == "type":
            out[s] = pres.all(axis=1) & (types.nunique(axis=1) > 1)
        elif s == "contour":
            out[s] = found.any(axis=1) & (pd.to_numeric(d["pair_dice"], errors="coerce") < rule["pair_dice_min"])
        elif s == "models":
            out[s] = (d["cls_argmax_0"] != "no_tumor") != found.iloc[:, 0]
        elif s == "type_prob":
            p = d[[f"p_{t}" for t in TYPES]].astype(float).to_numpy()
            share = p.max(1) / np.clip(p.sum(1), 1e-9, None)
            out[s] = pres.iloc[:, 0].to_numpy() & (share < rule["type_prob_min"])
        else:
            raise ValueError(f"неизвестный признак {s}")
    return out


def errors(d: pd.DataFrame) -> pd.Series:
    """Ошибка основной копии: «есть / нет» или тип найденной опухоли (у шванном тип не оценивается)."""
    has_t, pres = _bool(d["has_tumor"]), _bool(d["present_0"])
    typed = has_t & pres & d["cls"].isin(TYPES)
    return (pres != has_t) | (typed & (d["type_0"] != d["cls"]))


def score(check: pd.Series, err: pd.Series, crit: dict) -> dict:
    n, nf, ne = len(check), int(check.sum()), int(err.sum())
    e_f = float(err[check].mean()) if nf else 0.0
    e_u = float(err[~check].mean()) if n - nf else 0.0
    ratio = e_f / e_u if e_u > 0 else (np.inf if e_f > 0 else 0.0)
    caught = int((check & err).sum())
    ok = bool(ne and ratio >= crit["min_error_ratio"] and caught / ne >= crit["min_errors_caught"] and nf / n <= crit["max_flagged"])
    return {"n": n, "errors": ne, "flagged": nf, "flagged_share": nf / n if n else 0.0, "caught": caught,
            "caught_share": caught / ne if ne else 0.0, "err_flagged": e_f, "err_confident": e_u, "ratio": ratio, "passes": ok}


def parts(pred: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Только промежуточная проверка: наш val и внешний val (T1 с контрастом)."""
    ours = pred[pred["set"] == "val"]
    ext = pred[(pred["set"] == "ext") & (pred["split"] == "val")]
    if (pred["set"] == "ext").any() and set(pred.loc[pred["set"] == "ext", "split"]) - {"val"}:
        pass  # holdout и check в таблице есть, но здесь не используются
    return {"наш val": ours, "внешний val": ext, "оба val": pd.concat([ours, ext])}


def fmt(r: dict) -> str:
    return (f"{r['flagged']} из {r['n']} ({r['flagged_share']:.1%}) | {r['caught']} из {r['errors']} ({r['caught_share']:.0%}) | "
            f"{r['err_flagged']:.1%} / {r['err_confident']:.1%} (×{r['ratio']:.1f}) | {'да' if r['passes'] else 'нет'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--rules", type=Path, default=Path("configs/eval_external.yaml"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--report", type=Path, default=Path("reports/run3/confidence_val.md"))
    ap.add_argument("--before", type=Path, default=Path("reports/eval_external/before/predictions.csv"),
                    help="для справки: те же варианты на нынешней системе (только val)")
    args = ap.parse_args()
    crit = yaml.safe_load(args.rules.read_text(encoding="utf-8"))["confidence"]
    pred = pd.read_csv(args.pred, keep_default_na=False, low_memory=False)
    P = parts(pred)
    res = []
    for rule in CANDIDATES:
        r = {"rule": rule}
        for name, d in P.items():
            sig = signals(d, rule)
            r[name] = score(sig.any(axis=1), errors(d), crit)
        r["passes_both"] = r["наш val"]["passes"] and r["внешний val"]["passes"]
        res.append(r)
    light = lambda r: r["наш val"]["flagged_share"] <= crit["max_flagged"] and r["внешний val"]["flagged_share"] <= crit["max_flagged"]
    worst = lambda r: max(r["наш val"]["flagged_share"], r["внешний val"]["flagged_share"])
    order = sorted(res, key=lambda r: (not r["passes_both"], not light(r), -r["оба val"]["caught"] if light(r) else worst(r),
                                       r["оба val"]["flagged"], len(r["rule"]["signals"])))
    best = order[0]

    L = ["# Пометка «проверьте»: варианты на промежуточной проверке (val)", "",
         f"`python -m src.confidence --pred {args.pred.as_posix()}`, {datetime.now():%Y-%m-%d %H:%M}. Варианты и правило выбора записаны в "
         "`src/confidence.py` до запуска. Ошибка — неверный ответ «есть / нет» или тип найденной опухоли у основной копии. "
         "Условия П5: ошибок среди «проверьте» ≥ 3 раз больше, чем среди «уверенно»; поймано ≥ 50% ошибок; помечено ≤ 25%. "
         "В ячейке: помечено | поймано ошибок | доля ошибок среди «проверьте» / «уверенно» (во сколько раз) | П5 выполнено.", ""]
    for title, rr in (("Кандидат", res),):
        L += [f"## {title}", "", "| вариант | наш val | внешний val | оба val |", "|---|---|---|---|"]
        L += [f"| {r['rule']['name']} | {fmt(r['наш val'])} | {fmt(r['внешний val'])} | {fmt(r['оба val'])} |" for r in rr]
    L += ["", f"**Выбран: {best['rule']['name']}** — " + ("П5 выполнено на нашем и внешнем val." if best["passes_both"]
                                                          else "ни один вариант не выполнил П5 на обоих val; выбран лучший по запасному порядку (помечено ≤ 25% на обоих val, затем больше пойманных ошибок) — ограничение пишется в заключении."), ""]
    if args.before.exists():
        b = pd.read_csv(args.before, keep_default_na=False, low_memory=False)
        b = b[(b["set"] == "val") | ((b["set"] == "ext") & (b["split"] == "val"))]
        PB = parts(b)
        L += ["## Для справки: те же варианты на нынешней системе (`models/final` и копии), val", "",
              "| вариант | наш val | внешний val | оба val |", "|---|---|---|---|"]
        for rule in CANDIDATES:
            sc = {n: score(signals(d, rule).any(axis=1), errors(d), crit) for n, d in PB.items()}
            L.append(f"| {rule['name']} | {fmt(sc['наш val'])} | {fmt(sc['внешний val'])} | {fmt(sc['оба val'])} |")
        L.append("")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(L), encoding="utf-8")
    out = {"rule": best["rule"], "chosen_on": "наш val + внешний val", "passes_p5_on_val": best["passes_both"],
           "val": {k: {kk: (float(vv) if isinstance(vv, (float, np.floating)) else vv) for kk, vv in best[k].items()}
                   for k in ("наш val", "внешний val", "оба val")},
           "date": datetime.now().strftime("%Y-%m-%d %H:%M"), "pred": args.pred.as_posix()}
    args.out.write_text("# Пометка «проверьте» (src/confidence.py). Выбрана только на val.\n"
                        + yaml.safe_dump(out, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
