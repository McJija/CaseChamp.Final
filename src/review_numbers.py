"""Числа, которых не было в отчетах сравнения версий, — дополнительные разбивки (2026-11-16).

Только сводные числа из сохраненных предсказаний (reports/eval_external/<tag>/predictions.csv); модели не запускаются,
ничего не подбирается. Посчитано после просмотра результатов — это пояснения к уже сделанным замерам, а не новые проверки.
Функции и бутстреп — те же, что в отчетах (по пациентам, configs/eval_external.yaml: 2000 повторов, seed 42).

    python -m src.review_numbers            # → reports/run3/review_numbers.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.compare_holdout import load, main_check_rows, paired
from src.eval_external import add_flags, confidence
from src.evaluate import boot_ci
from src.metrics import wilson

VERSIONS = {"final": ("before", "fresh_final"), "run3": ("after", "fresh_run3"), "run4": ("after_run4", "fresh_run4")}
LABEL = {"final": "`models/final` (полуфинал)", "run3": "`run3` (финал)", "run4": "`run4`"}
BINS = [0, 20, 30, 40, 60, 1000]
BIN_RU = ["10–20 мм", "20–30 мм", "30–40 мм", "40–60 мм", "дальше 60 мм"]


def pp(x: float) -> str:
    return f"{x * 100:.1f}%"


def kn(k: int, n: int, wil: bool = False) -> str:
    s = f"{k} из {n} ({k / n:.1%})"
    if wil:
        lo, hi = wilson(k, n)
        s += f" [{lo:.1%}; {hi:.1%}]"
    return s


def dpp(diff: float, lo: float, hi: float) -> str:
    return f"{diff * 100:+.1f} п.п. [{lo * 100:+.1f}; {hi * 100:+.1f}]"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rules", type=Path, default=Path("configs/eval_external.yaml"))
    ap.add_argument("--out", type=Path, default=Path("reports/run3/review_numbers.md"))
    args = ap.parse_args()
    sys.stdout.reconfigure(errors="replace")
    rules = yaml.safe_load(args.rules.read_text(encoding="utf-8"))
    boot = rules["bootstrap"]
    P = {k: (load(a), load(b)) for k, (a, b) in VERSIONS.items()}
    both = {k: pd.concat([main_check_rows(x, rules), main_check_rows(y, rules)], ignore_index=True).sort_values("path").reset_index(drop=True)
            for k, (x, y) in P.items()}
    for k in VERSIONS:
        assert (both[k]["path"].to_numpy() == both["final"]["path"].to_numpy()).all(), "у версий разные срезы экзаменов"
    ex = both["final"]
    n_sl, n_pat = len(ex), ex["group"].nunique()

    L = ["# Пояснения к сравнению версий: дополнительные разбивки", "",
         "Только сводные числа из сохраненных предсказаний; модели не запускались, ничего не подбиралось. Посчитано после "
         "просмотра результатов экзаменов, поэтому это пояснения к сделанным замерам, а не новые проверки. Интервалы — бутстреп "
         f"по пациентам ({boot['n']} повторов, seed {boot['seed']}), для долей пациентов и малых счетов — интервал Уилсона. "
         "`python -m src.review_numbers`.", ""]

    # 1. Экзамены: состав
    src_n = ex["source"].value_counts()
    ms = ex[ex["source"] == "ms"]
    L += ["## 1. Из чего состоят экзамены", "",
          f"- Оба экзамена вместе: {n_sl} срезов T1 с контрастом без опухоли у {n_pat} пациентов.",
          f"- Это не здоровые люди: рассеянный склероз (Любляна) — {int(src_n.get('ms', 0))} срезов, из них с очагом склероза "
          f"{int(ms['ms_lesion'].sum())}; шванномы (Лондон) — {int(src_n.get('vs', 0))} срезов ({src_n.get('vs', 0) / n_sl:.0%}), "
          "срезы взяты не ближе 10 мм от опухоли.",
          "- Независимых центров два, но 90% срезов — из одного (Лондон); второй экзамен — только он.", ""]

    # 2. Ложные тревоги: срезы, копии, пациенты
    L += ["## 2. Ложные тревоги: по срезам, по копиям и по пациентам", "",
          "| версия | копия 42 (основная) | копии 42 / 43 / 44 | среднее трех копий | пациентов хотя бы с одной ложной тревогой |",
          "|---|---|---|---|---|"]
    for k, d in both.items():
        copies = [d[f"present_{c}"].mean() for c in range(3)]
        g = d.groupby("group")["present_0"].any()
        L.append(f"| {LABEL[k]} | {kn(int(d['present_0'].sum()), len(d))} | {' / '.join(pp(c) for c in copies)} | "
                 f"{pp(float(np.mean(copies)))} | {kn(int(g.sum()), len(g), wil=True)} |")
    L += ["", "Разница «вторая − первая версия», парный бутстреп по пациентам:", "",
          "| пара | по копии 42 | по среднему трех копий |", "|---|---|---|"]
    for b, a in (("run3", "final"), ("run4", "final"), ("run4", "run3")):
        r = paired(both[a], both[b], boot)
        m = pd.DataFrame({"group": both[a]["group"], "a": both[a][[f"present_{c}" for c in range(3)]].mean(1),
                          "b": both[b][[f"present_{c}" for c in range(3)]].mean(1)})
        fn = lambda x: x["b"].mean() - x["a"].mean()  # noqa: E731
        lo, hi = boot_ci(m, fn, boot["n"], boot["seed"])
        L.append(f"| `{b}` − `{a}` | {dpp(r['diff'], r['lo'], r['hi'])} | {dpp(fn(m), lo, hi)} |")
    d9 = [both["run3"][f"present_{j}"].mean() - both["final"][f"present_{i}"].mean() for i in range(3) for j in range(3)]
    L += ["", f"Девять пар «копия `final` × копия `run3`»: от {min(d9) * 100:+.1f} до {max(d9) * 100:+.1f} п.п. Интервал по пациентам "
          "не учитывает случайность обучения: разница между версиями сопоставима с разбросом между копиями одной версии. "
          "Преимущество `run3` над `run4` по копии 42 (+3,0 п.п.) — удача одной копии; по среднему копий разницы нет.", ""]
    fa3 = both["run3"][both["run3"]["present_0"]]
    L += [f"Кто дал ложную тревогу у `run3`: модель контура нашла область на {int(fa3['found_0'].sum())} из {len(fa3)} срезов, "
          f"на остальных {int((~fa3['found_0']).sum())} сработал только порог модели типа.", ""]

    # 3. Уровень среза
    L += ["## 3. Ложные тревоги зависят от уровня среза (шванномы, оба экзамена)", "",
          "Шваннома лежит у основания черепа, поэтому расстояние среза от опухоли — это и высота среза: рядом — основание "
          "черепа, дальше 60 мм — верх головы. Разбивка сделана после просмотра результатов.", "",
          "| расстояние от опухоли | срезов | " + " | ".join(LABEL[k] for k in VERSIONS) + " |", "|---|---|---|---|---|"]
    tabs = {}
    for k, d in both.items():
        vs = d[d["source"] == "vs"].copy()
        vs["bin"] = pd.cut(pd.to_numeric(vs["dist_mm"], errors="coerce"), BINS, labels=BIN_RU)
        tabs[k] = vs.groupby("bin", observed=True)["present_0"].agg(["sum", "count"])
    for b in BIN_RU:
        n = int(tabs["final"].loc[b, "count"])
        L.append(f"| {b} | {n} | " + " | ".join(pp(tabs[k].loc[b, "sum"] / n) for k in VERSIONS) + " |")
    es = pd.read_csv("data/processed/external_slices.csv", low_memory=False)
    tr = es[(es["split"] == "train") & ~es["has_tumor"].astype(str).str.lower().eq("true") & (es["plane"] == "ax")]
    L += ["", "Почему так может быть: в обучении у внешних наборов (UPENN-GBM, BraTS-MEN-RT) аксиальные срезы ниже 42% высоты "
          "головы отсекались — там стертое лицо (P11-1, `src/build_external_slices.py`, `--ax-min-frac`); на экзаменах этого "
          f"ограничения нет. Аксиальных срезов без опухоли в обучении — {len(tr)}, все выше этой границы. Два объяснения — "
          "«незнакомая больница» и «уровень среза, которого не было в обучении» — этой разбивкой не разделены; второе она "
          "подкрепляет: у `run3` у верха головы ложных тревог почти нет.", ""]

    # 4. Найдено опухолей
    F, R3 = P["final"][0], P["run3"][0]
    L += ["## 4. Найдено опухолей: `run3` против `models/final` (копия 42, парно)", "",
          "| набор | `models/final` | `run3` | разница |", "|---|---|---|---|"]
    for name, sel in (("наш val (порог подбирался здесь)", lambda x: (x["set"] == "val") & x["has_tumor"]),
                      ("тест полуфинала (уже просмотрен)", lambda x: (x["set"] == "test") & x["has_tumor"]),
                      ("UPENN-GBM, отложенные пациенты", lambda x: (x["set"] == "ext") & (x["split"] == "holdout") & (x["source"] == "upenn") & x["has_tumor"]),
                      ("BraTS-MEN-RT, отложенные пациенты", lambda x: (x["set"] == "ext") & (x["split"] == "holdout") & (x["source"] == "menrt") & x["has_tumor"])):
        a, b = F[sel(F)], R3[sel(R3)]
        r = paired(a, b, boot)
        L.append(f"| {name} | {kn(int(a['present_0'].sum()), len(a))} | {kn(int(b['present_0'].sum()), len(b))} | "
                 f"{dpp(r['diff'], r['lo'], r['hi'])} |")
    miss = {k: x[(x["set"] == "test") & x["has_tumor"] & ~x["present_0"]]["cls"].value_counts().to_dict() for k, x in (("final", F), ("run3", R3))}
    L += ["", f"Пропуски на тесте полуфинала по типам: `models/final` — {miss['final']}, `run3` — {miss['run3']}.", "",
          "Условие П2 «пропусков на нашем val не больше чем на 1 п.п. больше» выполняется по построению: порог подбирается на "
          "том же val с требованием найти не меньше 98,5% опухолей (не меньше 462 из 469), а у `models/final` 466 — потеря "
          "не может превысить 0,85 п.п. Там, где порог не подбирался, `run3` теряет больше.", ""]

    # 5. Пометка «проверьте»
    rule = yaml.safe_load(Path("configs/confidence_run3.yaml").read_text(encoding="utf-8"))["rule"]
    x = add_flags(R3, rule)
    hold = (x["set"] == "ext") & (x["split"] == "holdout") & (x["mode"] == "t1c")
    L += ["## 5. Пометка «проверьте» у `run3` (E0.6) по частям", "",
          "| набор | помечено | поймано ошибок | во сколько раз чаще ошибки среди помеченных | помогает по П5 |", "|---|---|---|---|---|"]
    for name, sel in (("наш val (способ выбран здесь из 12 вариантов)", x["set"] == "val"), ("тест полуфинала", x["set"] == "test"),
                      ("отложенные UPENN-GBM + BraTS-MEN-RT", hold), ("— из них UPENN-GBM", hold & (x["source"] == "upenn")),
                      ("— из них BraTS-MEN-RT", hold & (x["source"] == "menrt"))):
        c = confidence(x[sel], rules["confidence"])
        caught = int(round(c["errors_caught"] * c["errors"]))
        L.append(f"| {name} | {kn(c['flagged'], c['n'])} | {kn(caught, c['errors'])} | ×{c['ratio']:.1f} | {'да' if c['helps'] else 'нет'} |")
    t = x[x["set"] == "test"]
    tm = t["has_tumor"] & ~t["present_0"]
    h = x[hold & (x["source"] == "menrt")]
    hm = h["has_tumor"] & ~h["present_0"]
    L += ["", f"Тест полуфинала: помечены {int((tm & t['check']).sum())} из {int(tm.sum())} пропусков и {int((t['err_type'] & t['check']).sum())} "
          f"из {int(t['err_type'].sum())} ошибок типа. BraTS-MEN-RT: помечены {int((hm & h['check']).sum())} из {int(hm.sum())} пропущенных "
          "менингиом. На отложенных пациентах вместе условие «поймано не меньше половины» выполнено ровно на границе (50%).", ""]

    # 6. Режимы
    m = R3[(R3["source"] == "upenn") & (R3["plane"] == "ax") & R3["split"].isin(["val", "holdout"]) & R3["has_tumor"]]
    L += ["## 6. Режимы МРТ у `run3` (глиобластомы UPENN-GBM, аксиальные срезы, val + отложенные пациенты)", "",
          "| режим | найдено опухолей | тип верен среди найденных |", "|---|---|---|"]
    for mode, ru in (("t1c", "T1 с контрастом"), ("t1", "T1 без контраста"), ("t2", "T2"), ("flair", "FLAIR")):
        d = m[m["mode"] == mode]
        f = d[d["present_0"]]
        L.append(f"| {ru} | {kn(int(d['present_0'].sum()), len(d))} | {kn(int((f['type_0'] == f['cls']).sum()), len(f))} |")
    L += ["", f"Пациентов — {m['group'].nunique()}, другие типы опухоли на этих режимах не проверялись. Поэтому на режимах, "
          "отличных от T1 с контрастом, прототип не выдает «опухоли нет» как вывод и помечает тип как ненадежный.", ""]

    # 7. Малые счета
    v = R3[R3["set"] == "val"]
    nt = v[~v["has_tumor"]]
    pool = pd.read_csv("data/processed/pool.csv", keep_default_na=False)[["path", "nt_mode"]]
    nt = nt.drop(columns=[c for c in ("nt_mode",) if c in nt.columns]).merge(pool, on="path", how="left")
    t1 = nt[nt["nt_mode"] == "T1"]
    fv = F[(F["set"] == "val") & ~F["has_tumor"]]
    L += ["## 7. Малые счета на нашем val: интервал Уилсона", "",
          "Бутстреп при нуле событий дает интервал [0; 0] — это не значит, что ошибок не бывает.", "",
          f"- ложные тревоги на снимках «нет опухоли»: `models/final` {kn(int(fv['present_0'].sum()), len(fv), wil=True)}, "
          f"`run3` {kn(int(nt['present_0'].sum()), len(nt), wil=True)};",
          f"- из них режим T1: `run3` {kn(int(t1['present_0'].sum()), len(t1), wil=True)}.", ""]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print("\nотчет:", args.out)


if __name__ == "__main__":
    main()
