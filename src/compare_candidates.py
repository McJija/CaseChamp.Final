"""Сравнение трех версий системы на всех проверках сразу: прежняя `models/final`, `run3` (дообучение), `run4` (дообучение +
выравнивание резкости и шума). Нужно для выбора версии прототипа (решение команды 2026-11-15:
сравнить и предварительно взять run3). Правило П2 / П2ф не выполнено ни у `run3`, ни у `run4` — этот отчет его не
пересматривает, а дает полную картину для решения руководителя проекта.

Источники предсказаний (каждое — `python -m src.eval_external`):
  - первый экзамен (склероз и шванномы — check; holdout UPENN-GBM и BraTS-MEN-RT; режимы; старый test) и val:
    `before` (models/final), `after` (run3), `after_run4` (run4);
  - второй экзамен (новые пациенты VS-SEG): `fresh_final`, `fresh_run3`, `fresh_run4`.
По экзаменам — только сводные числа. Наш val и внешний val — промежуточная проверка (порог и «проверьте» кандидатов подобраны
на них — числа кандидатов оптимистичны). Старый test уже просмотрен при разборе ошибок — только «рядом».
Разницы — парный бутстреп по пациентам (одни и те же срезы у всех версий).

    python -m src.compare_candidates
Результат: reports/run4/candidates_compare.{md,json}.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

from src.compare_fresh import _fc, with_mode
from src.compare_holdout import PLANE_RU, f, fd, load, main_check_rows, paired  # noqa: F401
from src.eval_external import SRC_RU, compute, prop

MODE_RU = {"t1c": "T1 с контрастом", "t1": "T1", "t2": "T2", "flair": "FLAIR"}
KEYS = ("final", "run3", "run4")
LABEL = {"final": "прежняя `models/final`", "run3": "`run3` (дообучение)", "run4": "`run4` (дообучение + выравнивание)"}
SYSTEM = {"final": None, "run3": Path("configs/system_run3.yaml"), "run4": Path("configs/system_run4.yaml")}


def dpp(r: dict) -> str:
    """Разница долей «вторая − первая» в п.п. с интервалом."""
    return f"{r['diff'] * 100:+.1f} п.п. [{r['lo'] * 100:+.1f}; {r['hi'] * 100:+.1f}]"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rules", type=Path, default=Path("configs/eval_external.yaml"))
    ap.add_argument("--tags", nargs=3, default=["before", "after", "after_run4"], help="первый экзамен и val: final run3 run4")
    ap.add_argument("--fresh-tags", nargs=3, default=["fresh_final", "fresh_run3", "fresh_run4"], help="второй экзамен: final run3 run4")
    ap.add_argument("--out", type=Path, default=Path("reports/run4/candidates_compare.md"))
    args = ap.parse_args()
    sys.stdout.reconfigure(errors="replace")
    rules = yaml.safe_load(args.rules.read_text(encoding="utf-8"))
    boot = rules["bootstrap"]
    pool = pd.read_csv("data/processed/pool.csv", keep_default_na=False)[["path", "nt_mode"]]

    conf, P, PF, M, MF = {}, {}, {}, {}, {}
    for k, tag, ftag in zip(KEYS, args.tags, args.fresh_tags):
        sysd = yaml.safe_load(SYSTEM[k].read_text(encoding="utf-8")) if SYSTEM[k] else rules["system"]
        conf[k] = yaml.safe_load(Path(sysd["confidence"]).read_text(encoding="utf-8"))["rule"] if sysd.get("confidence") else None
        P[k], PF[k] = with_mode(load(tag), pool), with_mode(load(ftag), pool)
        M[k], MF[k] = compute(P[k], rules, conf[k])[0], compute(PF[k], rules, conf[k])[0]
        for need in ("val", "test", "main_check", "ext_holdout", "modes"):
            assert need in M[k], f"{tag}: нет части {need}"
        assert "main_check" in MF[k], f"{ftag}: нет второго экзамена"
    for part in (lambda m: m["main_check"]["all"]["n"], lambda m: m["test"]["recall"]["n"], lambda m: m["val"]["recall"]["n"]):
        assert len({part(M[k]) for k in KEYS}) == 1, "у версий разное число снимков"
    assert len({MF[k]["main_check"]["all"]["n"] for k in KEYS}) == 1

    # ложные тревоги на независимых источниках: первый, второй экзамен и оба вместе
    ex1 = {k: main_check_rows(P[k], rules) for k in KEYS}
    ex2 = {k: main_check_rows(PF[k], rules) for k in KEYS}
    both = {k: pd.concat([ex1[k], ex2[k]], ignore_index=True) for k in KEYS}
    assert not set(ex1["final"]["group"]) & set(ex2["final"]["group"]), "пациенты экзаменов пересекаются"
    assert not set(ex1["final"]["path"]) & set(ex2["final"]["path"])
    fa = {"первый экзамен (склероз + шванномы)": ex1, "второй экзамен (новые шванномы)": ex2, "оба экзамена вместе": both}
    pairs = (("run3", "final"), ("run4", "final"), ("run4", "run3"))
    fa_rate = {name: {k: prop(d[k].assign(err=d[k]["present_0"]), "err", boot) for k in KEYS} for name, d in fa.items()}
    fa_diff = {name: {f"{b}−{a}": paired(d[a], d[b], boot) for b, a in pairs} for name, d in fa.items()}
    fa_pat = {name: int(d["final"]["group"].nunique()) for name, d in fa.items()}

    def tum(p: pd.DataFrame, sel) -> pd.DataFrame:
        return p[sel(p)]
    found_sets = {
        "наш val (промежуточная проверка)": (P, lambda d: (d["set"] == "val") & d["has_tumor"]),
        "тест полуфинала (уже просмотрен)": (P, lambda d: (d["set"] == "test") & d["has_tumor"]),
        "UPENN-GBM, отложенные пациенты": (P, lambda d: (d["set"] == "ext") & (d["split"] == "holdout") & (d["source"] == "upenn") & d["has_tumor"]),
        "BraTS-MEN-RT, отложенные пациенты": (P, lambda d: (d["set"] == "ext") & (d["split"] == "holdout") & (d["source"] == "menrt") & d["has_tumor"]),
        "шванномы, первый экзамен": (P, lambda d: (d["set"] == "ext") & (d["source"] == "vs") & d["has_tumor"]),
        "шванномы, второй экзамен": (PF, lambda d: (d["set"] == "ext") & (d["source"] == "vs") & d["has_tumor"]),
    }
    found = {}
    for name, (src, sel) in found_sets.items():
        rows = {k: tum(src[k], sel) for k in KEYS}
        found[name] = {"rate": {k: prop(rows[k].assign(hit=rows[k]["present_0"]), "hit", boot) for k in KEYS},
                       "diff": {f"{b}−{a}": paired(rows[a], rows[b], boot) for b, a in pairs}}

    head = " | ".join(LABEL[k] for k in KEYS)
    L = ["# Сравнение версий: прежняя `models/final`, `run3` и `run4`", "",
         "Прежняя — `models/final` (`run1_cls` + `run1_seg_neg`, порог 0,976368, «проверьте» — способ A). `run3` — дообучение на "
         "внешних наборах (порог 0,989005, «проверьте» E0.6). `run4` — то же + выравнивание резкости и шума (R = 224; порог 0,981098, "
         "«проверьте» E0.9). Правило замены (П2 / П2ф) не выполнено ни у `run3`, ни у `run4`; отчет его не пересматривает. "
         "По экзаменам — только сводные числа; первый экзамен для `run4` открыт один раз (дата и версия записаны в журнале проекта). "
         f"Интервалы — бутстреп по пациентам ({boot['n']} повторов, seed {boot['seed']}); разницы — парные (одни и те же срезы). "
         "`python -m src.compare_candidates`.", "",
         "## 1. Ложная тревога на срезах T1 с контрастом без опухоли из незнакомых больниц (главная слабость)", "",
         f"| экзамен | пациентов | {head} |", "|---|---|---|---|---|"]
    L += [f"| {name} | {fa_pat[name]} | " + " | ".join(f(fa_rate[name][k]) for k in KEYS) + " |" for name in fa]
    L += ["", "Разница долей ложных тревог (минус — меньше тревог):", "",
          "| экзамен | `run3` − прежняя | `run4` − прежняя | `run4` − `run3` |", "|---|---|---|---|"]
    L += [f"| {name} | " + " | ".join(dpp(fa_diff[name][f'{b}−{a}']) for b, a in pairs) + " |" for name in fa]
    L += ["", "| часть | " + head + " |", "|---|---|---|---|"]
    L += [f"| первый экзамен: {SRC_RU.get(s, s)} | " + " | ".join(f(M[k]["main_check"]["by_source"].get(s)) for k in KEYS) + " |"
          for s in M["final"]["main_check"]["by_source"]]
    L += [f"| первый экзамен: склероз, срез {'с очагом' if s == 'True' else 'без очага'} | "
          + " | ".join(f(M[k]["main_check"]["ms_lesion"].get(s)) for k in KEYS) + " |" for s in M["final"]["main_check"]["ms_lesion"]]
    L += [f"| копии 0 / 1 / 2, {lab} | " + " | ".join(" / ".join(f"{x:.1%}" for x in mm[k]["main_check"]["copies"]) for k in KEYS) + " |"
          for lab, mm in (("первый экзамен", M), ("второй экзамен", MF))]
    L += [""]

    L += ["## 2. Опухоль найдена", "", f"| набор | {head} | `run4` − `run3` |", "|---|---|---|---|---|"]
    L += [f"| {name} | " + " | ".join(f(v["rate"][k]) for k in KEYS) + f" | {dpp(v['diff']['run4−run3'])} |" for name, v in found.items()]
    L += ["", "| набор: найдено по источнику | " + head + " |", "|---|---|---|---|"]
    for part, lab in (("val", "наш val"), ("test", "тест полуфинала")):
        L += [f"| {lab}, {s} | " + " | ".join(f(M[k][part]["recall_by_source"].get(s)) for k in KEYS) + " |"
              for s in ("figshare", "figshare_crop", "other")]
    L += [""]

    L += ["## 3. Ложная тревога на остальных наборах", "", f"| набор | {head} |", "|---|---|---|---|"]
    for part, lab in (("val", "наш val"), ("test", "тест полуфинала")):
        L += [f"| {lab}: все «нет опухоли» | " + " | ".join(f(M[k][part]["no_tumor_errors"]) for k in KEYS) + " |"]
        L += [f"| {lab}: режим «{m}» | " + " | ".join(f(M[k][part]["no_tumor_errors_by_mode"].get(m)) for k in KEYS) + " |"
              for m in sorted(M["final"][part]["no_tumor_errors_by_mode"])]
    L += [f"| {SRC_RU.get(s, s)}, отложенные пациенты | " + " | ".join(f(M[k]["ext_holdout"][s]["no_tumor_errors"]) for k in KEYS) + " |"
          for s in ("upenn", "menrt")]
    L += [""]

    L += ["## 4. Тип опухоли верен у найденных", "", f"| набор | {head} |", "|---|---|---|---|"]
    L += [f"| {lab} | " + " | ".join(f(M[k][part]["type_correct"]) for k in KEYS) + " |" for part, lab in (("val", "наш val"), ("test", "тест полуфинала"))]
    L += [f"| {SRC_RU.get(s, s)}, отложенные пациенты | " + " | ".join(f(M[k]["ext_holdout"][s]["type_correct"]) for k in KEYS) + " |"
          for s in ("upenn", "menrt")]
    L += [f"| П3: менингиомы и гипофиз нашего val с типом «глиома» (среднее трех копий) | "
          + " | ".join(f"{M[k]['val']['glioma_share']['mean']:.2%}" for k in KEYS) + " |", ""]

    L += ["## 5. Совпадение контура с обводкой (среднее по пациентам)", "", f"| набор | {head} |", "|---|---|---|---|"]
    L += [f"| {lab} | " + " | ".join(fd(M[k][part]["dice"]) for k in KEYS) + " |" for part, lab in (("val", "наш val"), ("test", "тест полуфинала"))]
    L += [f"| {SRC_RU.get(s, s)}, отложенные пациенты | " + " | ".join(fd(M[k]["ext_holdout"][s]["dice"]) for k in KEYS) + " |"
          for s in ("upenn", "menrt")]
    L += [f"| шванномы, {lab} | " + " | ".join(fd(mm[k]["schwannoma"]["dice"]) for k in KEYS) + " |"
          for lab, mm in (("первый экзамен", M), ("второй экзамен", MF))]
    L += [""]

    L += ["## 6. Пометка «проверьте» (П5: помечено ≤ 25%, поймано ≥ 50% ошибок, ошибок среди помеченных в 3 раза больше)", "",
          f"| набор | {head} |", "|---|---|---|---|"]
    L += [f"| {lab} | " + " | ".join(_fc(get(k)) for k in KEYS) + " |" for lab, get in (
        ("наш val (способ подобран здесь)", lambda k: M[k]["val"]["confidence"]),
        ("тест полуфинала", lambda k: M[k]["test"]["confidence"]),
        ("отложенные UPENN-GBM и BraTS-MEN-RT", lambda k: M[k]["ext_holdout"]["confidence"]),
        ("первый экзамен (без опухоли)", lambda k: M[k]["main_check"]["confidence"]),
        ("второй экзамен (без опухоли)", lambda k: MF[k]["main_check"]["confidence"]))]
    L += [""]

    L += ["## 7. По режимам МРТ (аксиальные срезы; склероз — первый экзамен, UPENN — val и отложенные вместе)", "",
          "| набор | режим | ложная тревога: " + " / ".join(f"`{k}`" for k in KEYS) + " | найдено: " + " / ".join(f"`{k}`" for k in KEYS) + " |",
          "|---|---|---|---|"]
    pct = lambda r: "—" if not r or not r.get("n") else f"{r['rate']:.1%}"  # noqa: E731
    for key in M["final"]["modes"]:
        s, m = key.split("|")
        g = {k: M[k]["modes"].get(key, {}) for k in KEYS}
        L.append(f"| {SRC_RU.get(s, s)} | {MODE_RU.get(m, m)} | " + " / ".join(pct(g[k].get("no_tumor_errors")) for k in KEYS)
                 + " | " + " / ".join(pct(g[k].get("found")) for k in KEYS) + " |")
    L += [""]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(L), encoding="utf-8")
    js = {"false_alarm": fa_rate, "false_alarm_diff": fa_diff, "found": found, "metrics_exam1_val_test": M, "metrics_exam2": MF}
    args.out.with_suffix(".json").write_text(json.dumps(js, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
