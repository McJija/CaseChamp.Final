"""Круг 4: подробное сравнение и решение по П2ф после единственного открытия свежей проверки.

Было — `models/final`; стало — кандидат круга 4 `run4` (`configs/system_run4.yaml`); для справки — `run3`
(не кандидат: уже не прошел П2). Правила записаны до обучения `run4` (2026-11-14):
  - П1ф — ложные тревоги на срезах T1 с контрастом без опухоли 152 новых пациентов VS-SEG (`external_fresh.csv`);
  - П2ф — у кандидата ложных тревог по П1ф меньше не менее чем на 20 п.п. (интервал разницы — парный бутстреп по пациентам),
    найденных опухолей на нашем val меньше не более чем на 1 п.п., П3 — рост «глиомы» не более 2 п.п.;
    и только после подтверждения руководителя проекта по этому сравнению.
По свежей проверке — только сводные числа. Наш val и внешний val — промежуточная проверка (порог и «проверьте» подобраны на них:
числа кандидатов оптимистичны). Примеры снимков — только из val, отбор правилом (`src.compare_holdout.pick_examples`).

Предсказания (каждое — `python -m src.eval_external`):
  - свежая проверка: `--tag fresh_<имя> --sets val,ext --splits check --ext-table external_fresh.csv --ext-cache extf [--system ...]`;
  - val: `before` (models/final), `run3_val`, `run4_val` (`--sets val,ext --splits val --system ...`).

    python -m src.compare_fresh
Результат: reports/run4/fresh_compare.{md,json}, reports/run4/examples_val.png.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

from src.compare_holdout import PLANE_RU, f, fd, load, main_check_rows, masks_for, paired, pick_examples, sheet
from src.dataset import DataConfig
from src.eval_external import SRC_RU, compute, find_run

SYSTEMS = {  # ключ → (подпись, тег свежей проверки, тег val, файл системы; None — models/final из правил)
    "final": ("было: `models/final`", "fresh_final", "before", None),
    "run4": ("стало: кандидат `run4`", "fresh_run4", "run4_val", Path("configs/system_run4.yaml")),
    "run3": ("справка: `run3` (не кандидат)", "fresh_run3", "run3_val", Path("configs/system_run3.yaml")),
}


def with_mode(p: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    p = p.drop(columns=["nt_mode"], errors="ignore").merge(pool, on="path", how="left")
    p["nt_mode"] = p["nt_mode"].fillna("")
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rules", type=Path, default=Path("configs/eval_external.yaml"))
    ap.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    ap.add_argument("--out", type=Path, default=Path("reports/run4/fresh_compare.md"))
    ap.add_argument("--no-examples", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(errors="replace")
    rules = yaml.safe_load(args.rules.read_text(encoding="utf-8"))
    boot, rp = rules["bootstrap"], rules["replacement"]
    pool = pd.read_csv("data/processed/pool.csv", keep_default_na=False)[["path", "nt_mode"]]

    sysd, conf, P, PV, M, MV = {}, {}, {}, {}, {}, {}
    for k, (_, tag, vtag, path) in SYSTEMS.items():
        sysd[k] = yaml.safe_load(path.read_text(encoding="utf-8")) if path else rules["system"]
        conf[k] = yaml.safe_load(Path(sysd[k]["confidence"]).read_text(encoding="utf-8"))["rule"] if sysd[k].get("confidence") else None
        P[k], PV[k] = with_mode(load(tag), pool), with_mode(load(vtag), pool)
        M[k] = compute(P[k], rules, conf[k])[0]
        MV[k] = compute(PV[k], rules, conf[k])[0]
        assert M[k]["main_check"]["all"]["n"] > 0, f"{tag}: нет срезов свежей проверки"
    n_fresh = {k: M[k]["main_check"]["all"]["n"] for k in M}
    assert len(set(n_fresh.values())) == 1, f"разное число срезов свежей проверки: {n_fresh}"

    pr = {k: paired(main_check_rows(P["final"], rules), main_check_rows(P[k], rules), boot) for k in ("run4", "run3")}
    rec = {k: M[k]["val"]["recall"] for k in M}
    drop = {k: -pr[k]["diff"] * 100 for k in pr}
    rdrop = {k: (rec["final"]["rate"] - rec[k]["rate"]) * 100 for k in pr}
    gl = {k: M[k]["val"]["glioma_share"]["mean"] for k in M}
    cond = {k: {"c1": drop[k] >= rp["min_drop_main_check_pp"], "c2": rdrop[k] <= rp["max_recall_drop_val_pp"],
                "c3": (gl[k] - gl["final"]) * 100 <= rules["glioma_shortcut"]["max_rise_pp"]} for k in pr}
    yes = lambda b: "да" if b else "нет"  # noqa: E731
    head = " | ".join(SYSTEMS[k][0] for k in SYSTEMS)

    L = ["# Круг 4: свежая проверка и решение по П2ф", "",
         "Было — `models/final` (модель типа `run1_cls` s42 + модель контура `run1_seg_neg` s42). Стало — кандидат `run4` "
         "(`run3` + выравнивание резкости и шума, R = 224; порог и «проверьте» выбраны только на val). Для справки — `run3` "
         "(уже не прошел П2). Свежая проверка открыта один раз (дата и версия записаны в журнале проекта); по ней — только сводные числа. Интервалы — бутстреп "
         f"по пациентам ({boot['n']} повторов, seed {boot['seed']}). `python -m src.compare_fresh`.", "",
         "## Итог по правилам", "", f"| правило | {head} |", "|---|---|---|---|",
         f"| П1ф: ложная тревога на срезах T1 с контрастом без опухоли (новые пациенты VS-SEG) | "
         + " | ".join(f(M[k]["main_check"]["all"]) for k in SYSTEMS) + " |",
         f"| П2ф.1: меньше не менее чем на {rp['min_drop_main_check_pp']} п.п. (разница, парно) | — | "
         + " | ".join(f"{drop[k]:+.1f} п.п. [{-pr[k]['hi'] * 100:+.1f}; {-pr[k]['lo'] * 100:+.1f}] → {yes(cond[k]['c1'])}" for k in pr) + " |",
         f"| П2ф.2: найдено опухолей на нашем val, ниже не более чем на {rp['max_recall_drop_val_pp']} п.п. | {f(rec['final'])} | "
         + " | ".join(f"{f(rec[k])}; ниже на {rdrop[k]:.2f} п.п. → {yes(cond[k]['c2'])}" for k in pr) + " |",
         f"| П3: менингиомы и гипофиз нашего val с типом «глиома» (среднее трех копий), рост ≤ {rules['glioma_shortcut']['max_rise_pp']} п.п. | "
         f"{gl['final']:.2%} | " + " | ".join(f"{gl[k]:.2%} → {yes(cond[k]['c3'])}" for k in pr) + " |",
         f"| П5: «проверьте» помогает находить ошибки — свежая проверка | "
         + " | ".join(yes(M[k]["main_check"]["confidence"]["helps"]) for k in SYSTEMS) + " |",
         f"| П5: то же — наш val | " + " | ".join(yes(M[k]["val"]["confidence"]["helps"]) for k in SYSTEMS) + " |", ""]
    ok4 = all(cond["run4"].values())
    L += [f"**П2ф для `run4`: {'все условия выполнены' if ok4 else 'не выполнено'}** — "
          + ("замена `models/final` возможна только после подтверждения руководителя проекта по этому сравнению."
             if ok4 else "`models/final` остается (решение о замене все равно за руководителем проекта)."), ""]

    L += ["## Свежая проверка подробно", "", f"Срезов без опухоли {pr['run4']['n']}, пациентов {pr['run4']['patients']}.", "",
          f"| часть | {head} |", "|---|---|---|---|"]
    planes = sorted(set().union(*(M[k]["main_check"]["by_plane"] for k in M)))
    L += [f"| плоскость {PLANE_RU.get(p, p)} | " + " | ".join(f(M[k]["main_check"]["by_plane"].get(p)) for k in SYSTEMS) + " |" for p in planes]
    L += ["| копии 0 / 1 / 2 | " + " | ".join(" / ".join(f"{x:.1%}" for x in M[k]["main_check"]["copies"]) for k in SYSTEMS) + " |",
          "| «проверьте» на срезах без опухоли | " + " | ".join(_fc(M[k]["main_check"]["confidence"]) for k in SYSTEMS) + " |"]
    if any(M[k].get("vs_unknown_mode") for k in M):
        L += ["| режим не установлен (в П1ф не входит) | " + " | ".join(f(M[k].get("vs_unknown_mode")) for k in SYSTEMS) + " |"]
    L += ["", "**Срезы со шванномой (незнакомая опухоль, тип не оценивается):**", "", f"| мера | {head} |", "|---|---|---|---|",
          "| опухоль найдена | " + " | ".join(f(M[k]["schwannoma"]["found"]) for k in SYSTEMS) + " |",
          "| названный тип | " + " | ".join(str(M[k]["schwannoma"]["types"]) for k in SYSTEMS) + " |",
          "| совпадение контура | " + " | ".join(fd(M[k]["schwannoma"]["dice"]) for k in SYSTEMS) + " |", ""]

    L += ["## Наш val (промежуточная проверка)", "", f"| мера | {head} |", "|---|---|---|---|"]
    for lab, key in (("найдено опухолей", "recall"), ("ложная тревога «нет опухоли»", "no_tumor_errors"), ("тип верен у найденных", "type_correct")):
        L.append(f"| {lab} | " + " | ".join(f(M[k]["val"][key]) for k in SYSTEMS) + " |")
    for s in ("figshare", "figshare_crop", "other"):
        L.append(f"| — найдено, источник {s} | " + " | ".join(f(M[k]["val"]["recall_by_source"].get(s)) for k in SYSTEMS) + " |")
    for m in sorted(M["final"]["val"]["no_tumor_errors_by_mode"]):
        L.append(f"| — ложная тревога, режим «{m}» | " + " | ".join(f(M[k]["val"]["no_tumor_errors_by_mode"].get(m)) for k in SYSTEMS) + " |")
    L += [f"| совпадение контура | " + " | ".join(fd(M[k]["val"]["dice"]) for k in SYSTEMS) + " |",
          f"| «проверьте» | " + " | ".join(_fc(M[k]["val"]["confidence"]) for k in SYSTEMS) + " |", ""]

    L += ["## Внешний val (UPENN-GBM и BraTS-MEN-RT; тот же источник, что в обучении `run3` и `run4`)", "", f"| мера | {head} |", "|---|---|---|---|"]
    for s in MV["run4"].get("ext_val", {}):
        if s == "confidence":
            continue
        g = {k: MV[k]["ext_val"][s] for k in SYSTEMS}
        nm = SRC_RU.get(s, s)
        L += [f"| {nm}: ложная тревога | " + " | ".join(f(g[k]["no_tumor_errors"]) for k in SYSTEMS) + " |",
              f"| {nm}: опухоль найдена | " + " | ".join(f(g[k]["found"]) for k in SYSTEMS) + " |",
              f"| {nm}: тип верен у найденных | " + " | ".join(f(g[k]["type_correct"]) for k in SYSTEMS) + " |",
              f"| {nm}: совпадение контура | " + " | ".join(fd(g[k]["dice"]) for k in SYSTEMS) + " |"]
    L += [""]

    if not args.no_examples:
        rows = pick_examples(PV["final"], PV["run4"])
        cfg = DataConfig.from_yaml(args.config)
        cls_seg = {"before": (find_run(rules["system"]["copies"][0]["classifier"]), find_run(rules["system"]["copies"][0]["segmenter"])),
                   "after": (find_run(sysd["run4"]["copies"][0]["classifier"]), find_run(sysd["run4"]["copies"][0]["segmenter"]))}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        png = args.out.parent / "examples_val.png"
        sheet(rows, masks_for(rows, cls_seg, cfg), png)
        L += ["## Примеры верных и ошибочных ответов (только val; было — `models/final`, стало — `run4`)", "",
              f"`{png.as_posix()}` — {len(rows)} снимков; группы (все, если ≤ 3, иначе 3 случайных, seed 42): "
              + "; ".join(sorted({n for n, _ in rows})) + ".", ""]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(L), encoding="utf-8")
    js = {"paired": pr, "drop_pp": drop, "recall_drop_pp": rdrop, "glioma_share": gl,
          "conditions": {k: {c: bool(v) for c, v in d.items()} for k, d in cond.items()}, "metrics": M, "metrics_val": MV}
    args.out.with_suffix(".json").write_text(json.dumps(js, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print("\n".join(L))


def _fc(r: dict) -> str:
    """Пометка «проверьте» по полям `src.eval_external.confidence`."""
    if not r.get("n"):
        return "—"
    caught = r["errors_caught"]
    return (f"помечено {r['flagged']} из {r['n']} ({r['flagged_share']:.1%}), поймано "
            f"{'—' if caught != caught else f'{caught:.0%}'} из {r['errors']} ошибок → {'помогает' if r['helps'] else 'не помогает'}")


if __name__ == "__main__":
    main()
