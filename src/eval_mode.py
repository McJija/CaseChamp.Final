"""П6 — определитель режима МРТ (`run3_mode`, seed 42) на проверочных наборах.

Правило (configs/eval_external.yaml, mode_detector; записано 2026-11-12): автоматически определенный «T1 с контрастом»
снимает предупреждение о перепроверке, только если среди снимков, названных так, не меньше 95% действительно
T1 с контрастом — на каждом наборе, где это можно посчитать. Иначе предупреждение показывается всегда, если режим
не выбрал врач.

Наборы (определитель учился только на аксиальных срезах UPENN-GBM train, эпоха — по UPENN val):
  - `upenn_val`     — аксиальные срезы UPENN val, 4 режима (промежуточная проверка, для справки);
  - `upenn_holdout` — аксиальные срезы UPENN holdout, 4 режима (отложенная, тот же источник, что в обучении);
  - `ms_check`      — склероз (Любляна), аксиальные, 4 режима (отложенная, независимый источник);
  - `case`          — снимки кейса класса «нет опухоли», у которых режим проверяющие отметили единогласно
                      (`reports/doctor/no_tumor_review/marks_merged.csv`: одно значение без «?», без расхождения;
                      T1 с контрастом, T1, T2, FLAIR; снимки «КТ» — отдельно, в меру не входят: это не режим МРТ);
                      определитель эти снимки не видел; препроцессинг — тот же `preprocess` из исходного JPG;
  - `figshare`      — срезы Figshare (все T1 с контрастом, 3064, все части pool): точность «T1 с контрастом» здесь
                      всегда 100%, считается только доля снимков, с которых предупреждение снимается.
Мера на наборе: точность ответа «T1 с контрастом» (k из n, интервал Уилсона и бутстреп по пациентам) и доля
настоящих T1 с контрастом, названных так. П6 выполнено, если точность ≥ 0,95 на каждом наборе, где есть снимки
других режимов (upenn_holdout, ms_check, case). По отложенным наборам — только сводные числа.

    python -m src.eval_mode --tag after                 # все наборы
    python -m src.eval_mode --tag dev --sets upenn_val  # проверка кода: должно совпасть с val_predictions.csv обучения
Результат: reports/eval_external/<tag>/{mode_pred.csv, mode_check.json, mode_check.md}.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.dataset import DataConfig, PreprocCache, to_tensor
from src.evaluate import boot_ci
from src.io_utils import read_image
from src.metrics import wilson
from src.preprocess import preprocess
from src.train_aux import MODES, build

MODE_RU = {"t1c": "T1 с контрастом", "t1": "T1", "t2": "T2", "flair": "FLAIR", "ct": "КТ"}
REVIEW_MODE = {"T1+К": "t1c", "T1": "t1", "T2": "t2", "FLAIR": "flair", "КТ": "ct"}
SET_RU = {"upenn_val": "UPENN-GBM val (промежуточная, для справки)", "upenn_holdout": "UPENN-GBM holdout (тот же источник, что в обучении)",
          "ms_check": "склероз, Любляна (независимый источник)", "case": "снимки кейса «нет опухоли», режим отмечен единогласно",
          "figshare": "срезы Figshare (все T1 с контрастом)"}
RULE_SETS = ("upenn_holdout", "ms_check", "case")
ALL_SETS = ("upenn_val", "upenn_holdout", "ms_check", "case", "figshare")


def load_model(pattern: str, models: Path, device) -> tuple[torch.nn.Module, str]:
    runs = sorted(models.glob(pattern))
    if len(runs) != 1:
        raise ValueError(f"шаблону {pattern!r} соответствует {len(runs)} папок, нужна одна")
    ck = torch.load(runs[0] / "best.pt", map_location=device, weights_only=False)
    if ck["kind"] != "mode" or list(ck["classes"]) != MODES:
        raise ValueError(f"{runs[0]} — не определитель режима")
    model = build("mode", ck["encoder"], False)
    model.load_state_dict(ck["model"])
    return model.eval().to(device), runs[0].name


def ext_set(cfg: DataConfig, source: str, split: str) -> tuple[pd.DataFrame, list]:
    rows, caches = [], []
    for name in ("ext", "extm"):
        c = PreprocCache(cfg.cache_dir, name, cfg.size)
        t = c.table
        t = t[(t["source"] == source) & (t["split"] == split) & (t["plane"] == "ax") & t["mode"].isin(MODES)]
        rows.append(t[["path", "mode", "patient", "group", "plane"]].assign(cache=len(caches)))
        caches.append(c)
    return pd.concat(rows, ignore_index=True), caches


def case_set(cfg: DataConfig, marks: Path) -> pd.DataFrame:
    m = pd.read_csv(marks, keep_default_na=False)
    one = m["mode_all"].isin(list(REVIEW_MODE)) & (m["mode_agree"] != "нет")  # одно значение, без «?», без расхождения
    x = m[one]
    return pd.DataFrame({"path": x["path"], "mode": x["mode_all"].map(REVIEW_MODE), "patient": x["patient_group_id"],
                         "group": x["patient_group_id"], "plane": "", "n_reviewers": x["n_reviewers"], "split_case": x["split"]})


@torch.no_grad()
def run(model, images, bs: int = 32) -> np.ndarray:
    out = []
    for i in range(0, len(images), bs):
        x = torch.stack([to_tensor(a) for a in images[i:i + bs]])
        out.append(torch.softmax(model(x).float(), 1).numpy())
    return np.concatenate(out)


def predict_set(name: str, cfg: DataConfig, model, marks: Path) -> pd.DataFrame:
    if name in ("upenn_val", "upenn_holdout", "ms_check"):
        source, split = {"upenn_val": ("upenn", "val"), "upenn_holdout": ("upenn", "holdout"), "ms_check": ("ms", "check")}[name]
        t, caches = ext_set(cfg, source, split)
        images = [caches[c].image(p) for p, c in zip(t["path"], t["cache"])]
    elif name == "case":
        t = case_set(cfg, marks)
        images = [preprocess(read_image(Path(cfg.data_root) / p), cfg.size)[0] for p in t["path"]]
    elif name == "figshare":
        pool = PreprocCache(cfg.cache_dir, "pool", cfg.size)
        p = pd.read_csv(Path(cfg.processed_dir) / "pool.csv", keep_default_na=False)
        p = p[p["source"] == "figshare"]
        t = pd.DataFrame({"path": p["path"], "mode": "t1c", "patient": p["group"], "group": p["group"], "plane": "",
                          "split_case": p["split"]})
        images = [pool.image(x) for x in t["path"]]
    else:
        raise ValueError(name)
    prob = run(model, images)
    t = t.drop(columns=["cache"], errors="ignore").reset_index(drop=True)
    for k, m in enumerate(MODES):
        t[f"p_{m}"] = prob[:, k]
    t["pred"] = [MODES[k] for k in prob.argmax(1)]
    t["set"] = name
    return t


def summarize(t: pd.DataFrame, boot: dict) -> dict:
    mri = t[t["mode"].isin(MODES)]
    called = mri[mri["pred"] == "t1c"]
    k, n = int((called["mode"] == "t1c").sum()), len(called)
    lo, hi = boot_ci(called, lambda x: (x["mode"] == "t1c").mean(), boot["n"], boot["seed"]) if n else (np.nan, np.nan)
    wl, wh = wilson(k, n) if n else (np.nan, np.nan)
    true_t1c = mri[mri["mode"] == "t1c"]
    has_other = bool((mri["mode"] != "t1c").any())
    cm = pd.crosstab(mri["mode"], mri["pred"]).reindex(index=[m for m in MODES if m in set(mri["mode"])], columns=MODES, fill_value=0)
    r = {"n": len(mri), "patients": int(mri["group"].nunique()), "by_mode": mri["mode"].value_counts().to_dict(),
         "t1c_precision": {"k": k, "n": n, "rate": k / n if n else np.nan, "lo": lo, "hi": hi, "wilson_lo": wl, "wilson_hi": wh},
         "t1c_recall": {"k": int((true_t1c["pred"] == "t1c").sum()), "n": len(true_t1c)},
         "has_other_modes": has_other, "accuracy": float((mri["pred"] == mri["mode"]).mean()) if len(mri) else np.nan,
         "confusion": {i: row.to_dict() for i, row in cm.iterrows()}}
    ct = t[t["mode"] == "ct"]
    if len(ct):
        r["ct"] = {"n": len(ct), "called": ct["pred"].value_counts().to_dict()}
    if "n_reviewers" in t.columns:
        two = mri[mri["n_reviewers"] == 2]
        c2 = two[two["pred"] == "t1c"]
        r["two_reviewers"] = {"n": len(two), "t1c_precision": f"{int((c2['mode'] == 't1c').sum())} из {len(c2)}"}
    return r


def pct(r: dict) -> str:
    return "—" if not r["n"] else f"{r['k']} из {r['n']} = {r['rate']:.1%} [{r['lo']:.1%}; {r['hi']:.1%}]"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--system", type=Path, default=Path("configs/system_run3.yaml"))
    ap.add_argument("--rules", type=Path, default=Path("configs/eval_external.yaml"))
    ap.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    ap.add_argument("--sets", default=",".join(ALL_SETS))
    ap.add_argument("--marks", type=Path, default=Path("reports/doctor/no_tumor_review/marks_merged.csv"))
    ap.add_argument("--models", type=Path, default=Path("models"))
    args = ap.parse_args()
    sys.stdout.reconfigure(errors="replace")
    torch.set_num_threads(6)
    rules = yaml.safe_load(args.rules.read_text(encoding="utf-8"))
    system = yaml.safe_load(args.system.read_text(encoding="utf-8"))
    cfg = DataConfig.from_yaml(args.config)
    sets = args.sets.split(",")
    if {"upenn_holdout", "ms_check"} & set(sets):
        print("ВНИМАНИЕ: открывается отложенная проверка определителя режима — запишите дату и версию в журнал проекта", flush=True)
    model, run_name = load_model(system["mode_detector"], args.models, torch.device("cpu"))
    out = Path("reports/eval_external") / args.tag
    out.mkdir(parents=True, exist_ok=True)
    frames, res, t0 = [], {}, time.time()
    for name in sets:
        t = predict_set(name, cfg, model, args.marks)
        frames.append(t)
        res[name] = summarize(t, rules["bootstrap"])
        print(f"{name}: {len(t)} снимков, {time.time() - t0:.0f} с", flush=True)
    pd.concat(frames, ignore_index=True).to_csv(out / "mode_pred.csv", index=False, encoding="utf-8")
    need = rules["mode_detector"]["min_t1c_precision"]
    checked = [s for s in RULE_SETS if s in res and res[s]["has_other_modes"]]
    passes = bool(checked) and all(res[s]["t1c_precision"]["rate"] >= need for s in checked) and set(RULE_SETS) <= set(res)
    js = {"tag": args.tag, "model": run_name, "min_t1c_precision": need, "rule_sets": list(RULE_SETS), "checked": checked,
          "passes": passes, "sets": res}
    (out / "mode_check.json").write_text(json.dumps(js, ensure_ascii=False, indent=1, default=float), encoding="utf-8")

    L = ["# П6. Определитель режима МРТ на проверочных наборах", "",
         f"`python -m src.eval_mode --tag {args.tag}`, {time.strftime('%Y-%m-%d %H:%M')}. Модель `{run_name}` (учился на аксиальных "
         "срезах UPENN-GBM train). Правило — `configs/eval_external.yaml` (mode_detector): автоматически определенный «T1 с контрастом» "
         f"снимает предупреждение, только если точность этого ответа ≥ {need:.0%} на каждом наборе, где есть снимки других режимов "
         f"({', '.join(RULE_SETS)}). Интервал точности — бутстреп по пациентам.", "",
         "| набор | снимков (пациентов) | названо «T1 с контрастом»: из них верно | настоящих T1 с контрастом названо так | верный режим, все снимки |",
         "|---|---|---|---|---|"]
    for name, r in res.items():
        rc = r["t1c_recall"]
        L.append(f"| {SET_RU[name]} | {r['n']} ({r['patients']}) | {pct(r['t1c_precision'])} | "
                 f"{rc['k']} из {rc['n']} | {r['accuracy']:.1%} |")
    L += ["", f"**П6 {'выполнено' if passes else 'не выполнено'}**: " +
          ("автоматически определенный «T1 с контрастом» снимает предупреждение." if passes else
           "предупреждение показывается всегда, если режим не выбрал врач; автоопределение — только подсказка.") +
          " Наборы правила: " + ", ".join(f"{s} {res[s]['t1c_precision']['rate']:.1%}" for s in checked if s in res) + ".", ""]
    for name, r in res.items():
        L += [f"**{SET_RU[name]}** — режим (строки) × ответ (столбцы):", "",
              "| режим \\ ответ | " + " | ".join(MODES) + " |", "|---" * (len(MODES) + 1) + "|"]
        L += [f"| {MODE_RU.get(i, i)} | " + " | ".join(str(row.get(m, 0)) for m in MODES) + " |" for i, row in r["confusion"].items()]
        if "ct" in r:
            L.append(f"\nСнимки «КТ» (в меру не входят): {r['ct']['n']}, ответы: {r['ct']['called']}")
        if "two_reviewers" in r:
            L.append(f"\nТолько снимки, отмеченные двумя проверяющими одинаково: {r['two_reviewers']['n']}; точность «T1 с контрастом» "
                     f"{r['two_reviewers']['t1c_precision']}")
        L.append("")
    (out / "mode_check.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
