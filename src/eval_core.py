"""П7 — ядро и отек при глиобластоме на отложенных пациентах UPENN-GBM (только отчет, без порога).

Правило П7 (`configs/eval_external.yaml`; записано 2026-11-12): Dice ядра и отека по пациентам на 30 пациентах holdout
UPENN-GBM с интервалом; «только T1 с контрастом» (`run3_core`, seed 42, 43, 44) против «T1 с контрастом + FLAIR»
(`run3_core_flair`, seed 42). Порогов нет — решение о том, что показывать в прототипе, принимает руководитель проекта.

Ядро = метки 1 + 3 обводки UPENN-GBM (некроз и контрастируемая часть), отек = вся опухоль (1 + 2 + 3) минус ядро.
Dice среза считается там, где эта часть есть в обводке; по пациенту — среднее его срезов; итог — среднее по пациентам,
интервал — бутстреп по пациентам (2000 повторов, seed 42). Разница «T1c + FLAIR» минус «T1c» (seed 42) — парный
бутстреп по пациентам. Ложная область — срез без опухоли, где модель нашла всю опухоль площадью > 50 пикселей.
Прогон в точности как при выборе эпохи (`src.train_aux.evaluate`, без аугментаций).

    python -m src.eval_core --tag after                  # holdout (отложенная проверка)
    python -m src.eval_core --tag dev --split val        # проверка кода: должно совпасть с val_predictions.csv обучения
Результат: reports/eval_external/<tag>/{core_pred.csv, core_check.json, core_check.md}.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.dataset import DataConfig, PreprocCache
from src.evaluate import boot_ci
from src.train_aux import CORE_OUT, build, evaluate, make_loader

MIN_AREA = 50
PLANE_RU = {"ax": "аксиальная", "cor": "корональная", "sag": "сагиттальная"}


def load_model(run_dir: Path, device) -> tuple[torch.nn.Module, bool]:
    ck = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    if ck["kind"] != "core" or list(ck["classes"]) != CORE_OUT:
        raise ValueError(f"{run_dir} — не модель ядра и отека")
    model = build("core", ck["encoder"], False)
    model.load_state_dict(ck["model"])
    return model.eval().to(device), ck["inputs"] == "t1c_flair"


def per_patient(frame: pd.DataFrame, col: str, boot: dict) -> dict:
    x = frame[frame[col].notna()]
    if x.empty:
        return {"n_patients": 0}
    pp = x.groupby("group")[col].mean().rename("v").reset_index()
    lo, hi = boot_ci(pp, lambda y: y["v"].mean(), boot["n"], boot["seed"])
    return {"patients_mean": float(pp["v"].mean()), "lo": lo, "hi": hi, "n_patients": int(len(pp)),
            "slices_mean": float(x[col].mean()), "n_slices": int(len(x))}


def paired_diff(a: pd.DataFrame, b: pd.DataFrame, col: str, boot: dict) -> dict:
    """b − a по пациентам (у кого часть есть в обводке), парный бутстреп."""
    pa = a[a[col].notna()].groupby("group")[col].mean()
    pb = b[b[col].notna()].groupby("group")[col].mean()
    d = (pb - pa).dropna().rename("v").reset_index()
    lo, hi = boot_ci(d, lambda y: y["v"].mean(), boot["n"], boot["seed"])
    return {"diff": float(d["v"].mean()), "lo": lo, "hi": hi, "n_patients": int(len(d))}


def summarize(frame: pd.DataFrame, boot: dict) -> dict:
    nt = frame[~frame["has_tumor"]]
    k = int((nt["area_whole"] > MIN_AREA).sum())
    return {"core": per_patient(frame, "dice_core", boot), "edema": per_patient(frame, "dice_edema", boot),
            "whole": per_patient(frame, "dice_whole", boot),
            "false_region": {"k": k, "n": int(len(nt)), "patients_with": int(nt.loc[nt["area_whole"] > MIN_AREA, "group"].nunique())},
            "by_plane": {p: {"core": float(g["dice_core"].mean()), "edema": float(g["dice_edema"].mean())}
                         for p, g in frame.groupby("plane")}}


def fmt(r: dict) -> str:
    return "—" if not r.get("n_patients") else (f"{r['patients_mean']:.3f} [{r['lo']:.3f}; {r['hi']:.3f}] "
                                                f"({r['n_patients']} пац.; по срезам {r['slices_mean']:.3f})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--split", default="holdout", choices=["holdout", "val"])
    ap.add_argument("--system", type=Path, default=Path("configs/system_run3.yaml"))
    ap.add_argument("--rules", type=Path, default=Path("configs/eval_external.yaml"))
    ap.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    ap.add_argument("--models", type=Path, default=Path("models"))
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()
    sys.stdout.reconfigure(errors="replace")
    torch.set_num_threads(6)
    if args.split == "holdout":
        print("ВНИМАНИЕ: открывается отложенная проверка ядра и отека (UPENN-GBM holdout) — запишите дату и версию в журнал проекта", flush=True)
    boot = yaml.safe_load(args.rules.read_text(encoding="utf-8"))["bootstrap"]
    system = yaml.safe_load(args.system.read_text(encoding="utf-8"))["core_edema"]
    cfg = replace(DataConfig.from_yaml(args.config), batch_size=args.batch_size, num_workers=0, domain_randomization=False)
    ext, fl = PreprocCache(cfg.cache_dir, "ext", cfg.size), PreprocCache(cfg.cache_dir, "upennf", cfg.size)
    table = fl.table[fl.table["split"] == args.split].copy()
    table["has_tumor"] = table["has_tumor"].astype(str).str.lower().eq("true")
    table["label"] = 0
    table = table.reset_index(drop=True)
    out = Path("reports/eval_external") / args.tag
    out.mkdir(parents=True, exist_ok=True)

    frames, res, t0 = {}, {}, time.time()
    for inputs, patterns in system.items():
        for pat in patterns:
            runs = sorted(args.models.glob(pat))
            if len(runs) != 1:
                raise ValueError(f"шаблону {pat!r} соответствует {len(runs)} папок")
            model, flair = load_model(runs[0], torch.device("cpu"))
            if flair != (inputs == "t1c_flair"):
                raise ValueError(f"{runs[0].name}: вход модели не совпадает с разделом {inputs}")
            loader = make_loader(table, "core", (ext, fl), cfg, False, flair)
            _, frame = evaluate(model, "core", loader, table, torch.device("cpu"), MIN_AREA)
            frames[runs[0].name] = frame
            res[runs[0].name] = {"inputs": inputs, **summarize(frame, boot)}
            print(f"{runs[0].name}: {time.time() - t0:.0f} с", flush=True)
    pd.concat([f.assign(run=k) for k, f in frames.items()], ignore_index=True).to_csv(out / "core_pred.csv", index=False, encoding="utf-8")

    t1c = [k for k in frames if res[k]["inputs"] == "t1c"]
    fla = [k for k in frames if res[k]["inputs"] == "t1c_flair"]
    main_t1c = next((k for k in t1c if "_s42_" in k), t1c[0])
    diffs = {part: paired_diff(frames[main_t1c], frames[fla[0]], f"dice_{part}", boot) for part in ("core", "edema")} if fla else {}
    avg = {part: float(np.mean([res[k][part]["patients_mean"] for k in t1c])) for part in ("core", "edema")}
    rng = {part: (float(min(res[k][part]["patients_mean"] for k in t1c)), float(max(res[k][part]["patients_mean"] for k in t1c)))
           for part in ("core", "edema")}
    js = {"tag": args.tag, "split": args.split, "patients": int(table["group"].nunique()), "slices": int(len(table)),
          "slices_with_tumor": int(table["has_tumor"].sum()), "runs": res, "t1c_mean_of_seeds": avg, "t1c_range_of_seeds": rng,
          "flair_minus_t1c_s42": diffs}
    (out / "core_check.json").write_text(json.dumps(js, ensure_ascii=False, indent=1, default=float), encoding="utf-8")

    part_ru = "отложенные пациенты (holdout)" if args.split == "holdout" else "val (промежуточная проверка)"
    L = ["# П7. Ядро и отек при глиобластоме (UPENN-GBM)", "",
         f"`python -m src.eval_core --tag {args.tag} --split {args.split}`, {time.strftime('%Y-%m-%d %H:%M')}. {part_ru}: "
         f"{js['patients']} пациентов, {js['slices']} срезов всех плоскостей ({js['slices_with_tumor']} с опухолью). Без порога — только отчет. "
         "Совпадение (Dice) — среднее по пациентам, интервал — бутстреп по пациентам. Ядро — некроз и контрастируемая часть; "
         "отек — вся опухоль минус ядро. Ложная область — срез без опухоли, где найдено больше 50 пикселей.", "",
         "| модель | вход | ядро | отек | вся опухоль | ложная область на срезах без опухоли |", "|---|---|---|---|---|---|"]
    for k, r in res.items():
        fr = r["false_region"]
        L.append(f"| {k} | {'T1 с контрастом + FLAIR' if r['inputs'] == 't1c_flair' else 'T1 с контрастом'} | {fmt(r['core'])} | "
                 f"{fmt(r['edema'])} | {fmt(r['whole'])} | {fr['k']} из {fr['n']} ({fr['patients_with']} пац.) |")
    L += ["", f"Только T1 с контрастом, среднее трех seed: ядро {avg['core']:.3f} (seed {rng['core'][0]:.3f}–{rng['core'][1]:.3f}), "
          f"отек {avg['edema']:.3f} ({rng['edema'][0]:.3f}–{rng['edema'][1]:.3f})."]
    if diffs:
        L += [f"Разница «T1 с контрастом + FLAIR» минус «только T1 с контрастом» (seed 42, парно по пациентам): ядро "
              f"{diffs['core']['diff']:+.3f} [{diffs['core']['lo']:+.3f}; {diffs['core']['hi']:+.3f}], отек "
              f"{diffs['edema']['diff']:+.3f} [{diffs['edema']['lo']:+.3f}; {diffs['edema']['hi']:+.3f}]."]
    L += ["", "По плоскостям (среднее по срезам, ядро / отек):", ""]
    L += [f"- {k}: " + ", ".join(f"{PLANE_RU.get(p, p)} {v['core']:.2f} / {v['edema']:.2f}" for p, v in r["by_plane"].items())
          for k, r in res.items()]
    (out / "core_check.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
