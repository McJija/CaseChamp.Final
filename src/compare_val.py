"""Сравнение моделей этапа 3 с нынешними на промежуточной проверке (val) — по каждой модели отдельно, без порога.

Нынешние модели (`models/final` и их копии на seed 43, 44) — ответы из замера «до»
(`reports/eval_external/before/pred_val.csv` и строки `split == val` из `pred_ext.csv`).
Новые — `val_predictions.csv` папок `models/<run>_s<seed>_*` (записаны при обучении).
Отложенная проверка (check, holdout) и test здесь не используются.

    python -m src.compare_val
Если есть ответы кандидата на val (`reports/eval_external/run3_val/predictions.csv`) и выбранная пометка «проверьте»
(`configs/confidence_run3.yaml`) — добавляется раздел «система целиком» (порог, П3, П5, режимы).
Результат: `reports/run3/val_compare.md`, `reports/run3/val_compare.json`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

TYPES = ["glioma", "meningioma", "pituitary"]
CLASSES = TYPES + ["no_tumor"]
SEEDS = (42, 43, 44)
MIN_AREA = 50  # пикселей в координатах модели, как в системе


def run_dir(models: Path, run: str, seed: int) -> Path:
    found = sorted(models.glob(f"{run}_s{seed}_*"))
    if len(found) != 1:
        raise SystemExit(f"{run} seed {seed}: найдено папок {len(found)}")
    return found[0]


def new_copies(models: Path, cls_run: str, seg_run: str) -> pd.DataFrame:
    """Ответы новых моделей по копиям: p_tumor_c, argmax_c, type_c, found_c, dice_c."""
    base = None
    for c, seed in enumerate(SEEDS):
        cl = pd.read_csv(run_dir(models, cls_run, seed) / "val_predictions.csv", keep_default_na=False)
        sg = pd.read_csv(run_dir(models, seg_run, seed) / "val_predictions.csv", keep_default_na=False)
        p = cl[[f"p_{k}" for k in CLASSES]].astype(float).to_numpy()
        part = pd.DataFrame({"path": cl["path"], f"p_tumor_{c}": 1 - p[:, 3],
                             f"argmax_{c}": [CLASSES[k] for k in p.argmax(1)],
                             f"type_{c}": [TYPES[k] for k in p[:, :3].argmax(1)]})
        sg_part = pd.DataFrame({"path": sg["path"], f"found_{c}": sg["pred_area"].astype(int) > MIN_AREA,
                                f"dice_{c}": pd.to_numeric(sg["dice"], errors="coerce")})
        part = part.merge(sg_part, on="path", validate="1:1")
        if base is None:
            base = cl[["path", "cls", "source", "origin", "group", "plane", "has_mask"]].merge(part, on="path", validate="1:1")
        else:
            base = base.merge(part, on="path", validate="1:1")
    return base


def old_copies(before: Path) -> pd.DataFrame:
    """Ответы нынешних моделей (три копии) на нашем val и внешнем val из замера «до»."""
    v = pd.read_csv(before / "pred_val.csv", keep_default_na=False, low_memory=False)
    e = pd.read_csv(before / "pred_ext.csv", keep_default_na=False, low_memory=False)
    e = e[e["split"] == "val"]
    out = []
    for d, origin in ((v, "pool"), (e, "ext")):
        x = pd.DataFrame({"path": d["path"], "origin": origin})
        for c in range(3):
            x[f"p_tumor_{c}"] = d[f"p_tumor_{c}"].astype(float).to_numpy()
            x[f"argmax_{c}"] = d[f"cls_argmax_{c}"].to_numpy()
            x[f"type_{c}"] = d[f"type_{c}"].to_numpy()
            x[f"found_{c}"] = d[f"found_{c}"].astype(str).str.lower().eq("true").to_numpy()
            x[f"dice_{c}"] = pd.to_numeric(d[f"dice_gt_{c}"], errors="coerce").to_numpy()
        out.append(x)
    return pd.concat(out, ignore_index=True)


def k_of_n(mask: pd.Series) -> tuple[int, int]:
    return int(mask.sum()), int(len(mask))


def per_copy(d: pd.DataFrame, fn) -> list:
    return [fn(d, c) for c in range(3)]


def model_metrics(d: pd.DataFrame) -> dict:
    """Метрики одной группы снимков по каждой из трех копий."""
    tu, nt = d[d["cls"] != "no_tumor"], d[d["cls"] == "no_tumor"]
    withmask = tu[tu["has_mask"]]
    m = {"n_tumor": len(tu), "n_no_tumor": len(nt), "n_mask": len(withmask)}
    # модель контура
    m["dice"] = per_copy(withmask, lambda x, c: float(x[f"dice_{c}"].fillna(0).mean()) if len(x) else np.nan)
    m["found"] = per_copy(tu, lambda x, c: k_of_n(x[f"found_{c}"]))
    m["false_mask"] = per_copy(nt, lambda x, c: k_of_n(x[f"found_{c}"]))
    # модель типа (по argmax, без порога)
    m["cls_recall"] = per_copy(tu, lambda x, c: k_of_n(x[f"argmax_{c}"] != "no_tumor"))
    m["cls_false"] = per_copy(nt, lambda x, c: k_of_n(x[f"argmax_{c}"] != "no_tumor"))
    m["type_ok"] = per_copy(tu, lambda x, c: k_of_n(x[f"type_{c}"] == x["cls"]))
    y = (d["cls"] != "no_tumor").to_numpy()
    m["auc"] = per_copy(d, lambda x, c: float(roc_auc_score(y, x[f"p_tumor_{c}"])) if 0 < y.sum() < len(y) else np.nan)
    return m


def macro_f1_pool(d: pd.DataFrame) -> list:
    return per_copy(d, lambda x, c: float(f1_score(x["cls"], x[f"argmax_{c}"], labels=CLASSES, average="macro")))


def glioma_share(d: pd.DataFrame) -> list:
    """Тип «глиома» у менингиом и опухолей гипофиза (по типу, без учета ответа «есть / нет»)."""
    x = d[d["cls"].isin(["meningioma", "pituitary"])]
    return per_copy(x, lambda z, c: k_of_n(z[f"type_{c}"] == "glioma"))


def fmt_kn(vals: list) -> str:
    ks = [k for k, _ in vals]
    n = vals[0][1]
    if n == 0:
        return "—"
    return f"{'/'.join(map(str, ks))} из {n} ({100 * np.mean(ks) / n:.1f}%)"


def fmt_f(vals: list, nd: int = 3) -> str:
    return f"{np.mean(vals):.{nd}f} ({min(vals):.{nd}f}–{max(vals):.{nd}f})"


def groups(d: pd.DataFrame) -> dict:
    g = {"наш val": d[d["origin"] == "pool"]}
    for src, name in (("upenn", "UPENN-GBM val"), ("menrt", "BraTS-MEN-RT val")):
        x = d[d["source"] == src]
        g[name] = x
        for plane, pn in (("ax", "аксиальные"), ("cor", "корональные"), ("sag", "сагиттальные")):
            g[f"{name}, {pn}"] = x[x["plane"] == plane]
    return g


def core_report(models: Path) -> tuple[list[str], dict]:
    lines, js = [], {}
    runs = [("run3_core", s) for s in SEEDS] + [("run3_core_flair", 42)]
    for run, seed in runs:
        f = pd.read_csv(run_dir(models, run, seed) / "val_predictions.csv", keep_default_na=False)
        for col in ("dice_core", "dice_edema", "dice_whole"):
            f[col] = pd.to_numeric(f[col], errors="coerce")
        f["has_tumor"] = f["has_tumor"].astype(str).str.lower().eq("true")
        f["area_whole"] = f["area_whole"].astype(int)
        nt = f[~f["has_tumor"]]
        pat = f.groupby("patient")[["dice_core", "dice_edema"]].mean()
        r = {"dice_core": float(f["dice_core"].mean()), "dice_edema": float(f["dice_edema"].mean()),
             "dice_whole": float(f["dice_whole"].mean()),
             "patients_core": float(pat["dice_core"].mean()), "patients_edema": float(pat["dice_edema"].mean()),
             "false_region": k_of_n(nt["area_whole"] > MIN_AREA),
             "by_plane": {p: {"core": float(g["dice_core"].mean()), "edema": float(g["dice_edema"].mean())}
                          for p, g in f.groupby("plane")}}
        js[f"{run}_s{seed}"] = r
        bp = ", ".join(f"{p} {v['core']:.2f}/{v['edema']:.2f}" for p, v in r["by_plane"].items())
        k, n = r["false_region"]
        lines.append(f"| {run} s{seed} | {r['dice_core']:.3f} | {r['dice_edema']:.3f} | {r['dice_whole']:.3f} | "
                     f"{r['patients_core']:.3f} / {r['patients_edema']:.3f} | {k} из {n} | {bp} |")
    return lines, js


def mode_report(models: Path) -> tuple[list[str], dict]:
    f = pd.read_csv(run_dir(models, "run3_mode", 42) / "val_predictions.csv", keep_default_na=False)
    cm = pd.crosstab(f["mode"], f["pred"]).reindex(index=["t1c", "t1", "t2", "flair"], columns=["t1c", "t1", "t2", "flair"], fill_value=0)
    lines = ["| режим \\ ответ | " + " | ".join(cm.columns) + " |", "|---" * (len(cm.columns) + 1) + "|"]
    lines += [f"| {i} | " + " | ".join(str(v) for v in row) + " |" for i, row in cm.iterrows()]
    return lines, {"confusion": cm.to_dict(), "patients": int(f["patient"].nunique())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", type=Path, default=Path("models"))
    ap.add_argument("--before", type=Path, default=Path("reports/eval_external/before"))
    ap.add_argument("--cls-run", default="run3_cls")
    ap.add_argument("--seg-run", default="run3_seg")
    ap.add_argument("--out", type=Path, default=Path("reports/run3"))
    ap.add_argument("--after-pred", type=Path, default=Path("reports/eval_external/run3_val/predictions.csv"),
                    help="ответы кандидата на val (src.eval_external --splits val); если есть — раздел «система целиком»")
    ap.add_argument("--after-conf", type=Path, default=Path("configs/confidence_run3.yaml"))
    args = ap.parse_args()

    new = new_copies(args.models, args.cls_run, args.seg_run)
    old = old_copies(args.before)
    if set(old["path"]) != set(new["path"]):
        raise SystemExit(f"наборы снимков val разошлись: старых {len(old)}, новых {len(new)}, общих {len(set(old.path) & set(new.path))}")
    old = new[["path", "cls", "source", "group", "plane", "has_mask"]].merge(old, on="path", validate="1:1")
    new["has_mask"] = new["has_mask"].astype(str).str.lower().eq("true")
    old["has_mask"] = new.set_index("path").loc[old["path"], "has_mask"].to_numpy()

    out = {"cls_run": args.cls_run, "seg_run": args.seg_run, "groups": {}}
    lines = [f"# Этап 3: новые модели против нынешних на промежуточной проверке (val)", "",
             f"Нынешние — `models/final` и копии на seed 43, 44 (`run1_cls`, `run1_seg_neg`); новые — `{args.cls_run}`, "
             f"`{args.seg_run}` на seed 42, 43, 44. Каждая модель отдельно, решение «опухоль есть» по двум моделям и порог — "
             "следующий шаг. Числа «k из n» — по трем копиям через косую черту, в скобках среднее. Найдено — площадь контура > 50 пикселей; "
             "модель типа — по самому вероятному из четырех классов, тип — самый вероятный из трех типов опухоли.", ""]
    rows = [("Модель контура: совпадение контура (Dice), срезы с обводкой", "dice", fmt_f),
            ("Модель контура: опухоль найдена", "found", fmt_kn),
            ("Модель контура: ложный контур на срезе без опухоли", "false_mask", fmt_kn),
            ("Модель типа: «опухоль» на срезе с опухолью", "cls_recall", fmt_kn),
            ("Модель типа: «опухоль» на срезе без опухоли", "cls_false", fmt_kn),
            ("Модель типа: тип опухоли верен", "type_ok", fmt_kn),
            ("Модель типа: AUC «есть / нет»", "auc", fmt_f)]
    for name, d_new in groups(new).items():
        d_old = old[old["path"].isin(d_new["path"])]
        mn, mo = model_metrics(d_new), model_metrics(d_old)
        out["groups"][name] = {"new": mn, "old": mo}
        if "," in name and mn["n_no_tumor"] + mn["n_tumor"] == 0:
            continue
        lines += [f"## {name}", "", f"Срезов с опухолью {mn['n_tumor']} (с обводкой {mn['n_mask']}), без опухоли {mn['n_no_tumor']}.", "",
                  "| мера | нынешние | новые |", "|---|---|---|"]
        for label, key, f in rows:
            lines.append(f"| {label} | {f(mo[key])} | {f(mn[key])} |")
        if name == "наш val":
            fo, fn_ = macro_f1_pool(d_old), macro_f1_pool(d_new)
            go, gn = glioma_share(d_old), glioma_share(d_new)
            out["groups"][name]["new"]["macro_f1"], out["groups"][name]["old"]["macro_f1"] = fn_, fo
            out["groups"][name]["new"]["glioma_share"], out["groups"][name]["old"]["glioma_share"] = gn, go
            lines.append(f"| Модель типа: macro-F1 по 4 классам | {fmt_f(fo)} | {fmt_f(fn_)} |")
            lines.append(f"| Менингиомы и гипофиз с типом «глиома» (к П3, без учета «есть / нет») | {fmt_kn(go)} | {fmt_kn(gn)} |")
            for cls in TYPES:
                x_o, x_n = d_old[d_old["cls"] == cls], d_new[d_new["cls"] == cls]
                lines.append(f"| тип верен: {cls} | {fmt_kn(per_copy(x_o, lambda z, c: k_of_n(z[f'type_{c}'] == cls)))} | "
                             f"{fmt_kn(per_copy(x_n, lambda z, c: k_of_n(z[f'type_{c}'] == cls)))} |")
            for src in ("figshare", "figshare_crop", "other"):
                x_o, x_n = d_old[(d_old["source"] == src) & (d_old["cls"] != "no_tumor")], d_new[(d_new["source"] == src) & (d_new["cls"] != "no_tumor")]
                lines.append(f"| найдено контуром, опухоли `{src}` | {fmt_kn(per_copy(x_o, lambda z, c: k_of_n(z[f'found_{c}'])))} | "
                             f"{fmt_kn(per_copy(x_n, lambda z, c: k_of_n(z[f'found_{c}'])))} |")
        lines.append("")

    core_lines, out["core"] = core_report(args.models)
    lines += ["## Модель ядра и отека (UPENN-GBM val, 29 пациентов)", "",
              "Dice по срезам, где эта часть есть в обводке; «по пациентам» — среднее по пациентам; ложная область — срезы без опухоли, "
              "где найдено больше 50 пикселей; по плоскостям — ядро / отек.", "",
              "| модель | ядро | отек | вся опухоль | по пациентам: ядро / отек | ложная область | по плоскостям |", "|---|---|---|---|---|---|---|"]
    lines += core_lines + [""]
    mode_lines, out["mode"] = mode_report(args.models)
    lines += [f"## Определитель режима (аксиальные срезы UPENN-GBM val, {out['mode']['patients']} пациентов)", ""] + mode_lines + [
        "", "Это один источник: проверка на других наборах (склероз, снимки кейса, Figshare) — по правилу П6.", ""]

    if args.after_pred.exists() and args.after_conf.exists():
        lines += system_val(args.before / "predictions.csv", args.after_pred, args.after_conf, Path("configs/eval_external.yaml"))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "val_compare.md").write_text("\n".join(lines), encoding="utf-8")
    (args.out / "val_compare.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("\n".join(lines))



# ---------------------------------------------------------------- система целиком (решение «есть / нет», тип, П3, П5)


def system_val(before_pred: Path, after_pred: Path, after_conf: Path, rules_path: Path) -> list[str]:
    """Система целиком на val: нынешняя (models/final, порог 0,976) против кандидата (порог и «проверьте» подобраны на val).
    Только части val: наш val, внешний val, режимы UPENN val (аксиальные)."""
    import yaml
    from src.confidence import errors as conf_errors, score, signals
    from src.eval_external import compute

    rules = yaml.safe_load(rules_path.read_text(encoding="utf-8"))
    conf_new = yaml.safe_load(after_conf.read_text(encoding="utf-8"))["rule"]
    conf_old = {"signals": ["presence", "type", "contour"], "pair_dice_min": rules["confidence"]["min_pair_dice"]}
    res = {}
    for tag, path in (("old", before_pred), ("new", after_pred)):
        p = pd.read_csv(path, keep_default_na=False, low_memory=False)
        p = p[(p["set"] == "val") | ((p["set"].isin(["ext", "modes"])) & (p["split"] == "val"))]
        for col in [c for c in p.columns if c.startswith(("present_", "found_")) or c in ("has_tumor", "has_mask", "ms_lesion")]:
            p[col] = p[col].astype(str).str.lower().eq("true")
        for col in [c for c in p.columns if c.startswith(("dice_gt_", "p_tumor_", "p_", "pair_dice", "area_", "gt_area"))]:
            p[col] = pd.to_numeric(p[col], errors="coerce")
        M, _ = compute(p, rules)
        conf = {}
        for cname, rule in (("A", conf_old), ("new", conf_new)):
            for part, d in (("наш val", p[p["set"] == "val"]), ("внешний val", p[p["set"] == "ext"])):
                conf[(cname, part)] = score(signals(d, rule).any(axis=1), conf_errors(d), rules["confidence"])
        pool = pd.read_csv("data/processed/pool.csv", keep_default_na=False)[["path", "nt_mode"]]
        v = p[(p["set"] == "val") & ~p["has_tumor"]].drop(columns=["nt_mode"], errors="ignore").merge(pool, on="path")
        res[tag] = (M, conf, {m: (int(g["present_0"].sum()), len(g)) for m, g in v.groupby("nt_mode")})

    def f(r):
        return f"{r['k']} из {r['n']} ({r['rate']:.1%}) [{r['lo']:.1%}; {r['hi']:.1%}]" if r and r.get("n") else "—"

    def fd(r):
        return f"{r['patients_mean']:.3f} [{r['lo']:.3f}; {r['hi']:.3f}]" if r and r.get("n_slices") else "—"

    def fc(r):
        return f"помечено {r['flagged_share']:.1%}, поймано {r['caught']} из {r['errors']} ({r['caught_share']:.0%}), ×{r['ratio']:.1f} → {'да' if r['passes'] else 'нет'}"

    (Mo, co, nto), (Mn, cn, ntn) = res["old"], res["new"]
    L = ["## Система целиком на val: нынешняя против кандидата", "",
         "Нынешняя — `models/final` (порог 0,976368), кандидат — `configs/system_run3.yaml` (порог 0,989005 по записанному правилу; "
         "пометка «проверьте» — `configs/confidence_run3.yaml`). Порог и пометка кандидата подобраны на этих же снимках — числа кандидата "
         "на val оптимистичны; решение — по отложенной проверке (П1, П2). Интервалы — бутстреп по пациентам.", "",
         "| мера | нынешняя | кандидат |", "|---|---|---|",
         f"| наш val: найдено опухолей | {f(Mo['val']['recall'])} | {f(Mn['val']['recall'])} |",
         f"| наш val: ложная тревога на снимке без опухоли | {f(Mo['val']['no_tumor_errors'])} | {f(Mn['val']['no_tumor_errors'])} |"]
    for m in sorted(nto):
        L.append(f"| — в том числе режим «{m}» | {nto[m][0]} из {nto[m][1]} | {ntn[m][0]} из {ntn[m][1]} |")
    L += [f"| наш val: тип верен у найденных | {f(Mo['val']['type_correct'])} | {f(Mn['val']['type_correct'])} |",
          f"| наш val: совпадение контура (по пациентам) | {fd(Mo['val']['dice'])} | {fd(Mn['val']['dice'])} |",
          f"| П3: менингиомы и гипофиз с типом «глиома», среднее трех копий | {Mo['val']['glioma_share']['mean']:.2%} | {Mn['val']['glioma_share']['mean']:.2%} |"]
    for s, name in (("upenn", "UPENN-GBM"), ("menrt", "BraTS-MEN-RT")):
        o, n = Mo["ext_val"][s], Mn["ext_val"][s]
        L += [f"| {name} val: ложная тревога на срезе без опухоли | {f(o['no_tumor_errors'])} | {f(n['no_tumor_errors'])} |",
              f"| {name} val: найдено опухолей | {f(o['found'])} | {f(n['found'])} |",
              f"| {name} val: тип верен у найденных | {f(o['type_correct'])} | {f(n['type_correct'])} |",
              f"| {name} val: совпадение контура (по пациентам) | {fd(o['dice'])} | {fd(n['dice'])} |"]
    for part in ("наш val", "внешний val"):
        L += [f"| «проверьте» прежним способом (A), {part} | {fc(co[('A', part)])} | {fc(cn[('A', part)])} |",
              f"| «проверьте» выбранным способом, {part} | {fc(co[('new', part)])} | {fc(cn[('new', part)])} |"]
    L += ["", "**По режимам МРТ** (UPENN-GBM val, аксиальные срезы, одни и те же места; 29 пациентов):", "",
          "| режим | найдено: нынешняя | найдено: кандидат | ложная тревога: нынешняя | ложная тревога: кандидат | тип верен: кандидат |",
          "|---|---|---|---|---|---|"]
    for m in ("t1c", "t1", "t2", "flair"):
        o, n = Mo["modes"].get(f"upenn|{m}", {}), Mn["modes"].get(f"upenn|{m}", {})
        L.append(f"| {m} | {f(o.get('found'))} | {f(n.get('found'))} | {f(o.get('no_tumor_errors'))} | {f(n.get('no_tumor_errors'))} | {f(n.get('type_correct'))} |")
    return L + [""]


if __name__ == "__main__":
    main()
