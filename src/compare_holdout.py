"""Подробное сравнение «было / стало» после единственного открытия отложенной проверки (правила П1–П7).

Было — нынешняя система `models/final` (reports/eval_external/before), стало — кандидат этапа 3
(`configs/system_run3.yaml`, reports/eval_external/after). Все меры считаются одним кодом (`src.eval_external.compute`)
по таблицам предсказаний обеих систем; пометка «проверьте» — по прежнему способу A и по выбранному на val E0.6 для
обеих систем. Решение по П2 — подсказка для руководителя проекта: замена `models/final` только после его «ок».

По отложенной проверке (check, holdout) — только сводные числа. Примеры снимков с верными и ошибочными ответами —
только из val (наш val и внешний val), отбор правилом ниже (фиксированный seed), не по результатам отложенной проверки.

    python -m src.compare_holdout
Результат: reports/run3/before_after.{md,json}, reports/run3/examples_val.png.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from src.confidence import errors as conf_errors, score, signals  # noqa: E402
from src.dataset import DataConfig, PreprocCache, to_tensor  # noqa: E402
from src.eval_external import SRC_RU, compute, default_rule, find_run, typed  # noqa: E402
from src.evaluate import boot_ci  # noqa: E402
from src.inference import _load  # noqa: E402
from src.labels import CLASSES, NO_TUMOR_IDX  # noqa: E402

TYPE_RU = {"glioma": "глиома", "meningioma": "менингиома", "pituitary": "гипофиз", "no_tumor": "нет опухоли",
           "schwannoma": "шваннома", "ms": "склероз"}
PLANE_RU = {"ax": "аксиальная", "cor": "корональная", "sag": "сагиттальная"}
MODE_RU = {"t1c": "T1 с контрастом", "t1": "T1", "t2": "T2", "flair": "FLAIR"}


def load(tag: str) -> pd.DataFrame:
    return typed(pd.read_csv(Path("reports/eval_external") / tag / "predictions.csv", keep_default_na=False, low_memory=False))


def f(r: dict | None) -> str:
    if not r or not r.get("n"):
        return "—"
    return f"{r['k']} из {r['n']} ({r['rate']:.1%}) [{r['lo']:.1%}; {r['hi']:.1%}]"


def fd(r: dict | None) -> str:
    return "—" if not r or not r.get("n_slices") else f"{r['patients_mean']:.3f} [{r['lo']:.3f}; {r['hi']:.3f}] ({r['n_patients']} пац.)"


def fc(r: dict) -> str:
    return (f"помечено {r['flagged']} из {r['n']} ({r['flagged_share']:.1%}), поймано {r['caught']} из {r['errors']} "
            f"({r['caught_share']:.0%}), ×{r['ratio']:.1f} → {'да' if r['passes'] else 'нет'}")


def paired(a: pd.DataFrame, b: pd.DataFrame, boot: dict) -> dict:
    """Разница долей ошибок «стало» − «было» на одних и тех же срезах; парный бутстреп по пациентам."""
    m = a[["path", "group", "present_0"]].merge(b[["path", "present_0"]], on="path", suffixes=("_a", "_b"), validate="1:1")
    fn = lambda x: x["present_0_b"].mean() - x["present_0_a"].mean()  # noqa: E731
    lo, hi = boot_ci(m, fn, boot["n"], boot["seed"])
    return {"diff": float(fn(m)), "lo": lo, "hi": hi, "n": int(len(m)), "patients": int(m["group"].nunique())}


def main_check_rows(d: pd.DataFrame, rules: dict) -> pd.DataFrame:
    mc = rules["main_check"]
    return d[(d["set"] == "ext") & (d["split"] == mc["split"]) & (~d["has_tumor"]) & (d["mode"] == mc["mode"])]


def conf_table(pred: dict[str, pd.DataFrame], rules: dict, rule_new: dict) -> list[str]:
    crit = rules["confidence"]
    parts = {"наш val": lambda d: d[d["set"] == "val"],
             "внешний holdout (T1 с контрастом)": lambda d: d[(d["set"] == "ext") & (d["split"] == "holdout")],
             "П1 (склероз и шванномы без опухоли)": lambda d: main_check_rows(d, rules)}
    L = ["| часть | способ | было | стало |", "|---|---|---|---|"]
    for pname, sel in parts.items():
        for rname, rule in (("A (прежний)", default_rule(crit)), (rule_new["name"], rule_new)):
            cells = []
            for tag in ("before", "after"):
                x = sel(pred[tag])
                cells.append(fc(score(signals(x, rule).any(axis=1), conf_errors(x), crit)))
            L.append(f"| {pname} | {rname} | {cells[0]} | {cells[1]} |")
    return L


# ---------------------------------------------------------------- примеры (только val)


def pick_examples(a: pd.DataFrame, b: pd.DataFrame, seed: int = 42) -> list[tuple[str, pd.Series]]:
    """Отбор правилом: по каждой группе — все случаи, если их ≤ 3, иначе 3 случайных (seed 42). Только val."""
    m = a.merge(b[["path", "present_0", "type_0", "answer_0"]], on="path", suffixes=("_a", "_b"))
    val = m[(m["set"] == "val") | ((m["set"] == "ext") & (m["split"] == "val"))]
    ours, ext = val[val["set"] == "val"], val[val["set"] == "ext"]
    ok_b = lambda x: (x["present_0_b"] == x["has_tumor"]) & (~x["has_tumor"] | (x["type_0_b"] == x["cls"]))  # noqa: E731
    ok_a = lambda x: (x["present_0_a"] == x["has_tumor"]) & (~x["has_tumor"] | (x["type_0_a"] == x["cls"]))  # noqa: E731
    groups = [
        ("наш val: верно у обеих", ours[ok_a(ours) & ok_b(ours) & ours["has_tumor"]]),
        ("наш val: опухоль пропущена новой системой", ours[ours["has_tumor"] & ~ours["present_0_b"]]),
        ("наш val: ложная тревога новой системы", ours[~ours["has_tumor"] & ours["present_0_b"]]),
        ("внешний val: ложная тревога исправлена", ext[~ext["has_tumor"] & ext["present_0_a"] & ~ext["present_0_b"]]),
        ("внешний val (менингиома): тип исправлен", ext[(ext["source"] == "menrt") & ext["has_tumor"] & ext["present_0_a"] & ext["present_0_b"]
                                                      & (ext["type_0_a"] != ext["cls"]) & (ext["type_0_b"] == ext["cls"])]),
        ("внешний val: новая система ошибается", ext[~ok_b(ext)]),
    ]
    out = []
    for name, g in groups:
        g = g if len(g) <= 3 else g.sample(3, random_state=seed)
        out += [(name, r) for _, r in g.iterrows()]
    return out


@torch.no_grad()
def masks_for(rows: list[pd.Series], cls_seg: dict[str, tuple[Path, Path]], cfg: DataConfig) -> dict:
    caches = {"val": PreprocCache(cfg.cache_dir, "pool", cfg.size), "ext": PreprocCache(cfg.cache_dir, "ext", cfg.size)}
    imgs = {r["path"]: caches[r["set"]].image(r["path"]) for _, r in rows}
    gts = {r["path"]: caches[r["set"]].mask(r["path"]) > 0 for _, r in rows}
    out = {}
    for tag, (_, ps) in cls_seg.items():
        seg = _load(ps, torch.device("cpu"))[0]
        for p, im in imgs.items():
            m = (torch.sigmoid(seg(to_tensor(im)[None]).float())[0, 0] > 0.5).numpy()
            out[(tag, p)] = m if m.sum() > 50 else np.zeros_like(m)
    return {"img": imgs, "gt": gts, "pred": out}


def overlay(img: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    rgb = np.repeat(img[..., None], 3, 2).astype(float) / 255
    rgb[pred] = 0.55 * rgb[pred] + 0.45 * np.array([0.85, 0.2, 0.2])
    edge = gt & ~np.roll(gt, 1, 0) | gt & ~np.roll(gt, 1, 1) | gt & ~np.roll(gt, -1, 0) | gt & ~np.roll(gt, -1, 1)
    rgb[edge] = [0.1, 0.9, 0.3]
    return rgb


def sheet(rows: list[tuple[str, pd.Series]], mk: dict, path: Path) -> None:
    """Строка — группа, в строке до трех пар «было / стало»."""
    names = list(dict.fromkeys(n for n, _ in rows))
    fig = plt.figure(figsize=(15, 3.0 * len(names) + 0.6))
    subs = np.atleast_1d(fig.subfigures(len(names), 1))
    for sf, name in zip(subs, names):
        grp = [r for n, r in rows if n == name]
        sf.suptitle(f"{name} (показано {len(grp)})", fontsize=11, fontweight="bold", x=0.01, ha="left")
        axes = sf.subplots(1, 6)
        for ax in axes:
            ax.axis("off")
        for k, r in enumerate(grp):
            truth = TYPE_RU.get(r["cls"], r["cls"]) if r["has_tumor"] else "нет опухоли"
            src = "наш val" if r["set"] == "val" else SRC_RU.get(r["source"], r["source"])
            for j, (tag, label) in enumerate((("before", "было"), ("after", "стало"))):
                s = "a" if tag == "before" else "b"
                ans = TYPE_RU.get(r[f"answer_0_{s}"] if f"answer_0_{s}" in r else r["answer_0"], "?")
                ax = axes[2 * k + j]
                ax.imshow(overlay(mk["img"][r["path"]], mk["gt"][r["path"]], mk["pred"][(tag, r["path"])]))
                ax.set_title(f"{label}: {ans}" + (f"\n{src}; правда: {truth}" if j == 0 else "\n "), fontsize=8.5)
    fig.suptitle("Примеры ответов на val: зеленый контур — обводка врача, красное — область модели контура", fontsize=12, y=1.0 + 0.35 / (3.0 * len(names) + 0.6))
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rules", type=Path, default=Path("configs/eval_external.yaml"))
    ap.add_argument("--system", type=Path, default=Path("configs/system_run3.yaml"))
    ap.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    ap.add_argument("--out", type=Path, default=Path("reports/run3/before_after.md"))
    ap.add_argument("--before-tag", default="before")
    ap.add_argument("--after-tag", default="after")
    ap.add_argument("--aux-dir", type=Path, default=None, help="где mode_check и core_check (по умолчанию — папка after-tag)")
    args = ap.parse_args()
    sys.stdout.reconfigure(errors="replace")
    rules = yaml.safe_load(args.rules.read_text(encoding="utf-8"))
    system = yaml.safe_load(args.system.read_text(encoding="utf-8"))
    rule_new = yaml.safe_load(Path(system["confidence"]).read_text(encoding="utf-8"))["rule"]
    boot = rules["bootstrap"]
    pool = pd.read_csv("data/processed/pool.csv", keep_default_na=False)[["path", "nt_mode"]]
    pred = {}
    for tag, src_tag in (("before", args.before_tag), ("after", args.after_tag)):
        p = load(src_tag)
        p = p.drop(columns=["nt_mode"], errors="ignore").merge(pool, on="path", how="left")
        p["nt_mode"] = p["nt_mode"].fillna("")
        pred[tag] = p
    M = {"before": compute(pred["before"], rules)[0], "after": compute(pred["after"], rules, rule_new)[0]}
    Mo, Mn = M["before"], M["after"]

    # П1 и П2
    p1 = paired(main_check_rows(pred["before"], rules), main_check_rows(pred["after"], rules), boot)
    drop_pp = -p1["diff"] * 100
    rec_o, rec_n = Mo["val"]["recall"], Mn["val"]["recall"]
    rec_drop_pp = (rec_o["rate"] - rec_n["rate"]) * 100
    rp = rules["replacement"]
    c1, c2 = drop_pp >= rp["min_drop_main_check_pp"], rec_drop_pp <= rp["max_recall_drop_val_pp"]
    g_o, g_n = Mo["val"]["glioma_share"]["mean"], Mn["val"]["glioma_share"]["mean"]
    c3 = (g_n - g_o) * 100 <= rules["glioma_shortcut"]["max_rise_pp"]
    out_dir = args.aux_dir or Path("reports/eval_external") / args.after_tag
    mode_js = json.loads((out_dir / "mode_check.json").read_text(encoding="utf-8")) if (out_dir / "mode_check.json").exists() else None
    core_js = json.loads((out_dir / "core_check.json").read_text(encoding="utf-8")) if (out_dir / "core_check.json").exists() else None

    L = ["# Было / стало: нынешняя система против кандидата этапа 3", "",
         "Было — `models/final` (модель типа `run1_cls` s42 + модель контура `run1_seg_neg` s42, порог 0,976368). "
         "Стало — кандидат `configs/system_run3.yaml` (`run3_cls` + `run3_seg` s42, порог 0,989005; копии — seed 43, 44). "
         "Отложенная проверка открыта для кандидата один раз (дата и версия записаны в журнале проекта); по ней — только сводные числа. "
         "Интервалы — бутстреп по пациентам (2000 повторов, seed 42). `python -m src.compare_holdout`.", "",
         "## Итог по правилам", "",
         "| правило | было | стало | выполнено? |", "|---|---|---|---|",
         f"| П1: ложная тревога на срезах T1 с контрастом без опухоли (склероз + шванномы) | {f(Mo['main_check']['all'])} | "
         f"{f(Mn['main_check']['all'])} | — |",
         f"| П2.1: ложных тревог по П1 меньше не менее чем на {rp['min_drop_main_check_pp']} п.п. | — | меньше на {drop_pp:.1f} п.п. "
         f"[{-p1['hi'] * 100:.1f}; {-p1['lo'] * 100:.1f}] | {'да' if c1 else 'нет'} |",
         f"| П2.2: найдено опухолей на нашем val ниже не более чем на {rp['max_recall_drop_val_pp']} п.п. | {f(rec_o)} | {f(rec_n)} | "
         f"{'да' if c2 else 'нет'} (ниже на {rec_drop_pp:.2f} п.п.) |",
         f"| П3: менингиомы и гипофиз нашего val с типом «глиома» (среднее трех копий), рост ≤ {rules['glioma_shortcut']['max_rise_pp']} п.п. | "
         f"{g_o:.2%} | {g_n:.2%} | {'да' if c3 else 'нет'} |"]
    for part, key in (("наш val", "val"), ("внешний holdout", "ext_holdout"), ("П1", "main_check")):
        co, cn = Mo[key]["confidence"], Mn[key]["confidence"]
        L.append(f"| П5: «проверьте» помогает находить ошибки — {part} | {'да' if co['helps'] else 'нет'} (способ A) | "
                 f"{'да' if cn['helps'] else 'нет'} ({rule_new['name'].split(':')[0]}) | — |")
    if mode_js:
        L.append(f"| П6: определитель режима — точность «T1 с контрастом» ≥ 95% на каждом наборе | — | " +
                 "; ".join(f"{s} {mode_js['sets'][s]['t1c_precision']['k']} из {mode_js['sets'][s]['t1c_precision']['n']}"
                           for s in mode_js["checked"]) + f" | {'да' if mode_js['passes'] else 'нет'} |")
    if core_js:
        a, r = core_js["t1c_mean_of_seeds"], core_js["runs"]
        fl = next((v for v in r.values() if v["inputs"] == "t1c_flair"), None)
        L.append(f"| П7: ядро / отек на {core_js['patients']} отложенных пациентах UPENN (только отчет) | — | T1c: {a['core']:.3f} / {a['edema']:.3f}"
                 + (f"; T1c + FLAIR: {fl['core']['patients_mean']:.3f} / {fl['edema']['patients_mean']:.3f}" if fl else "") + " | без порога |")
    L += ["", f"**П2: {'условия 1 и 2 выполнены' if c1 and c2 else 'условия не выполнены'}** "
          + ("— замена возможна только после подтверждения руководителя проекта по этому сравнению." if c1 and c2 else "— `models/final` остается.")
          + (" П3 выполнено." if c3 else " П3 не выполнено → по П2 проверяется сочетание «новая модель контура + нынешняя модель типа»."), ""]

    # П1 подробно
    mo, mn = Mo["main_check"], Mn["main_check"]
    L += ["## П1 подробно: срезы T1 с контрастом без опухоли из независимых источников", "",
          f"Срезов {p1['n']}, пациентов {p1['patients']}. Разница «стало − было» {p1['diff']:+.1%} [{p1['lo']:+.1%}; {p1['hi']:+.1%}] (парно).", "",
          "| часть | было | стало |", "|---|---|---|", f"| все | {f(mo['all'])} | {f(mn['all'])} |"]
    L += [f"| {SRC_RU.get(s, s)} | {f(mo['by_source'].get(s))} | {f(r)} |" for s, r in mn["by_source"].items()]
    L += [f"| плоскость {PLANE_RU.get(p, p)} | {f(mo['by_plane'].get(p))} | {f(r)} |" for p, r in mn["by_plane"].items()]
    L += [f"| склероз, срез {'с очагом' if k == 'True' else 'без очага'} | {f(mo['ms_lesion'].get(k))} | {f(r)} |" for k, r in mn["ms_lesion"].items()]
    L += [f"| копии 0 / 1 / 2 | " + " / ".join(f"{x:.1%}" for x in mo["copies"]) + " | " + " / ".join(f"{x:.1%}" for x in mn["copies"]) + " |"]
    if Mn.get("vs_unknown_mode"):
        L += [f"| шванномы, режим не установлен (в П1 не входят) | {f(Mo.get('vs_unknown_mode'))} | {f(Mn['vs_unknown_mode'])} |"]
    so, sn = Mo["schwannoma"], Mn["schwannoma"]
    L += ["", "**Срезы со шванномой (незнакомая опухоль, тип не оценивается):**", "",
          "| мера | было | стало |", "|---|---|---|",
          f"| опухоль найдена | {f(so['found'])} | {f(sn['found'])} |",
          f"| названный тип | {so['types']} | {sn['types']} |",
          f"| совпадение контура | {fd(so['dice'])} | {fd(sn['dice'])} |", ""]

    # holdout того же источника
    L += ["## Отложенные пациенты UPENN-GBM и BraTS-MEN-RT (тот же источник, что в обучении кандидата)", "",
          "| мера | было | стало |", "|---|---|---|"]
    for s in Mn["ext_holdout"]:
        if s == "confidence":
            continue
        o, n = Mo["ext_holdout"][s], Mn["ext_holdout"][s]
        nm = SRC_RU.get(s, s)
        L += [f"| {nm}: ложная тревога на срезе без опухоли | {f(o['no_tumor_errors'])} | {f(n['no_tumor_errors'])} |",
              f"| {nm}: опухоль найдена | {f(o['found'])} | {f(n['found'])} |",
              f"| {nm}: тип верен у найденных | {f(o['type_correct'])} | {f(n['type_correct'])} |",
              f"| {nm}: совпадение контура | {fd(o['dice'])} | {fd(n['dice'])} |"]
        for p in n["by_plane"]:
            L += [f"| — {PLANE_RU.get(p, p)}: ложная тревога / найдено | {f(o['by_plane'][p]['no_tumor_errors'])} / {f(o['by_plane'][p]['found'])} | "
                  f"{f(n['by_plane'][p]['no_tumor_errors'])} / {f(n['by_plane'][p]['found'])} |"]
    L += [""]

    # режимы
    L += ["## По режимам МРТ (аксиальные срезы; одни и те же места; склероз — check, UPENN — val и holdout вместе)", "",
          "| набор | режим | было: ложная тревога | стало: ложная тревога | было: найдено | стало: найдено | стало: тип верен |",
          "|---|---|---|---|---|---|---|"]
    for key, n in Mn["modes"].items():
        o = Mo["modes"].get(key, {})
        s, m = key.split("|")
        L.append(f"| {SRC_RU.get(s, s)} | {MODE_RU.get(m, m)} | {f(o.get('no_tumor_errors'))} | {f(n['no_tumor_errors'])} | "
                 f"{f(o.get('found'))} | {f(n.get('found'))} | {f(n.get('type_correct'))} |")
    L += ["", f"Части: {', '.join(Mn.get('modes_parts', []))}.", ""]

    # П5
    L += ["## П5: помогает ли пометка «проверьте» находить ошибки", "",
          "Условия: доля ошибок среди «проверьте» ≥ 3 раз выше, чем среди «уверенно»; поймано ≥ 50% ошибок; помечено ≤ 25% снимков. "
          "Способ E0.6 выбран на val; для нынешней системы он показан для справки.", ""]
    L += conf_table(pred, rules, rule_new) + [""]

    # наш val и старый test
    L += ["## Наш val и старый test", "", "Наш val — промежуточная проверка (порог и «проверьте» кандидата подобраны на нем — числа кандидата "
          "оптимистичны). Старый test уже просмотрен при разборе ошибок — только «рядом».", "",
          "| мера | val: было | val: стало | test: было | test: стало |", "|---|---|---|---|---|"]
    for lab, key in (("найдено опухолей", "recall"), ("ложная тревога «нет опухоли»", "no_tumor_errors"), ("тип верен у найденных", "type_correct")):
        L.append(f"| {lab} | {f(Mo['val'][key])} | {f(Mn['val'][key])} | {f(Mo.get('test', {}).get(key))} | {f(Mn.get('test', {}).get(key))} |")
    for s in ("figshare", "figshare_crop", "other"):
        L.append(f"| — найдено, источник {s} | {f(Mo['val']['recall_by_source'].get(s))} | {f(Mn['val']['recall_by_source'].get(s))} | "
                 f"{f(Mo.get('test', {}).get('recall_by_source', {}).get(s))} | {f(Mn.get('test', {}).get('recall_by_source', {}).get(s))} |")
    for m in sorted(Mn["val"]["no_tumor_errors_by_mode"]):
        L.append(f"| — ложная тревога, режим «{m}» | {f(Mo['val']['no_tumor_errors_by_mode'].get(m))} | {f(Mn['val']['no_tumor_errors_by_mode'].get(m))} | "
                 f"{f(Mo.get('test', {}).get('no_tumor_errors_by_mode', {}).get(m))} | {f(Mn.get('test', {}).get('no_tumor_errors_by_mode', {}).get(m))} |")
    L.append(f"| совпадение контура | {fd(Mo['val']['dice'])} | {fd(Mn['val']['dice'])} | {fd(Mo.get('test', {}).get('dice'))} | {fd(Mn.get('test', {}).get('dice'))} |")
    L += [""]
    if "ext_val" in Mn:
        L += ["## Внешний val (тот же источник, что в обучении кандидата; промежуточная проверка)", "", "| мера | было | стало |", "|---|---|---|"]
        for s in Mn["ext_val"]:
            if s == "confidence":
                continue
            o, n = Mo["ext_val"][s], Mn["ext_val"][s]
            L += [f"| {SRC_RU.get(s, s)}: ложная тревога | {f(o['no_tumor_errors'])} | {f(n['no_tumor_errors'])} |",
                  f"| {SRC_RU.get(s, s)}: найдено / тип верен | {f(o['found'])} / {f(o['type_correct'])} | {f(n['found'])} / {f(n['type_correct'])} |"]
        L += [""]

    # П6 и П7 — ссылки
    if mode_js:
        L += ["## П6: определитель режима", "", (out_dir / "mode_check.md").read_text(encoding="utf-8").split("\n", 2)[2], ""]
    if core_js:
        L += ["## П7: ядро и отек", "", (out_dir / "core_check.md").read_text(encoding="utf-8").split("\n", 2)[2], ""]

    # примеры
    rows = pick_examples(pred["before"], pred["after"])
    cfg = DataConfig.from_yaml(args.config)
    before_sys = rules["system"]["copies"][0]
    cls_seg = {"before": (find_run(before_sys["classifier"]), find_run(before_sys["segmenter"])),
               "after": (find_run(system["copies"][0]["classifier"]), find_run(system["copies"][0]["segmenter"]))}
    mk = masks_for(rows, cls_seg, cfg)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    png = args.out.parent / "examples_val.png"
    sheet(rows, mk, png)
    L += ["## Примеры верных и ошибочных ответов (только val)", "",
          f"`{png.as_posix()}` — {len(rows)} снимков; группы и отбор (все, если ≤ 3, иначе 3 случайных, seed 42): "
          + "; ".join(sorted({n for n, _ in rows})) + ".", ""]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(L), encoding="utf-8")
    js = {"p1_paired": p1, "p2": {"drop_pp": drop_pp, "cond1": bool(c1), "recall_drop_pp": rec_drop_pp, "cond2": bool(c2)},
          "p3": {"before": g_o, "after": g_n, "passes": bool(c3)}, "before": Mo, "after": Mn}
    args.out.with_suffix(".json").write_text(json.dumps(js, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
