"""Узнается ли набор данных по техническому виду снимка после единого препроцессинга (как проба shortcut шага 1.3).

Вопрос (2026-11-14): приводили ли мы снимки к одному виду? Да: все снимки идут
через `src.preprocess.preprocess` (обрезка полей, 256×256, серый, яркость: фон → 0, 99,5-й перцентиль тканей → 1).
Проверяется, остаются ли после этого различия, по которым можно узнать источник снимка.

Определение (записано и закоммичено до запуска):
- Снимки: только обучающая часть этапа 3 (`load_pool`, таблица poolx, split == train): наш train 2875 и внешний
  train 3170 (UPENN-GBM, BraTS-MEN-RT). Проверочные, отложенные и проверочные внешние наборы не используются.
- Вход: снимок ровно в том виде, в каком он идет в модель (кэш препроцессинга 256×256); второй вариант — после
  одной случайной аугментации обучения с рандомизацией домена (`build_transforms(True, True, 42)`), как в 1.6.
- Признаки (по области головы — пиксели ярче 0,02): средняя яркость, разброс яркости, 10/50/90-й перцентили,
  доля пересвеченных (> 0,95) и темных внутри рамки головы (< 0,05), число уровней серого, энтропия гистограммы,
  резкость (дисперсия лапласиана), шум (разброс остатка после легкого размытия), «блочность» JPEG (перепад на
  границах блоков 8×8 к перепаду внутри), доля кадра под головой, пропорции рамки головы, заполненность рамки.
- Модель: HistGradientBoosting (200 итераций, сбалансированные веса), 5 фолдов StratifiedGroupKFold по пациенту
  (группа), seed 42; меры — AUC «один против остальных» и сбалансированная точность по предсказаниям вне фолда.
- Задачи: (А) набор: наш / UPENN-GBM / BraTS-MEN-RT; (Б) «опухоль есть / нет» по всей обучающей части;
  (В) «опухоль есть / нет» внутри каждого набора.
- Толкование (до запуска): AUC ≥ 0,9 — узнается легко; 0,7–0,9 — частично; < 0,7 — почти не узнается.
  Это проба: она показывает, что различие ЕСТЬ в простых признаках; что им пользуется сеть, она не доказывает.

    python -m src.check_source_probe
Результат: reports/source_probe_report.md, reports/source_probe/{features.csv, top_features.png}.
"""
from __future__ import annotations

import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

from src.dataset import DataConfig, build_transforms, load_pool

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SEED = 42
OUT = Path("reports/source_probe")
REPORT = Path("reports/source_probe_report.md")
FEAT_RU = {
    "mean": "средняя яркость головы", "std": "разброс яркости", "p10": "10-й перцентиль яркости",
    "p50": "медиана яркости", "p90": "90-й перцентиль яркости", "sat": "доля пересвеченных",
    "dark": "доля темных внутри рамки головы", "levels": "число уровней серого", "entropy": "энтропия гистограммы",
    "sharp": "резкость (лапласиан)", "noise": "шум (остаток после размытия)", "block": "блочность JPEG (8×8)",
    "area": "доля кадра под головой", "aspect": "пропорции рамки головы", "fill": "заполненность рамки головы",
}
FEATS = list(FEAT_RU)
DS_RU = {"ours": "наш набор", "upenn": "UPENN-GBM", "menrt": "BraTS-MEN-RT"}
GROUP_RU = {"ours_fig": "наш: срезы Figshare (опухоль)", "ours_oth": "наш: опухоли других источников",
            "ours_no": "наш: нет опухоли", "upenn_t": "UPENN: опухоль", "upenn_no": "UPENN: нет опухоли",
            "menrt_t": "MEN-RT: опухоль", "menrt_no": "MEN-RT: нет опухоли"}


def features(img: np.ndarray) -> dict:
    x = img.astype(np.float32) / 255
    head = x > 0.02
    if head.sum() < 100:
        return {f: np.nan for f in FEATS}
    ys, xs = np.nonzero(head)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    t = x[head]
    q = img[head]
    hist = np.bincount(q, minlength=256) / q.size
    nz = hist[hist > 0]
    lap = cv2.Laplacian(img, cv2.CV_64F)
    res = img.astype(np.float32) - cv2.GaussianBlur(img, (3, 3), 0.8).astype(np.float32)
    box = img[y0:y1, x0:x1].astype(np.float32)
    dh, dv = np.abs(np.diff(box, axis=1)), np.abs(np.diff(box, axis=0))
    on_h = (np.arange(dh.shape[1]) + x0 + 1) % 8 == 0
    on_v = (np.arange(dv.shape[0]) + y0 + 1) % 8 == 0
    edge = np.concatenate([dh[:, on_h].ravel(), dv[on_v].ravel()]).mean()
    inner = np.concatenate([dh[:, ~on_h].ravel(), dv[~on_v].ravel()]).mean()
    return {
        "mean": float(t.mean()), "std": float(t.std()), "p10": float(np.percentile(t, 10)),
        "p50": float(np.percentile(t, 50)), "p90": float(np.percentile(t, 90)), "sat": float((t > 0.95).mean()),
        "dark": float((x[y0:y1, x0:x1] < 0.05).mean()), "levels": int(len(nz)),
        "entropy": float(-(nz * np.log2(nz)).sum()), "sharp": float(lap[head].var()), "noise": float(res[head].std()),
        "block": float(edge / max(inner, 1e-6)), "area": float(head.mean()), "aspect": float((x1 - x0) / (y1 - y0)),
        "fill": float(head[y0:y1, x0:x1].mean()),
    }


def cv_proba(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, class_weight="balanced", random_state=SEED)
    return cross_val_predict(clf, X, y, groups=groups, cv=cv, method="predict_proba")


def word(auc: float) -> str:
    return "легко" if auc >= 0.9 else ("частично" if auc >= 0.7 else "почти не узнается")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0, help="только для проверки кода: случайные N снимков, отчет не пишется")
    args = ap.parse_args()
    sys.stdout.reconfigure(errors="replace")
    out, report = OUT, REPORT
    cfg = replace(DataConfig.from_yaml("configs/base.yaml"), table="poolx")
    table, cache = load_pool(cfg)
    d = table[table["split"] == "train"].reset_index(drop=True)
    if args.limit:
        d = d.sample(args.limit, random_state=SEED).reset_index(drop=True)
        out, report = Path(tempfile.mkdtemp(prefix="source_probe_dev_")), None
    out.mkdir(parents=True, exist_ok=True)
    is_ext = d["origin"] == "ext"
    d["ds"] = np.where(is_ext, d["source"], "ours")
    d["tumor"] = np.where(is_ext, d["has_tumor"].astype(str).str.lower() == "true", d["cls"] != "no_tumor")
    d["grp"] = np.select(
        [~is_ext & (d["cls"] == "no_tumor"), ~is_ext & (d["source"] == "figshare"), ~is_ext,
         is_ext & d["tumor"], is_ext & ~d["tumor"]],
        ["ours_no", "ours_fig", "ours_oth", d["ds"] + "_t", d["ds"] + "_no"], default="?")
    d["cvgroup"] = d["origin"] + ":" + d["group"].astype(str)
    print(d.groupby("grp").size().to_dict(), flush=True)

    tf = build_transforms(True, True, SEED)
    rows = []
    for i, p in enumerate(d["path"]):
        img = cache.image(p)
        a = tf(image=img)["image"]
        rows.append({**{f"{k}": v for k, v in features(img).items()}, **{f"aug_{k}": v for k, v in features(a).items()}})
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(d)}", flush=True)
    F = pd.concat([d[["path", "ds", "grp", "tumor", "cvgroup"]], pd.DataFrame(rows)], axis=1)
    F.to_csv(out / "features.csv", index=False, encoding="utf-8")

    res: dict = {}
    for var, cols in (("как идет в модель", FEATS), ("после аугментаций с рандомизацией домена", [f"aug_{f}" for f in FEATS])):
        X, g = F[cols], F["cvgroup"].to_numpy()
        ds = F["ds"].to_numpy()
        order = ["ours", "upenn", "menrt"]
        pr = cv_proba(X, ds, g)
        classes = sorted(order)  # порядок столбцов predict_proba — отсортированные метки
        auc_ds = {c: roc_auc_score(ds == c, pr[:, classes.index(c)]) for c in order}
        pred = np.array(classes)[pr.argmax(1)]
        cm = confusion_matrix(ds, pred, labels=order)
        yt = F["tumor"].to_numpy().astype(int)
        auc_t = roc_auc_score(yt, cv_proba(X, yt, g)[:, 1])
        within = {}
        for c in order:
            m = ds == c
            within[c] = roc_auc_score(yt[m], cv_proba(X[m], yt[m], g[m])[:, 1])
        single = {f: max(a, 1 - a) for f, a in
                  ((f, roc_auc_score(ds == "ours", F[col])) for f, col in zip(FEATS, cols))}
        res[var] = {"auc_ds": auc_ds, "bal_acc": balanced_accuracy_score(ds, pred), "cm": cm, "auc_tumor": auc_t,
                    "within": within, "single_ours": single}
        print(var, {k: round(v, 3) for k, v in auc_ds.items()}, round(auc_t, 3), {k: round(v, 3) for k, v in within.items()}, flush=True)

    # рисунок: 4 признака, лучше всего отделяющие наш набор от внешних (без аугментаций)
    top = sorted(res["как идет в модель"]["single_ours"], key=lambda f: -res["как идет в модель"]["single_ours"][f])[:4]
    grps = list(GROUP_RU)
    colors = ["#2a78d6", "#7aa9e6", "#eda100", "#1baf7a", "#8fd6b9", "#eb6834", "#f3ab8a"]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.6))
    for ax, f in zip(axes, top):
        data = [F.loc[F["grp"] == k, f].dropna().to_numpy() for k in grps]
        bp = ax.boxplot(data, vert=False, showfliers=False, patch_artist=True, widths=0.6)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
        ax.set_yticks(range(1, len(grps) + 1), [GROUP_RU[k] for k in grps] if ax is axes[0] else [""] * len(grps), fontsize=9)
        ax.invert_yaxis()
        ax.set_title(f"{FEAT_RU[f]}\n(AUC «наш / внешний» {res['как идет в модель']['single_ours'][f]:.2f})", fontsize=10)
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle("Обучающие снимки после единого препроцессинга: признаки, по которым наш набор сильнее всего отличается от внешних", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "top_features.png", dpi=110)
    plt.close(fig)

    n = F.groupby("grp").size()
    L = ["# Узнается ли набор по виду снимка после единого препроцессинга", "",
         "`python -m src.check_source_probe`. Только обучающая часть этапа 3 (наш train + внешний train), снимок в том виде, "
         "в каком он идет в модель (256×256 после `preprocess`), и после одной случайной аугментации обучения. Простая модель "
         "(градиентный бустинг) по 15 техническим признакам, 5 фолдов по пациентам. Толкование AUC (записано до запуска): "
         "≥ 0,9 — узнается легко; 0,7–0,9 — частично; < 0,7 — почти не узнается. Проба показывает, что различие есть в простых "
         "признаках; что им пользуется нейросеть, она не доказывает.", "",
         "Снимков: " + ", ".join(f"{GROUP_RU[k]} {int(n.get(k, 0))}" for k in GROUP_RU) + ".", "",
         "| задача | как идет в модель | после аугментаций |", "|---|---|---|"]
    a, b = res["как идет в модель"], res["после аугментаций с рандомизацией домена"]
    for c in ("ours", "upenn", "menrt"):
        L.append(f"| набор: {DS_RU[c]} против остальных, AUC | {a['auc_ds'][c]:.3f} ({word(a['auc_ds'][c])}) | {b['auc_ds'][c]:.3f} ({word(b['auc_ds'][c])}) |")
    L.append(f"| набор из трех, сбалансированная точность (случайно — 0,333) | {a['bal_acc']:.3f} | {b['bal_acc']:.3f} |")
    L.append(f"| «опухоль есть / нет», все обучающие снимки, AUC | {a['auc_tumor']:.3f} | {b['auc_tumor']:.3f} |")
    for c in ("ours", "upenn", "menrt"):
        L.append(f"| «опухоль есть / нет» внутри набора {DS_RU[c]}, AUC | {a['within'][c]:.3f} | {b['within'][c]:.3f} |")
    L += ["", "Какой набор угадывает модель (строки — настоящий, столбцы — ответ; как идет в модель):", "",
          "| | наш | UPENN-GBM | BraTS-MEN-RT |", "|---|---|---|---|"]
    for c, row in zip(("ours", "upenn", "menrt"), a["cm"]):
        L.append(f"| {DS_RU[c]} | " + " | ".join(str(int(v)) for v in row) + " |")
    L += ["", "Отдельные признаки: AUC «наш набор / внешние» (как идет в модель → после аугментаций):", ""]
    so_a, so_b = a["single_ours"], b["single_ours"]
    for f in sorted(FEATS, key=lambda f: -so_a[f]):
        L.append(f"- {FEAT_RU[f]}: {so_a[f]:.3f} → {so_b[f]:.3f}")
    med = F.groupby("grp")[FEATS].median().reindex(list(GROUP_RU))
    med.index = [GROUP_RU[k] for k in med.index]
    med.columns = [FEAT_RU[c] for c in med.columns]
    mt = med.T.round(3)
    L += ["", "Медианы признаков по группам (как идет в модель):", "",
          "| признак | " + " | ".join(mt.columns) + " |", "|---" * (len(mt.columns) + 1) + "|"]
    L += [f"| {i} | " + " | ".join(f"{v:g}" for v in r) + " |" for i, r in mt.iterrows()]
    L += ["",
          "![](source_probe/top_features.png)", ""]
    if report is not None:
        report.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:30]))


if __name__ == "__main__":
    main()
