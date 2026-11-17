"""Срезы из объемных снимков внешних наборов → PNG, таблица и кэш `ext_256`.

Наборы (скачаны в `data/external/`, журнал проекта, 2026-11-12):
  - `upenn` — UPENN-GBM, глиобластома, T1 с контрастом + FLAIR, обводка 1 — некроз, 2 — отек,
    3 — контрастируемая часть; атлас SRI24, 1 мм;
  - `menrt` — BraTS-MEN-RT, менингиома, T1 с контрастом, одна обводка GTV;
  - `ms` — рассеянный склероз (Любляна), T1 с контрастом; опухолей нет, только для проверки.

Пациенты UPENN и MEN-RT делятся на train / val / holdout (60 / 20 / 20, seed 42); деление
записывается в `data/processed/external_split.csv` и при повторном запуске не меняется. Склероз —
часть `check` (только проверка).

Из каждого объема в трех плоскостях (аксиальная, корональная, сагиттальная) берутся:
  - срезы с опухолью — площадь ядра опухоли не меньше `--min-tumor-mm2`, до `--k` срезов,
    равномерно по протяженности опухоли;
  - срезы без опухоли того же пациента — ни одной метки на срезе и расстояние до ближайшего
    среза с любой меткой не меньше `--gap-upenn` (метки UPENN включают отек) или `--gap-menrt`
    (у MEN-RT отек не обведен, поэтому отступ больше), до `--k` срезов;
  - у склероза — до `--k-ms` срезов, с пометкой, есть ли на срезе очаг склероза.
У склероза и шванномы (только проверка) — одна плоскость, в которой снимал аппарат (остальные
у них — размытый пересчет), без правила про лицо. Для каждого среза с опухолью записывается
`enh_ratio` — во сколько раз обведенная область ярче полоски ткани вокруг нее: у MEN-RT часть
пациентов после операции, и обводка там — ложе удаленной опухоли (темная полость), а не опухоль.
Срезы MEN-RT с опухолью и `enh_ratio < --min-enh-menrt` (по умолчанию 1,1; решение команды
2026-11-12) в таблицу не входят, их список — `reports/external/menrt_dark_excluded.csv`.
Берутся только срезы, где голова занимает не меньше `--min-head` от наибольшей площади в этой
плоскости (без крайних срезов). Лицо у UPENN и MEN-RT стерто ради обезличивания, и на срезах через
лицо видны блочные помехи; опухоли чаще выше, поэтому помехи стали бы подсказкой «нет опухоли».
Поэтому для всех срезов (с опухолью и без, у всех наборов) одно правило: аксиальные — не ниже
`--ax-min-frac` высоты головы, корональные — не впереди `--cor-max-frac` ее длины (считая от
затылка). Все, что вне головы (метки рамки для облучения у части MEN-RT), обнуляется. Срез приводится к квадратному пикселю `--px-mm` мм и к 8 битам
(окно 0–99,8-й перцентиль среза), ориентация — как принято у рентгенологов.

Маска для модели контура — ядро опухоли: у UPENN метки 1 + 3 (некроз и контрастируемая часть,
отек не входит), у MEN-RT — GTV. У срезов без опухоли маска пустая и достоверная (`has_mask`).
У UPENN дополнительно сохраняются FLAIR того же среза и полная обводка 0–3 (для модели ядра и
отека).

    python -m src.build_external_slices
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from dataclasses import fields
from pathlib import Path

import cv2
import nibabel as nib
import nibabel.processing
import numpy as np
import pandas as pd

from src.dataset import DataConfig, build_cache
from src.io_utils import read_image
from src.preprocess import PreprocMeta, mask_to_model
from src.labels import CLS_RU

SEED = 42
EXT = Path("data/external")
PLANES = ("ax", "cor", "sag")
PLANE_RU = {"ax": "аксиальная", "cor": "корональная", "sag": "сагиттальная"}


# ---------------------------------------------------------------- чтение и срезы


LPS2RAS = np.diag([-1.0, -1.0, 1.0, 1.0])


def itk_affine(path: Path) -> np.ndarray:
    """Файл ITK `MatrixOffsetTransformBase_double_3_3` → матрица 4×4 (точки LPS, мм):
    T(x) = A (x − c) + c + t. По соглашению ITK переводит точку неподвижного снимка (FLAIR) в точку
    подвижного (T1WKS и т.д.)."""
    kv = dict(line.split(":", 1) for line in Path(path).read_text().splitlines() if ":" in line)
    p, c = np.array(kv["Parameters"].split(), float), np.array(kv["FixedParameters"].split(), float)
    a, t = p[:9].reshape(3, 3), p[9:12]
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = a, c + t - a @ c
    return m


def ms_to_ref(src: Path, ref: Path, src_xfm: Path | None, ref_xfm: Path, order: int) -> nib.Nifti1Image:
    """Склероз (Любляна): снимок или обводку src — в сетку ref через файлы совмещения авторов
    (`*_to_FLAIR.txt`, общая точка — FLAIR). src_xfm=None — src уже в пространстве FLAIR (FLAIR, обводка).
    По координатам аппарата совмещать нельзя: FLAIR повернут на ~11° и сдвинут на ~2 см; направление
    проверено по взаимной информации с T1WKS (0,37–0,39 против 0,04–0,05 без файлов; проверено 2026-11-12)."""
    from scipy.ndimage import affine_transform
    s, r = nib.load(str(src)), nib.load(str(ref))
    lps = (itk_affine(src_xfm) if src_xfm else np.eye(4)) @ np.linalg.inv(itk_affine(ref_xfm))  # точка ref → точка src
    m = np.linalg.inv(s.affine) @ LPS2RAS @ lps @ LPS2RAS @ r.affine  # воксель ref → воксель src
    data = affine_transform(np.asarray(s.dataobj, np.float32), m, output_shape=r.shape[:3], order=order)
    return nib.Nifti1Image(data, r.affine)


def load(path: Path | nib.Nifti1Image, order: int = 1, like: nib.Nifti1Image | None = None,
         labels: bool = False) -> tuple[np.ndarray, tuple[float, ...]]:
    """Объем в осях RAS (x — вправо, y — вперед, z — вверх) и шаг в мм. Если задан like — объем
    пересчитывается в сетку like (order 0 — для меток). Снимок остается в исходном типе (int16 и т.п. —
    вдвое меньше памяти, чем float32), метки — uint8. path может быть уже загруженным снимком."""
    img = path if isinstance(path, nib.Nifti1Image) else nib.load(str(path))
    if like is not None:
        img = nibabel.processing.resample_from_to(img, like, order=order)
    proxy = img.dataobj
    if isinstance(proxy, np.ndarray):
        arr = proxy
    elif getattr(proxy, "slope", 1.0) in (0.0, 1.0) and getattr(proxy, "inter", 0.0) == 0.0:
        arr = proxy.get_unscaled()
    else:
        arr = np.asarray(proxy, dtype=np.float32)
    ornt = nib.orientations.io_orientation(img.affine)
    arr = np.ascontiguousarray(nib.orientations.apply_orientation(arr, ornt))
    z_in = img.header.get_zooms()[:3]
    zooms = [0.0, 0.0, 0.0]
    for i, (ax, _) in enumerate(ornt):
        zooms[int(ax)] = float(z_in[i])
    if labels:
        return (np.rint(arr) if arr.dtype.kind == "f" else arr).astype(np.uint8), tuple(zooms)
    return arr, tuple(zooms)


def axis_of(plane: str) -> int:
    return {"sag": 0, "cor": 1, "ax": 2}[plane]


def take(vol: np.ndarray, zooms: tuple[float, ...], plane: str, k: int) -> tuple[np.ndarray, tuple[float, float]]:
    """Срез k в плоскости plane, повернутый как на снимке: верх — вперед (аксиальная) или вверх,
    правая сторона пациента — слева на снимке; у сагиттальной лицо слева. Возвращает срез и шаг
    (строки, столбцы) в мм."""
    if plane == "ax":
        s, sp = vol[:, :, k].T[::-1, ::-1], (zooms[1], zooms[0])
    elif plane == "cor":
        s, sp = vol[:, k, :].T[::-1, ::-1], (zooms[2], zooms[0])
    else:
        s, sp = vol[k, :, :].T[::-1, ::-1], (zooms[2], zooms[1])
    return np.ascontiguousarray(s), sp


def to_px(s: np.ndarray, sp: tuple[float, float], px_mm: float, nearest: bool) -> np.ndarray:
    h, w = s.shape
    nh, nw = max(1, round(h * sp[0] / px_mm)), max(1, round(w * sp[1] / px_mm))
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.resize(s.astype(np.float32), (nw, nh), interpolation=interp)


def to_u8(s: np.ndarray) -> np.ndarray:
    hi = np.percentile(s, 99.8)
    if hi <= 0:
        return np.zeros(s.shape, np.uint8)
    return np.clip(s / hi * 255, 0, 255).round().astype(np.uint8)


def head_thr(vol: np.ndarray) -> float:
    return float(0.1 * np.percentile(vol[::2, ::2, ::2], 99))


def keep_head(img: np.ndarray, raw: np.ndarray, thr: float) -> np.ndarray:
    """Обнуляет все вне головы: наибольшая связная область выше порога, с закрытыми щелями и
    заполненными полостями."""
    m = (raw > thr).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return img
    big = (lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    flood = big.copy()
    cv2.floodFill(flood, np.zeros((big.shape[0] + 2, big.shape[1] + 2), np.uint8), (0, 0), 1)
    head = big | (1 - flood)  # полости внутри головы — тоже голова
    return np.where(head > 0, img, 0).astype(img.dtype)


def enhancement(raw: np.ndarray, m: np.ndarray, thr: float, px_mm: float, ring_mm: float = 5.0) -> float:
    """Во сколько раз обведенная область ярче полоски ткани вокруг нее (медианы). Опухоль,
    накапливающая контраст, светлее мозга (> 1); полость после операции темнее (< 1)."""
    if m.sum() < 5:
        return -1.0
    r = max(1, round(ring_mm / px_mm))
    ring = cv2.dilate(m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0
    ring &= ~m & (raw > thr)
    if ring.sum() < 5:
        return -1.0
    ref = float(np.median(raw[ring]))
    return round(float(np.median(raw[m])) / ref, 3) if ref > 0 else -1.0


def spread(cands: np.ndarray, k: int) -> list[int]:
    """До k индексов, равномерно по списку кандидатов (без случайности)."""
    if len(cands) == 0:
        return []
    if len(cands) <= k:
        return [int(c) for c in cands]
    pos = np.unique(np.linspace(0, len(cands) - 1, k + 2)[1:-1].round().astype(int))  # без повторов
    return [int(cands[p]) for p in pos]


def dist_to_labels(any_lab: np.ndarray, step_mm: float) -> np.ndarray:
    """Расстояние (мм) от каждого среза до ближайшего среза с меткой; inf, если меток нет."""
    idx = np.flatnonzero(any_lab)
    n = len(any_lab)
    if len(idx) == 0:
        return np.full(n, np.inf)
    return np.abs(np.arange(n)[:, None] - idx[None, :]).min(1) * step_mm


# ---------------------------------------------------------------- пациенты


def patients() -> pd.DataFrame:
    rows = []
    for p in sorted((EXT / "upenn_gbm" / "nifti").glob("*_seg.nii.gz")):
        sid = p.name.removesuffix("_seg.nii.gz")
        rows.append({"source": "upenn", "patient": sid, "cls": "glioma",
                     "t1c": f"upenn_gbm/nifti/{sid}_ce-gd_T1w.nii.gz", "flair": f"upenn_gbm/nifti/{sid}_FLAIR.nii.gz",
                     "lab": p.relative_to(EXT).as_posix()})
    for p in sorted((EXT / "brats_men_rt" / "nifti").glob("*_gtv.nii.gz")):
        sid = p.name.removesuffix("_gtv.nii.gz")
        t1c = p.with_name(f"{sid}_t1c.nii.gz")
        if t1c.exists() and t1c.stat().st_size > 0:
            rows.append({"source": "menrt", "patient": sid, "cls": "meningioma", "t1c": t1c.relative_to(EXT).as_posix(),
                         "flair": "", "lab": p.relative_to(EXT).as_posix()})
    for d in sorted((EXT / "ms_ljubljana").glob("patient*")):
        rows.append({"source": "ms", "patient": d.name, "cls": "no_tumor", "t1c": f"ms_ljubljana/{d.name}/T1WKS.nii.gz",
                     "flair": "", "lab": f"ms_ljubljana/{d.name}/consensus_gt.nii.gz"})
    vs = EXT / "vs_nifti" / "series.csv"  # шванномы (src.convert_vs): только серии с обводкой
    if vs.exists():
        t = pd.read_csv(vs, keep_default_na=False)
        # с обводкой и непустой: при пустой (перенос с T2 вне полосы T1) все срезы ошибочно считались бы срезами без опухоли
        for r in t[(t["status"] == "ок") & (t["seg"] != "") & (t["tumor_voxels"].astype(int) > 0)].itertuples(index=False):
            rows.append({"source": "vs", "patient": r.stem, "cls": "schwannoma", "t1c": f"vs_nifti/{r.stem}_t1.nii.gz",
                         "flair": "", "lab": f"vs_nifti/{r.stem}_seg.nii.gz", "group": f"vs:{r.patient}", "mode": r.mode})
    out = pd.DataFrame(rows)
    out["group"] = out.get("group", pd.Series(index=out.index, dtype=object)).fillna(out["source"] + ":" + out["patient"])
    out["mode"] = out.get("mode", pd.Series(index=out.index, dtype=object)).fillna("t1c")
    return out


def split_patients(pts: pd.DataFrame, path: Path) -> pd.DataFrame:
    """train / val / holdout = 60 / 20 / 20 по пациентам внутри каждого набора; склероз — check.
    Уже записанное деление не меняется (новые пациенты дописываются)."""
    old = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["source", "patient", "split"])
    known = set(zip(old["source"], old["patient"]))
    new = []
    for src, g in pts.groupby("source"):
        todo = g[[(src, p) not in known for p in g["patient"]]]
        if todo.empty:
            continue
        if src in ("ms", "vs"):  # только проверка
            new.append(pd.DataFrame({"source": src, "patient": todo["patient"], "split": "check"}))
            continue
        order = todo["patient"].sample(frac=1, random_state=SEED).tolist()
        n = len(order)
        n_tr, n_va = round(0.6 * n), round(0.2 * n)
        parts = ["train"] * n_tr + ["val"] * n_va + ["holdout"] * (n - n_tr - n_va)
        new.append(pd.DataFrame({"source": src, "patient": order, "split": parts}))
    out = pd.concat([old] + new, ignore_index=True) if new else old
    out.to_csv(path, index=False)
    return out


def process(pt: pd.Series, split: str, out_dir: Path, args) -> list[dict]:
    t1, zooms = load(EXT / pt.t1c)
    if pt.source == "ms":  # обводка очагов — в сетке FLAIR; переносится через файлы совмещения авторов
        d = (EXT / pt.t1c).parent
        lab, _ = load(ms_to_ref(EXT / pt.lab, EXT / pt.t1c, None, d / "T1WKS_to_FLAIR.txt", order=0), labels=True)
    else:
        lab, _ = load(EXT / pt.lab, order=0, labels=True)
    if lab.shape != t1.shape:
        raise ValueError(f"{pt.patient}: обводка {lab.shape} не совпадает со снимком {t1.shape}")
    fl = load(EXT / pt.flair)[0] if pt.flair else None
    core = np.isin(lab, (1, 3)) if pt.source == "upenn" else (lab > 0)
    gap = args.gap_menrt if pt.source == "menrt" else args.gap_upenn
    min_tumor = args.min_tumor_vs if pt.source == "vs" else args.min_tumor_mm2
    planes = PLANES
    if pt.source in ("vs", "ms"):  # только плоскость, в которой снимали (по наибольшему шагу)
        planes = (("sag", "cor", "ax")[int(np.argmax(zooms))] if max(zooms) > 1.3 * min(zooms) else "ax",)

    thr = head_thr(t1)
    head = t1 > thr
    rows = []
    for plane in planes:
        ax = axis_of(plane)
        other = tuple(a for a in range(3) if a != ax)
        inplane = np.prod([zooms[a] for a in other])
        area = head.sum(axis=other).astype(np.float64)
        ok_head = area >= args.min_head * area.max()
        lo, hi = np.flatnonzero(ok_head)[[0, -1]]
        level = np.arange(len(area))
        if pt.source in ("vs", "ms"):  # только проверка: правило про лицо не нужно
            pass
        elif plane == "ax":  # выше лица
            ok_head &= level >= lo + args.ax_min_frac * (hi - lo)
        elif plane == "cor":  # позади лица
            ok_head &= level <= lo + args.cor_max_frac * (hi - lo)
        core_mm2 = core.sum(axis=other) * inplane
        any_lab = (lab > 0).sum(axis=other) > 0
        dist = dist_to_labels(any_lab, zooms[ax])
        if pt.source == "ms":
            picks = [(k, False) for k in spread(np.flatnonzero(ok_head), args.k_ms)]
        else:
            tum = np.flatnonzero(ok_head & (core_mm2 >= min_tumor))
            free = np.flatnonzero(ok_head & ~any_lab & (dist >= gap))
            k_free = args.k_ms if pt.source == "vs" else args.k
            picks = [(k, True) for k in spread(tum, args.k)] + [(k, False) for k in spread(free, k_free)]
        for k, has_tumor in picks:
            stem = f"{pt.source}/{pt.patient}_{plane}{k:03d}"
            raw = to_px(*take(t1, zooms, plane, k), args.px_mm, nearest=False)
            img = keep_head(to_u8(raw), raw, thr)
            lab2 = to_px(*take(lab, zooms, plane, k), args.px_mm, nearest=True).astype(np.uint8)
            m = (np.isin(lab2, (1, 3)) if pt.source == "upenn" else (lab2 > 0)) if pt.source != "ms" else np.zeros_like(lab2, bool)
            enh = enhancement(raw, m, thr, args.px_mm) if has_tumor else -1.0
            (out_dir / pt.source).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_dir / f"{stem}_t1c.png"), img)
            cv2.imwrite(str(out_dir / f"{stem}_mask.png"), m.astype(np.uint8) * 255)
            row = {"root": "ext", "path": f"{stem}_t1c.png", "mask": f"{stem}_mask.png", "has_mask": True,
                   "cls": pt.cls if has_tumor else "no_tumor", "split": split, "source": pt.source,
                   "patient": pt.patient, "group": pt.group, "plane": plane, "slice": k,
                   "mode": pt.mode, "has_tumor": has_tumor, "tumor_mm2": round(float(core_mm2[k]), 1),
                   "dist_mm": round(float(dist[k]), 1) if np.isfinite(dist[k]) else -1.0,
                   "ms_lesion": bool(pt.source == "ms" and any_lab[k]), "enh_ratio": enh, "flair": "", "label4": "", "lesion": ""}
            if pt.source == "ms":
                cv2.imwrite(str(out_dir / f"{stem}_lesion.png"), (lab2 > 0).astype(np.uint8) * 255)
                row["lesion"] = f"{stem}_lesion.png"
            if fl is not None:
                cv2.imwrite(str(out_dir / f"{stem}_flair.png"), keep_head(to_u8(to_px(*take(fl, zooms, plane, k), args.px_mm, nearest=False)), raw, thr))
                cv2.imwrite(str(out_dir / f"{stem}_lab.png"), lab2)
                row["flair"], row["label4"] = f"{stem}_flair.png", f"{stem}_lab.png"
            rows.append(row)
    return rows


# ---------------------------------------------------------------- проверки и отчет


def _to_model(mask: np.ndarray, g: pd.Series) -> np.ndarray:
    """Маска исходного среза → координаты модели по геометрии из таблицы кэша (bg и hi для маски не нужны)."""
    meta = PreprocMeta(**{f.name: g[f.name] for f in fields(PreprocMeta) if f.name in g}, size=256, bg=0.0, hi=255.0)
    return mask_to_model(mask, meta)


def sheet(table: pd.DataFrame, cache_dir: Path, out: Path, args_out: Path, n: int = 6, plane: str = "ax",
          cache_name: str = "ext") -> None:
    """Что видит модель: срез после общей предобработки, опухоль — красный контур, очаги склероза —
    зеленый. По n пациентов каждого набора в плоскости plane: срез с опухолью и без."""
    from src.dataset import PreprocCache
    cache = PreprocCache(cache_dir, cache_name, 256)
    geo = cache.table.set_index("path")
    tiles = []
    for src, g in table[table["plane"] == plane].groupby("source", sort=False):
        pts = g["patient"].drop_duplicates().sample(min(n, g["patient"].nunique()), random_state=SEED)
        for p in pts:
            r = g[g["patient"] == p]
            row = []
            for want in (True, False):
                rr = r[r["ms_lesion"] == want] if src == "ms" else r[r["has_tumor"] == want]
                if rr.empty:
                    row.append(np.zeros((256, 256, 3), np.uint8))
                    continue
                x = rr.iloc[len(rr) // 2]
                im = cv2.cvtColor(cache.image(x["path"]), cv2.COLOR_GRAY2BGR)
                cnts, _ = cv2.findContours(cache.mask(x["path"]), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(im, cnts, -1, (0, 0, 255), 1)
                if src == "ms" and x["ms_lesion"]:  # очаги склероза — зеленым
                    les = _to_model(read_image(Path(args_out) / x["lesion"]), geo.loc[x["path"]])
                    cnts, _ = cv2.findContours(les, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    cv2.drawContours(im, cnts, -1, (0, 255, 0), 1)
                label = f"{src} {p[-8:]} {'tumor' if x['has_tumor'] else 'no tumor'}"
                if src != "ms" and not x["has_tumor"]:
                    label += f" {x['dist_mm']:.0f}mm"
                if src == "ms":
                    label = f"ms {p[-9:]} {'lesion' if x['ms_lesion'] else 'no lesion'}"
                cv2.putText(im, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                row.append(im)
            tiles.append(np.hstack(row))
    pairs = [np.hstack(tiles[i:i + 3]) if len(tiles[i:i + 3]) == 3 else
             np.hstack(tiles[i:i + 3] + [np.zeros_like(tiles[0])] * (3 - len(tiles[i:i + 3])))
             for i in range(0, len(tiles), 3)]
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack(pairs))


def report(table: pd.DataFrame, split: pd.DataFrame, args, secs: float, path: Path) -> None:
    def md(df: pd.DataFrame) -> str:
        df = df.reset_index()
        return "\n".join(["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
                         + ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)])
    parts = ["train", "val", "holdout", "check"]
    pts = pd.crosstab(split["source"], split["split"]).reindex(columns=parts, fill_value=0)
    t = table.assign(kind=np.where(table["has_tumor"], "с опухолью", "без опухоли"))
    sl = pd.crosstab([t["source"], t["kind"]], t["split"]).reindex(columns=parts, fill_value=0)
    pl = pd.crosstab([t["source"], t["kind"]], t["plane"].map(PLANE_RU))
    no_tum = table[~table["has_tumor"] & (table["source"] != "ms")]
    lines = [
        "# Срезы из внешних наборов", "",
        f"Скрипт `src/build_external_slices.py`, {time.strftime('%Y-%m-%d')}; сборка {secs / 60:.0f} мин. "
        f"Параметры: k={args.k}, k_ms={args.k_ms}, ядро опухоли на срезе ≥ {args.min_tumor_mm2} мм², отступ срезов без опухоли "
        f"от любой метки: UPENN ≥ {args.gap_upenn} мм, MEN-RT ≥ {args.gap_menrt} мм; голова ≥ {args.min_head:.0%} наибольшей площади; "
        f"пиксель {args.px_mm} мм; аксиальные срезы — не ниже {args.ax_min_frac:.0%} высоты головы, корональные — "
        f"не впереди {args.cor_max_frac:.0%} ее длины от затылка (без срезов через стертое лицо); все вне головы обнулено. "
        f"Исключено срезов MEN-RT с темной обводкой (ярче ткани вокруг меньше чем в {args.min_enh_menrt} раза, вероятно ложе "
        f"удаленной опухоли): {args.n_dark} у {args.n_dark_pts} пациентов (`reports/external/menrt_dark_excluded.csv`).", "",
        "## Пациенты (`data/processed/external_split.csv`)", "", md(pts), "",
        "## Срезы по частям", "", md(sl), "",
        "## Срезы по плоскостям", "", md(pl), "",
        f"- срезов без опухоли у пациентов с опухолью: {len(no_tum)}; расстояние до ближайшей метки — медиана "
        f"{no_tum['dist_mm'].median():.0f} мм (мин. {no_tum['dist_mm'].min():.0f})",
        f"- склероз: срезов с очагом {int(table['ms_lesion'].sum())} из {int((table['source'] == 'ms').sum())}",
        f"- площадь ядра опухоли на срезах с опухолью: медиана {table.loc[table['has_tumor'], 'tumor_mm2'].median():.0f} мм²",
        "", "Классы срезов: " + ", ".join(f"{CLS_RU.get(c, c)} {n}" for c, n in table["cls"].value_counts().items()), "",
        "Листы примеров (что видит модель после общей предобработки): `reports/external/examples_{ax,cor,sag}.png`.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def build_parser() -> argparse.ArgumentParser:
    """Параметры нарезки; значения по умолчанию — те, с которыми собраны все срезы (их же берет src.build_fresh_slices)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    ap.add_argument("--out", type=Path, default=EXT / "slices")
    ap.add_argument("--k", type=int, default=4, help="срезов с опухолью и без нее на плоскость")
    ap.add_argument("--k-ms", type=int, default=10, help="срезов у склероза и срезов без опухоли у шванном (одна плоскость)")
    ap.add_argument("--min-tumor-mm2", type=float, default=100.0)
    ap.add_argument("--min-tumor-vs", type=float, default=25.0, help="шванномы маленькие: порог площади ниже")
    ap.add_argument("--gap-upenn", type=float, default=10.0)
    ap.add_argument("--gap-menrt", type=float, default=20.0)
    ap.add_argument("--min-head", type=float, default=0.55)
    ap.add_argument("--px-mm", type=float, default=0.7)
    ap.add_argument("--ax-min-frac", type=float, default=0.42, help="аксиальные срезы не ниже этой доли высоты головы")
    ap.add_argument("--cor-max-frac", type=float, default=0.6, help="корональные — не впереди этой доли длины головы")
    ap.add_argument("--min-enh-menrt", type=float, default=1.1,
                    help="срезы MEN-RT с опухолью, где обводка ярче ткани вокруг меньше чем во столько раз, исключаются")
    ap.add_argument("--limit", type=int, default=0, help="только первые N пациентов каждого набора (проверка)")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    sys.stdout.reconfigure(errors="replace")  # консоль Windows (cp1251) не печатает «≥»
    cfg = DataConfig.from_yaml(args.config)
    processed = Path(cfg.processed_dir)

    pts = patients()
    split = split_patients(pts, processed / "external_split.csv")
    part = dict(zip(zip(split["source"], split["patient"]), split["split"]))
    if args.limit:
        pts = pts.groupby("source").head(args.limit)
    t0 = time.time()
    done_dir = args.out / "_rows"  # строки по каждому пациенту: перезапуск продолжает с места остановки
    done_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    for i, pt in enumerate(pts.itertuples(index=False), 1):
        f = done_dir / f"{pt.source}__{pt.patient}.csv"
        if not f.exists():
            d = pd.DataFrame(process(pt, part[(pt.source, pt.patient)], args.out, args))
            d.to_csv(f, index=False, encoding="utf-8") if len(d) else f.write_text("", encoding="utf-8")
            gc.collect()
        parts.append(f)
        if i % 25 == 0 or i == len(pts):
            print(f"{i}/{len(pts)} пациентов, {time.time() - t0:.0f} с", flush=True)
    table = pd.concat([pd.read_csv(f, keep_default_na=False) for f in parts if f.stat().st_size > 0], ignore_index=True)
    table = table.drop_duplicates("path").reset_index(drop=True)  # строки, записанные до исправления spread
    for col in ("has_mask", "has_tumor", "ms_lesion"):
        table[col] = table[col].astype(str).str.lower().eq("true")
    # темная обводка у MEN-RT — вероятно, ложе удаленной опухоли, а не опухоль (P11-5)
    dark = ((table["source"] == "menrt") & table["has_tumor"]
            & (pd.to_numeric(table["enh_ratio"], errors="coerce") < args.min_enh_menrt))
    Path("reports/external").mkdir(parents=True, exist_ok=True)
    table[dark].to_csv("reports/external/menrt_dark_excluded.csv", index=False, encoding="utf-8")
    args.n_dark, args.n_dark_pts = int(dark.sum()), int(table.loc[dark, "patient"].nunique())
    table = table[~dark].reset_index(drop=True)
    print(f"исключено темных срезов MEN-RT: {args.n_dark} у {args.n_dark_pts} пациентов", flush=True)
    table.to_csv(processed / "external_slices.csv", index=False, encoding="utf-8")
    build_cache(table, None, cfg.cache_dir, "ext", cfg.size, roots={"ext": args.out})
    for plane in PLANES:
        sheet(table, Path(cfg.cache_dir), Path(f"reports/external/examples_{plane}.png"), args.out, plane=plane)
    report(table, pd.read_csv(processed / "external_split.csv"), args, time.time() - t0, Path("reports/external_report.md"))


if __name__ == "__main__":
    main()
