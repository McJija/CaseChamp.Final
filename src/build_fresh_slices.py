"""Свежая проверка круга 4 (П1ф): срезы 152 пациентов VS-SEG, которых нет ни среди 40 скачанных раньше, ни в VS-MC-RC.

Правило записано до обучения `run4` (2026-11-14): срезы T1 с контрастом без опухоли и со шванномой — тем же
кодом и с теми же параметрами, что прежние срезы шванном; отдельная таблица и кэш. Поэтому здесь нет своей нарезки:
берется функция `process` и значения по умолчанию из `src.build_external_slices` (одна плоскость, в которой снимал
аппарат; до 4 срезов с опухолью площадью ядра ≥ 25 мм² и до 10 срезов без опухоли на расстоянии ≥ 10 мм от обводки;
пиксель 0,7 мм; все вне головы обнулено). Все пациенты — часть `check` (только проверка, ничего не подбирается).

Порядок:
    python -m src.download_idc --preset vs_seg_fresh --out data/external/vs_seg_fresh
    python -m src.convert_vs --sources vs_seg_fresh:vs_seg --out data/external/vs_nifti_fresh
    python -m src.build_fresh_slices
Результат: срезы `data/external/slices_fresh/vs/`, таблица `data/processed/external_fresh.csv`, кэш `extf_256`,
лист `reports/external/examples_fresh.png`, отчет `reports/external_fresh_report.md`.
Замер: `python -m src.eval_external --tag <...> --sets val,ext --splits check --ext-table external_fresh.csv --ext-cache extf`.
"""
from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import pandas as pd

from src.build_external_slices import EXT, build_parser, process, sheet
from src.dataset import DataConfig, build_cache

NIFTI = EXT / "vs_nifti_fresh"
OUT = EXT / "slices_fresh"
TABLE = "external_fresh.csv"
CACHE = "extf"
REPORT = Path("reports/external_fresh_report.md")


def fresh_patients() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Пациенты (серии с непустой обводкой — так же, как шванномы в `patients()` прежней сборки) и таблица всех серий."""
    t = pd.read_csv(NIFTI / "series.csv", keep_default_na=False)
    ok = t[(t["status"] == "ок") & (t["seg"] != "") & (t["tumor_voxels"].astype(int) > 0)]
    return pd.DataFrame([{"source": "vs", "patient": r.stem, "cls": "schwannoma", "t1c": f"vs_nifti_fresh/{r.stem}_t1.nii.gz",
                          "flair": "", "lab": f"vs_nifti_fresh/{r.stem}_seg.nii.gz", "group": f"vs:{r.patient}", "mode": r.mode}
                         for r in ok.itertuples(index=False)]), t


def main() -> None:
    sys.stdout.reconfigure(errors="replace")
    ap = build_parser()
    ap.set_defaults(out=OUT)
    args = ap.parse_args()
    cfg = DataConfig.from_yaml(args.config)
    processed = Path(cfg.processed_dir)
    old = set(pd.read_csv(EXT / "vs_nifti" / "series.csv", keep_default_na=False)["patient"])
    pts, series = fresh_patients()
    clash = set(series["patient"]) & old
    if clash:
        raise ValueError(f"пациенты уже были в прежней проверке: {sorted(clash)[:5]}")
    if args.limit:
        pts = pts.head(args.limit)
    t0 = time.time()
    done = args.out / "_rows"
    done.mkdir(parents=True, exist_ok=True)
    files = []
    for i, pt in enumerate(pts.itertuples(index=False), 1):
        f = done / f"{pt.source}__{pt.patient}.csv"
        if not f.exists():
            d = pd.DataFrame(process(pt, "check", args.out, args))
            d.to_csv(f, index=False, encoding="utf-8") if len(d) else f.write_text("", encoding="utf-8")
            gc.collect()
        files.append(f)
        if i % 25 == 0 or i == len(pts):
            print(f"{i}/{len(pts)} пациентов, {time.time() - t0:.0f} с", flush=True)
    table = pd.concat([pd.read_csv(f, keep_default_na=False) for f in files if f.stat().st_size > 0], ignore_index=True)
    for col in ("has_mask", "has_tumor", "ms_lesion"):
        table[col] = table[col].astype(str).str.lower().eq("true")
    if args.limit:
        print(table.groupby(["has_tumor", "plane", "mode"]).size())
        return
    table.to_csv(processed / TABLE, index=False, encoding="utf-8")
    build_cache(table, None, cfg.cache_dir, CACHE, cfg.size, roots={"ext": args.out})
    plane = table["plane"].value_counts().index[0]
    sheet(table, Path(cfg.cache_dir), Path("reports/external/examples_fresh.png"), args.out, n=12, plane=plane, cache_name=CACHE)

    ok = series[series["status"] == "ок"]
    nt, tu = table[~table["has_tumor"]], table[table["has_tumor"]]
    L = ["# Свежая проверка круга 4 (П1ф): срезы новых пациентов VS-SEG", "",
         f"`python -m src.build_fresh_slices`, {time.strftime('%Y-%m-%d')}; сборка {(time.time() - t0) / 60:.0f} мин. Тот же код "
         f"(`src.build_external_slices.process`) и те же параметры, что у прежних срезов шванном: k={args.k}, k_ms={args.k_ms}, "
         f"ядро опухоли ≥ {args.min_tumor_vs} мм², отступ срезов без опухоли ≥ {args.gap_upenn} мм, голова ≥ {args.min_head:.0%} "
         f"наибольшей площади, пиксель {args.px_mm} мм.", "",
         f"- серий скачано {len(series)}, переведено {len(ok)}, с непустой обводкой {len(pts)}; пациентов {pts['group'].nunique()}; "
         f"общих с прежней проверкой — 0 (проверено по номеру пациента)",
         f"- режим серий: {ok['mode'].value_counts().to_dict()}; откуда известен: {ok['mode_source'].value_counts().to_dict()}",
         f"- аппараты: {ok.groupby(['maker', 'model']).size().sort_values(ascending=False).to_dict()}",
         f"- плоскость срезов: {table['plane'].value_counts().to_dict()}",
         f"- срезов без опухоли {len(nt)} (T1 с контрастом {int((nt['mode'] == 't1c').sum())}; пациентов "
         f"{nt['group'].nunique()}), расстояние до обводки — медиана {nt['dist_mm'].median():.0f} мм (мин. {nt['dist_mm'].min():.0f})",
         f"- срезов со шванномой {len(tu)} (пациентов {tu['group'].nunique()}), площадь ядра — медиана {tu['tumor_mm2'].median():.0f} мм²",
         "", "Лист примеров (что видит модель): `reports/external/examples_fresh.png`. Проверка открывается один раз (правило записано до обучения).", ""]
    REPORT.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
