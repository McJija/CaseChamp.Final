"""Сборка архивов для загрузки в Kaggle.

  kaggle/code_bundle.zip — код ровно из последнего коммита (src/, configs/, requirements-kaggle.txt)
                           и файл COMMIT с его хешем. Собирать заново после каждого коммита.
  kaggle/data_bundle.zip — в одной папке (ключ --data), только части train и val: pool.csv и кэш pool_256;
                           внешние срезы UPENN-GBM и BraTS-MEN-RT (external_slices.csv, кэш ext_256);
                           FLAIR и вся опухоль UPENN (кэш upennf_256); аксиальные срезы UPENN в режимах T1, T2,
                           FLAIR (external_modes.csv, кэш extm_256, без масок). Собирать заново после изменения данных.
Исходные снимки, test пула, holdout и check внешних наборов в архивы не входят.

Пример:
    python -m src.make_kaggle_bundle            # только код
    python -m src.make_kaggle_bundle --data     # код и данные
"""
from __future__ import annotations

import argparse
import compileall
import hashlib
import json
import subprocess
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.dataset import EXT_TRAIN_SOURCES

CODE_PATHS = ["src", "configs", "requirements-kaggle.txt"]
SPLITS = ["train", "val"]  # части пула, которые уходят в Kaggle
REPO = Path(__file__).resolve().parents[1]


def git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    except FileNotFoundError:
        raise SystemExit("git не найден: запускайте из Git Bash (в PowerShell git не в PATH)") from None


def build_code(out: Path) -> str:
    dirty = git("status", "--porcelain", "--", *CODE_PATHS)
    if dirty:
        raise SystemExit("в коде есть незакоммиченные правки — сначала сделайте коммит:\n" + dirty)
    if not compileall.compile_dir(str(REPO / "src"), quiet=1, force=True):
        raise SystemExit("в src/ есть файлы с синтаксическими ошибками — архив не собран")
    commit = git("rev-parse", "HEAD")
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "archive", "--format=zip", "-o", str(out.resolve()), "HEAD", *CODE_PATHS], cwd=REPO, check=True)
    with zipfile.ZipFile(out, "a", zipfile.ZIP_DEFLATED) as z:
        z.writestr("COMMIT", commit + "\n")
    return commit


# кэш → (его таблица в processed_dir или None, какие строки уходят в Kaggle). Только части train и val (правило 5):
# test пула, holdout и check внешних наборов в Kaggle не отправляются.
PARTS = {
    "pool": ("pool.csv", lambda t: t["split"].isin(SPLITS)),
    "ext": ("external_slices.csv", lambda t: t["split"].isin(SPLITS) & t["source"].isin(EXT_TRAIN_SOURCES)),
    "upennf": (None, lambda t: t["split"].isin(SPLITS)),  # FLAIR и вся опухоль UPENN (src.build_core_cache)
    "extm": ("external_modes.csv", lambda t: t["split"].isin(SPLITS) & (t["source"] == "upenn") & (t["plane"] == "ax")),
}
NO_MASKS = {"extm"}  # определителю режима маски не нужны


def build_data(out: Path, processed: Path, cache_dir: Path, size: int) -> dict:
    info = {"created": datetime.now().strftime("%Y-%m-%d %H:%M"), "test_included": False, "parts": {}}
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=3) as z:
        for name, (table_name, rule) in PARTS.items():
            files = {s: cache_dir / f"{name}_{size}{s}" for s in (".npy", "_mask.npy", ".csv", ".json")}
            missing = [str(p) for p in files.values() if not p.exists()]
            if missing:
                raise SystemExit(f"нет файлов кэша: {missing}")
            cache = pd.read_csv(files[".csv"], keep_default_na=False, dtype=str)
            keep = np.flatnonzero(rule(cache).to_numpy())
            if table_name:
                table = pd.read_csv(processed / table_name, keep_default_na=False, dtype=str)
                if set(table["path"]) != set(cache["path"]) or len(table) != len(cache):
                    raise SystemExit(f"кэш {name} не соответствует {table_name} — пересоберите кэш")
                sub = table[rule(table)]
                if set(sub["path"]) != set(cache.iloc[keep]["path"]):
                    raise SystemExit(f"отбор строк {table_name} и кэша {name} разошелся")
                z.writestr(table_name, sub.to_csv(index=False, lineterminator="\n"))
                info["parts"][name] = {"table": table_name, "table_md5": hashlib.md5((processed / table_name).read_bytes()).hexdigest()}
            cache_info = {**json.loads(files[".json"].read_text(encoding="utf-8")), "n": len(keep)}
            if name in NO_MASKS:
                cache_info["has_masks"] = False
            z.writestr(f"{name}_{size}.csv", cache.iloc[keep].to_csv(index=False, lineterminator="\n"))
            z.writestr(f"{name}_{size}.json", json.dumps(cache_info))
            for suffix in (".npy",) if name in NO_MASKS else (".npy", "_mask.npy"):
                arr = np.load(files[suffix], mmap_mode="r")[keep]
                with z.open(f"{name}_{size}{suffix}", "w", force_zip64=True) as f:
                    np.save(f, arr)
                del arr
            info["parts"].setdefault(name, {}).update({"rows": len(keep), "split": cache.iloc[keep]["split"].value_counts().to_dict()})
        z.writestr("data_info.json", json.dumps(info, ensure_ascii=False, indent=1))
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", action="store_true", help="собрать и архив данных")
    ap.add_argument("--out-dir", type=Path, default=Path("kaggle"))
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--cache-dir", type=Path, default=Path("data/processed/cache"))
    ap.add_argument("--size", type=int, default=256)
    args = ap.parse_args()

    code = args.out_dir / "code_bundle.zip"
    commit = build_code(code)
    with zipfile.ZipFile(code) as z:
        n = sum(not name.endswith("/") for name in z.namelist())
    print(f"код: {code} — {n} файлов, {code.stat().st_size / 1024:.0f} КБ, коммит {commit}")
    if args.data:
        data = args.out_dir / "data_bundle.zip"
        info = build_data(data, args.processed, args.cache_dir, args.size)
        print(f"данные: {data} — {data.stat().st_size / 2**20:.0f} МБ (test, holdout и check не входят)")
        for name, p in info["parts"].items():
            print(f"  {name}: {p['rows']} строк, части {p['split']}")


if __name__ == "__main__":
    main()
