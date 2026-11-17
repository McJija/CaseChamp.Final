"""Загрузка UPENN-GBM (снимки с черепом + ручные обводки) из пофайлового зеркала на GitHub.

Источник: https://github.com/data-nih/tcia/releases/tag/upenn-gbm (неофициальное зеркало
выпуска TCIA, DOI 10.7937/TCIA.709X-DN49, CC BY 4.0). Берутся только пациенты, у которых есть
все три файла: T1 с контрастом (`ce-gd_T1w`), FLAIR и обводка (`seg`; метки 1 — некроз,
2 — отек, 3 — контрастируемая часть). Плюс таблицы `participants.tsv`, `qc.tsv`.
Каждый файл сверяется с контрольной суммой sha256 из описания выпуска.
`--extra T1w.nii.gz T2w.nii.gz` — для тех же пациентов докачать другие режимы (определитель
режима МРТ и проверка по режимам; T1 без контраста и T2 есть у всех 147, ~0,8 ГБ).

    python -m src.download_upenn_gbm --out data/external/upenn_gbm [--extra T1w.nii.gz T2w.nii.gz]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
import urllib.request
from pathlib import Path

API = "https://api.github.com/repos/data-nih/tcia/releases/tags/upenn-gbm"
NEEDED = ("ce-gd_T1w.nii.gz", "FLAIR.nii.gz", "seg.nii.gz")
EXTRA = ("participants.tsv", "qc.tsv")


def fetch_assets(tries: int = 6) -> list[dict]:
    req = urllib.request.Request(API, headers={"Accept": "application/vnd.github+json",
                                               "User-Agent": "research-download-script"})
    for k in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read().decode("utf-8"))["assets"]
        except Exception as e:  # DNS на этой машине отвечает через раз
            if k == tries - 1:
                raise
            print(f"  повтор {k + 1} для списка файлов: {e}", flush=True)
            time.sleep(3 * (k + 1))
    return []


def select(assets: list[dict], extra: tuple[str, ...] = ()) -> list[dict]:
    by_subj: dict[str, dict[str, dict]] = {}
    for a in assets:
        m = re.fullmatch(r"(sub-\d+)_(.+)", a["name"])
        if m:
            by_subj.setdefault(m.group(1), {})[m.group(2)] = a
    chosen = []
    for subj in sorted(by_subj):
        files = by_subj[subj]
        if all(k in files for k in NEEDED):
            chosen += [files[k] for k in NEEDED] + [files[k] for k in extra if k in files]
    chosen += [a for a in assets if a["name"] in EXTRA]
    return chosen


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(a: dict, dest: Path, tries: int = 6) -> str:
    want = (a.get("digest") or "").removeprefix("sha256:")
    if dest.exists() and dest.stat().st_size == a["size"] and (not want or sha256(dest) == want):
        return "уже есть"
    tmp = dest.with_suffix(dest.suffix + ".part")
    for k in range(tries):
        try:
            req = urllib.request.Request(a["browser_download_url"],
                                         headers={"User-Agent": "research-download-script"})
            with urllib.request.urlopen(req, timeout=300) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            got = sha256(tmp)
            if tmp.stat().st_size != a["size"] or (want and got != want):
                raise IOError(f"не совпала сумма или размер: {a['name']}")
            tmp.replace(dest)
            return "скачан"
        except Exception as e:  # обрывы, 5xx
            if k == tries - 1:
                raise
            print(f"  повтор {k + 1} для {a['name']}: {e}", flush=True)
            time.sleep(3 * (k + 1))
    return "?"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/external/upenn_gbm")
    ap.add_argument("--limit", type=int, default=0, help="только первые N файлов (проверка)")
    ap.add_argument("--extra", nargs="*", default=[], help="другие режимы тех же пациентов, например T1w.nii.gz T2w.nii.gz")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "nifti").mkdir(parents=True, exist_ok=True)

    chosen = select(fetch_assets(), tuple(args.extra))
    if args.limit:
        chosen = chosen[: args.limit]
    total = sum(a["size"] for a in chosen)
    n_subj = len({a["name"].split("_")[0] for a in chosen if a["name"].startswith("sub-")})
    print(f"файлов {len(chosen)}, пациентов {n_subj}, всего {total / 1e9:.3f} ГБ", flush=True)

    rows = []
    t0 = time.time()
    done = 0
    for i, a in enumerate(chosen, 1):
        dest = (out / a["name"]) if a["name"] in EXTRA else (out / "nifti" / a["name"])
        status = download(a, dest)
        done += a["size"]
        rows.append({"name": a["name"], "size": a["size"], "sha256": (a.get("digest") or "").removeprefix("sha256:"),
                     "url": a["browser_download_url"], "status": status})
        if i % 30 == 0 or i == len(chosen):
            print(f"{i}/{len(chosen)}  {done / 1e9:.2f} ГБ  {time.time() - t0:.0f} с", flush=True)

    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("готово:", out / "manifest.csv", flush=True)


if __name__ == "__main__":
    main()
