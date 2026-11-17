"""Загрузка снимков T1 с контрастом пациентов с рассеянным склерозом (Любляна).

Lesjak Ž. et al. A novel public MR image dataset of multiple sclerosis patients with lesion
segmentations based on multi-rater consensus. Neuroinformatics, 2018. Лицензия CC BY.
Зеркало: https://github.com/muschellij2/open_ms_data (часть `cross_sectional/raw`, 30 пациентов,
Siemens Trio 3 Т). Берется `T1WKS.nii.gz` (T1 после контраста, голова целиком) и
`consensus_gt.nii.gz` (обводка очагов склероза — для справки). Опухолей у пациентов нет:
набор нужен только для проверки системы на снимках T1 с контрастом без опухоли.
`--extra T1W.nii.gz T2W.nii.gz FLAIR.nii.gz` — другие режимы тех же пациентов (определитель
режима МРТ и проверка по режимам; ~1,5 ГБ).

`--xfm` — файлы совмещения авторов (`T1WKS_to_FLAIR.txt` и др., ITK): в исходных файлах FLAIR и
обводка очагов (она в сетке FLAIR) не совпадают с T1 по координатам аппарата (поворот ~11°, сдвиг ~2 см).

    python -m src.download_ms_ljubljana --out data/external/ms_ljubljana [--extra T1W.nii.gz T2W.nii.gz FLAIR.nii.gz] [--xfm]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
import urllib.request
from pathlib import Path

REPO = "muschellij2/open_ms_data"
TREE = f"https://api.github.com/repos/{REPO}/git/trees/master?recursive=1"
RAW = f"https://raw.githubusercontent.com/{REPO}/master/"
FILES = ("T1WKS.nii.gz", "consensus_gt.nii.gz")
UA = {"User-Agent": "research-download-script"}


def get(url: str, tries: int = 6) -> bytes:
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300) as r:
                return r.read()
        except Exception as e:  # DNS на этой машине отвечает через раз
            if k == tries - 1:
                raise
            print(f"  повтор {k + 1}: {url.rsplit('/', 3)[-3:]}: {e}", flush=True)
            time.sleep(3 * (k + 1))
    return b""


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/external/ms_ljubljana")
    ap.add_argument("--extra", nargs="*", default=[], help="другие режимы, например T1W.nii.gz T2W.nii.gz FLAIR.nii.gz")
    ap.add_argument("--xfm", action="store_true",
                    help="файлы совмещения авторов `coregistered/*_to_FLAIR.txt` (ITK): FLAIR и обводка очагов "
                         "не совпадают с T1 по координатам аппарата, переносить их можно только через эти файлы")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = FILES + tuple(args.extra)

    tree = json.loads(get(TREE).decode("utf-8"))["tree"]
    want = [t for t in tree if t["type"] == "blob" and (
        (t["path"].startswith("cross_sectional/raw/") and t["path"].rsplit("/", 1)[-1] in files)
        or (args.xfm and t["path"].startswith("cross_sectional/coregistered/patient") and t["path"].endswith("_to_FLAIR.txt")))]
    print(f"файлов {len(want)}, {sum(t['size'] for t in want) / 1e9:.3f} ГБ", flush=True)
    rows = []
    for i, t in enumerate(sorted(want, key=lambda x: x["path"]), 1):
        patient, name = t["path"].split("/")[-2:]
        dest = out / patient / name
        dest.parent.mkdir(exist_ok=True)
        if dest.exists() and git_blob_sha(dest.read_bytes()) == t["sha"]:
            status = "уже есть"
        else:
            data = get(RAW + t["path"])
            if git_blob_sha(data) != t["sha"]:
                raise IOError(f"не совпала контрольная сумма: {t['path']}")
            dest.write_bytes(data)
            status = "скачан"
        rows.append({"patient": patient, "file": name, "size": t["size"], "git_sha1": t["sha"], "status": status})
        if i % 10 == 0 or i == len(want):
            print(f"{i}/{len(want)}", flush=True)
    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("готово:", out / "manifest.csv", flush=True)


if __name__ == "__main__":
    main()
