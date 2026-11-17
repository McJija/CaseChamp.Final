"""Шванномы (VS-MC-RC, VS-SEG): серии DICOM T1 → NIfTI + обводка опухоли в той же сетке.

Только для проверки системы, в обучение не идут (рама Лекселла и единственный источник). Обе
коллекции из Лондона и частично про одних и тех же пациентов (номера `VS-SEG-xxx` встречаются в
обеих), поэтому при подсчетах единица — пациент по `PatientID`, общий для двух коллекций.

Обводка опухоли:
  - VS-MC-RC — NIfTI из архива TCIA (`seg_T1` своей серии, а если ее нет — обводка того же
    исследования на T2, перенесенная по координатам аппарата);
  - VS-SEG — контуры опухоли (`TV`, `AN`, `Rt AN`, `TvT1`) из архива TCIA в координатах LPS,
    закрашиваются в сетке своей серии T1.
Режим серии: `t1c`, если в DICOM заполнено поле «контрастное вещество» (ContrastBolusAgent) или
описание серии говорит о контрасте (gad, +c, c+, gd, post, contrast); `t1`, если описание говорит
«до контраста» (pre, plain, ohne); иначе `?`.

    python -m src.convert_vs
    python -m src.convert_vs --sources vs_seg_fresh:vs_seg --out data/external/vs_nifti_fresh   # свежая проверка, круг 4
Результат: `data/external/vs_nifti/<PatientID>__<конец UID серии>_{t1,seg}.nii.gz` и `series.csv`.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import zipfile
from pathlib import Path

import cv2
import nibabel as nib
import nibabel.processing
import numpy as np
import pandas as pd
import pydicom

EXT = Path("data/external")
MCRC_ZIP = EXT / "vs_mc_rc" / "VS-MC-RC-Segmentations_with_pathname_spreadsheet-Release2023.zip"
SEG_ZIP = EXT / "vs_seg" / "Vestibular-Schwannoma-SEG-contours-Mar-2021.zip"
POST = re.compile(r"gad|\+ ?c\b|\bc ?\+|\bgd\b|post|contrast|kontrast|\bce\b|mprage.*\+", re.I)
PRE = re.compile(r"\bpre\b|plain|ohne|nativ", re.I)
LPS2RAS = np.diag([-1.0, -1.0, 1.0, 1.0])
# опухоль в контурах VS-SEG называется по-разному: TV, AN (акустическая неврома), Rt AN, TvT1;
# улитка (cochlea) и контуры повторных снимков (Vol 2y и т.п.) не нужны
TUMOR_NAME = re.compile(r"((rt|lt|left|right) )?(tv|an)(t1)?")


def read_series(sdir: Path) -> tuple[nib.Nifti1Image, pydicom.Dataset]:
    """Серия DICOM → объем (столбец, строка, срез) с матрицей координат RAS."""
    ds = [pydicom.dcmread(str(f)) for f in sorted(sdir.glob("*.dcm"))]
    ds = [d for d in ds if "PixelData" in d and "ImageOrientationPatient" in d]
    iop0 = tuple(np.round(np.array(ds[0].ImageOrientationPatient, float), 3))
    shape0 = (int(ds[0].Rows), int(ds[0].Columns))
    ds = [d for d in ds if tuple(np.round(np.array(d.ImageOrientationPatient, float), 3)) == iop0
          and (int(d.Rows), int(d.Columns)) == shape0]
    r, c = np.array(iop0[:3]), np.array(iop0[3:])
    n = np.cross(r, c)
    pos = {}
    for d in ds:  # повторы положения (несколько эхо, копии) — первый по номеру
        key = round(float(np.dot(n, np.array(d.ImagePositionPatient, float))), 2)
        if key not in pos or int(getattr(d, "InstanceNumber", 0) or 0) < int(getattr(pos[key], "InstanceNumber", 0) or 0):
            pos[key] = d
    keys = sorted(pos)
    ds = [pos[k] for k in keys]
    vol = np.stack([d.pixel_array.astype(np.float32) * float(getattr(d, "RescaleSlope", 1) or 1)
                    + float(getattr(d, "RescaleIntercept", 0) or 0) for d in ds], axis=-1)  # (строка, столбец, срез)
    vol = vol.transpose(1, 0, 2)  # (столбец, строка, срез)
    ps = [float(v) for v in ds[0].PixelSpacing]  # [между строками, между столбцами]
    dz = (keys[-1] - keys[0]) / (len(keys) - 1) if len(keys) > 1 else float(getattr(ds[0], "SliceThickness", 1) or 1)
    lps = np.eye(4)
    lps[:3, 0], lps[:3, 1], lps[:3, 2] = r * ps[1], c * ps[0], n * dz
    lps[:3, 3] = np.array(ds[0].ImagePositionPatient, float)
    return nib.Nifti1Image(vol, LPS2RAS @ lps), ds[0]


def mode_of(d0: pydicom.Dataset, coll: str) -> tuple[str, str]:
    """Режим серии и откуда он известен."""
    desc = f"{getattr(d0, 'SeriesDescription', '')} {getattr(d0, 'ProtocolName', '')}"
    if str(getattr(d0, "ContrastBolusAgent", "") or "").strip():
        return "t1c", "поле DICOM о контрасте"
    if POST.search(desc):
        return "t1c", "описание серии"
    if PRE.search(desc):
        return "t1", "описание серии"
    if coll == "vs_seg":  # Shapey et al. 2021: в наборе T1 только с контрастом (ceT1)
        return "t1c", "описание набора"
    return "?", ""


def rasterize(contours: list, img: nib.Nifti1Image) -> np.ndarray:
    """Контуры LPS (список многоугольников [[x, y, z], ...]) → маска в сетке img."""
    inv = np.linalg.inv(img.affine) @ LPS2RAS  # точка LPS → RAS → индексы снимка
    mask = np.zeros(img.shape[:3], np.uint8)
    for poly in contours:
        p = np.c_[np.array(poly, float), np.ones(len(poly))] @ inv.T
        k = int(round(float(np.median(p[:, 2]))))
        if not 0 <= k < mask.shape[2]:
            continue
        sl = np.zeros((mask.shape[1], mask.shape[0]), np.uint8)  # (строка, столбец)
        cv2.fillPoly(sl, [np.round(p[:, :2]).astype(np.int32)], 1)
        mask[:, :, k] |= sl.T
    return mask


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=EXT / "vs_nifti")
    ap.add_argument("--limit", type=int, default=0, help="только первые N серий каждой коллекции (проверка)")
    ap.add_argument("--sources", default="vs_mc_rc:vs_mc_rc,vs_seg:vs_seg",
                    help="папка в data/external:коллекция через запятую; свежая проверка (круг 4) — vs_seg_fresh:vs_seg")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    mz = zipfile.ZipFile(MCRC_ZIP)
    info = pd.read_csv(io.BytesIO(mz.read(next(n for n in mz.namelist() if n.endswith(".csv") and "__MACOSX" not in n))),
                       encoding="latin1")
    seg_names = {n.split("NIfTI-Release 2023/", 1)[1]: n for n in mz.namelist()
                 if n.endswith(".nii.gz") and "__MACOSX" not in n and "NIfTI-Release 2023/" in n}
    sz = zipfile.ZipFile(SEG_ZIP)
    contours = {}
    for n in sz.namelist():
        if n.endswith(".json"):
            for s in json.loads(sz.read(n)):
                if TUMOR_NAME.fullmatch(s["structure_name"].strip().lower()):
                    contours.setdefault(s["MR_SeriesInstanceUID"], []).extend(s["LPS_contour_points"])

    rows = []
    for sub, coll in (item.split(":") for item in args.sources.split(",")):
        sl = pd.read_csv(EXT / sub / "series.csv")
        if args.limit:
            sl = sl[sl["Modality"] == "MR"].head(args.limit)
        for s in sl[sl["Modality"] == "MR"].itertuples(index=False):
            sdir = EXT / sub / "dicom" / s.PatientID / f"MR_{s.SeriesInstanceUID}"
            row = {"collection": coll, "patient": s.PatientID, "series_uid": s.SeriesInstanceUID,
                   "description": s.SeriesDescription, "maker": s.Manufacturer, "model": s.ManufacturerModelName}
            if not sdir.exists() or not any(sdir.glob("*.dcm")):
                rows.append({**row, "status": "нет файлов"})
                continue
            img, d0 = read_series(sdir)
            seg, seg_src = None, ""
            if coll == "vs_mc_rc":
                own = info[info["series_instance_uid"] == s.SeriesInstanceUID]["SegmentationPath"].dropna()
                study = info[info["study_instance_uid"] == s.StudyInstanceUID]["SegmentationPath"].dropna()
                path = own.iloc[0] if len(own) else (study.iloc[0] if len(study) else None)
                if path and path in seg_names:
                    tmp = args.out / "_seg_tmp.nii.gz"
                    tmp.write_bytes(mz.read(seg_names[path]))
                    src = nib.load(str(tmp))
                    src = nib.Nifti1Image(np.asarray(src.dataobj), src.affine)  # отвязать от файла
                    tmp.unlink()
                    seg = (np.asarray(nibabel.processing.resample_from_to(src, img, order=0).dataobj) > 0).astype(np.uint8)
                    seg_src = "своя серия" if len(own) else "перенос с T2 того же исследования"
            elif s.SeriesInstanceUID in contours:
                seg, seg_src = rasterize(contours[s.SeriesInstanceUID], img), "контуры опухоли"
            stem = f"{s.PatientID}__{s.SeriesInstanceUID[-8:]}"
            vol = np.asarray(img.dataobj)
            if np.array_equal(vol, np.rint(vol)) and vol.min() >= -32768 and vol.max() <= 32767:  # целые: int16 вдвое компактнее
                img = nib.Nifti1Image(vol.astype(np.int16), img.affine)
            nib.save(img, str(args.out / f"{stem}_t1.nii.gz"))
            if seg is not None:
                nib.save(nib.Nifti1Image(seg, img.affine), str(args.out / f"{stem}_seg.nii.gz"))
            zooms = img.header.get_zooms()
            mode, mode_src = mode_of(d0, coll)
            rows.append({**row, "status": "ок", "stem": stem, "mode": mode, "mode_source": mode_src,
                         "contrast_agent": str(getattr(d0, "ContrastBolusAgent", "") or ""),
                         "shape": "x".join(map(str, img.shape)), "px_mm": round(float(zooms[0]), 2),
                         "step_mm": round(float(zooms[2]), 2), "extent_mm": round(float(zooms[2]) * (img.shape[2] - 1)),
                         "seg": seg_src, "tumor_voxels": int(seg.sum()) if seg is not None else -1})
            print(f"{len(rows)} {stem} {rows[-1]['mode']} {rows[-1]['shape']} {seg_src}", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(args.out / "series.csv", index=False, encoding="utf-8")
    ok = t[t["status"] == "ок"]
    print(f"серий {len(ok)}, пациентов {ok['patient'].nunique()}; режим: {ok['mode'].value_counts().to_dict()}; "
          f"обводка: {ok['seg'].replace('', 'нет').value_counts().to_dict()}")


if __name__ == "__main__":
    main()
