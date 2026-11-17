"""Система финала (`run3`): три копии моделей типа и контура, ядро и отек, пометка «проверьте», измерения.

    from src.system import load_system
    sp = load_system()                         # configs/system_run3.yaml
    r = sp.predict(image, mode="t1c")          # image: HxW или HxWx3, uint8 или 16 бит; mode — режим, выбранный врачом, или None

Ответ считается ровно так же, как при замерах (src/eval_external.predict, src/confidence.signals):
  - копия 0 (seed 42) — основная: «опухоль есть», если модель контура нашла область > 50 пикселей (в кадре 256×256)
    ИЛИ вероятность опухоли у модели типа ≥ порога (configs/presence_run3.yaml); тип — самый вероятный из трех;
  - копии 1 и 2 (seed 43, 44) — для пометки «проверьте» (configs/confidence_run3.yaml, вариант E0.6) и зоны
    неуверенной границы: пиксели, которые выделили одна или две копии из трех (все три — уверенная часть);
  - ядро и отек — модель `run3_core` seed 42 (вход T1 с контрастом), только если найдена опухоль типа «глиома»;
    обучена на глиобластомах одного набора (UPENN-GBM) — результат ориентировочный;
  - тепловая карта Grad-CAM основной модели типа — по желанию (показывает, на что смотрела модель типа, а не где опухоль).
Маски возвращаются в координатах исходного снимка. Измерения — в пикселях исходного снимка: у JPG и PNG нет сведений
о масштабе, поэтому миллиметры не указываются.

Проверка, что движок повторяет записанные замеры (наш val и внешний val, по кэшу препроцессинга):
    python -m src.system --check
То же из исходных файлов через predict(), как в прототипе:
    python -m src.system --check-raw
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml

from src.confidence import signals as conf_signals
from src.dataset import to_tensor
from src.eval_core import load_model as load_core
from src.eval_external import find_run, md5
from src.inference import MODELS, ROOT, _load, cam_layer
from src.labels import CLASSES, CLS_RU, NO_TUMOR_IDX
from src.preprocess import map_to_original, preprocess

SYSTEM = ROOT / "configs" / "system_run3.yaml"
TUMOR_TYPES = CLASSES[:NO_TUMOR_IDX]
MODE_RU = {"t1c": "T1 с контрастом", "t1": "T1 без контраста", "t2": "T2", "flair": "FLAIR", None: "не указан"}
SIGNAL_RU = {
    "presence": "копии системы расходятся в ответе «опухоль есть / нет»",
    "type": "копии системы называют разный тип опухоли",
    "models": "модель типа и модель контура отвечают по-разному",
    "type_prob": "тип опухоли неуверенный: доля самого вероятного типа меньше {p:.0%}",
}
H_POS = ("левая", "средняя", "правая")   # трети ширины головы на изображении
V_POS = ("верхняя", "средняя", "нижняя")  # трети высоты


def _head(x256: np.ndarray) -> np.ndarray:
    """Голова на кадре модели: после препроцессинга фон = 0; самая крупная связная область без дыр."""
    m = (x256 > 8).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return m.astype(bool)
    big = (lab == 1 + int(stats[1:, cv2.CC_STAT_AREA].argmax())).astype(np.uint8)
    cnt, _ = cv2.findContours(big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(big)
    cv2.drawContours(out, cnt, -1, 1, -1)
    return out.astype(bool)


def max_diameter(mask: np.ndarray) -> float:
    """Наибольшее расстояние между двумя точками области (по выпуклой оболочке), в пикселях."""
    pts = cv2.findNonZero(mask.astype(np.uint8))
    if pts is None:
        return 0.0
    hull = cv2.convexHull(pts).reshape(-1, 2).astype(np.float64)
    if len(hull) < 2:
        return 1.0
    d = np.sqrt(((hull[:, None, :] - hull[None, :, :]) ** 2).sum(-1))
    return float(d.max()) + 1.0  # +1: размер в пикселях включает оба крайних пикселя


def position(mask: np.ndarray, head: np.ndarray) -> dict:
    """Положение центра области в рамке головы на изображении (не анатомическая сторона)."""
    ys, xs = np.nonzero(mask)
    hy, hx = np.nonzero(head)
    if not len(xs) or not len(hx):
        return {}
    fx = (xs.mean() - hx.min()) / max(1, hx.max() - hx.min())
    fy = (ys.mean() - hy.min()) / max(1, hy.max() - hy.min())
    return {"h": H_POS[min(2, int(np.clip(fx, 0, 0.999) * 3))], "v": V_POS[min(2, int(np.clip(fy, 0, 0.999) * 3))],
            "fx": float(fx), "fy": float(fy)}


class SystemPredictor:
    def __init__(self, system_path: Path = SYSTEM, device: str = "cpu", core: bool = True, cam: bool = True):
        self.device = torch.device(device)
        cfg = yaml.safe_load(Path(system_path).read_text(encoding="utf-8"))
        res = lambda p: Path(p) if Path(p).is_absolute() else ROOT / p  # noqa: E731
        pres = yaml.safe_load(res(cfg["presence"]).read_text(encoding="utf-8"))
        self.thr, self.min_area = float(pres["cls_threshold"]), int(pres["min_area_px"])
        self.rule = yaml.safe_load(res(cfg["confidence"]).read_text(encoding="utf-8"))["rule"]
        self.copies, files = [], {}
        for c in cfg["copies"]:
            pc, ps = find_run(c["classifier"]), find_run(c["segmenter"])
            self.copies.append((_load(pc, self.device)[0], _load(ps, self.device)[0]))
            files[pc.parent.name], files[ps.parent.name] = md5(pc), md5(ps)
        self.core = None
        if core and cfg.get("core_edema"):
            run = find_run(cfg["core_edema"]["t1c"][0]).parent
            self.core, flair = load_core(run, self.device)
            assert not flair, "ядро и отек — модель по одному T1 с контрастом"
            files[run.name] = md5(run / "best.pt")
        self.cam = None
        if cam:
            from pytorch_grad_cam import GradCAM
            cls0, ck = _load(find_run(cfg["copies"][0]["classifier"]), self.device)
            self.cam = GradCAM(model=cls0, target_layers=[cam_layer(cls0, ck["encoder"])])
        self.info = {"name": cfg["name"], "system": str(Path(system_path).name), "threshold": self.thr, "min_area_px": self.min_area,
                     "check_rule": self.rule["name"], "files_md5": files}

    @torch.no_grad()
    def forward(self, xb: torch.Tensor) -> tuple[list[dict], np.ndarray]:
        """Батч кадров модели (N×3×256×256) → строки в формате src/eval_external.predict и маски копий (N×3×256×256)."""
        n, rows, masks = len(xb), [{} for _ in range(len(xb))], np.zeros((len(xb), len(self.copies), *xb.shape[-2:]), bool)
        for c, (cls, seg) in enumerate(self.copies):
            probs = torch.softmax(cls(xb).float(), 1).cpu().numpy()
            m = (torch.sigmoid(seg(xb).float())[:, 0] > 0.5).cpu().numpy()
            area = m.reshape(n, -1).sum(1)
            m[area <= self.min_area] = False  # площадь ≤ порога — область не найдена
            masks[:, c] = m
            for j in range(n):
                pr, found = probs[j], bool(area[j] > self.min_area)
                p_t = float(1 - pr[NO_TUMOR_IDX])
                present = found or p_t >= self.thr
                typ = TUMOR_TYPES[int(pr[:NO_TUMOR_IDX].argmax())]
                rows[j].update({f"p_tumor_{c}": p_t, f"found_{c}": found, f"area_{c}": int(area[j]) if found else 0,
                                f"present_{c}": present, f"type_{c}": typ, f"answer_{c}": typ if present else "no_tumor",
                                f"cls_argmax_{c}": CLASSES[int(pr.argmax())]})
                if c == 0:
                    rows[j].update({f"p_{k}": float(pr[ki]) for ki, k in enumerate(CLASSES)})
        return rows, masks

    def check(self, rows: list[dict]) -> pd.DataFrame:
        """Признаки пометки «проверьте» (src/confidence.signals) и итог."""
        sig = conf_signals(pd.DataFrame(rows), self.rule).astype(bool)
        sig["check"] = sig.any(axis=1)
        return sig

    @torch.no_grad()
    def predict(self, image: np.ndarray, mode: str | None = None, heatmap: bool = True) -> dict:
        t0 = time.perf_counter()
        x256, meta = preprocess(image)
        x = to_tensor(x256).unsqueeze(0).to(self.device)
        rows, masks = self.forward(x)
        row, m = rows[0], masks[0]
        sig = self.check(rows).iloc[0]
        present, typ = bool(row["present_0"]), row["type_0"]
        p3 = np.array([row[f"p_{t}"] for t in TUMOR_TYPES])
        share = float(p3.max() / max(p3.sum(), 1e-9))
        reasons = [SIGNAL_RU[s].format(p=self.rule.get("type_prob_min", 0)) for s in self.rule["signals"] if bool(sig[s])]
        if bool(sig.get("models", False)):  # уточнение: какая модель что увидела
            said = ("модель типа подозревает опухоль, а модель контура область не выделила" if not row["found_0"]
                    else "модель контура выделила область, а модель типа опухоли не видит")
            reasons = [said if x == SIGNAL_RU["models"] else x for x in reasons]

        to_orig = lambda a: map_to_original(a.astype(np.uint8), meta, binary=True).astype(bool)  # noqa: E731
        votes = m.sum(0)
        head256 = _head(x256)
        main256 = m[0] if present else np.zeros_like(m[0])
        r = {"present": present, "type": typ if present else None, "type_ru": CLS_RU[typ] if present else None,
             "type_share": share, "p_tumor": float(row["p_tumor_0"]), "probs": {k: float(row[f"p_{k}"]) for k in CLASSES},
             "found": bool(row["found_0"]), "check": bool(sig["check"]), "check_reasons": reasons,
             "signals": {s: bool(sig[s]) for s in self.rule["signals"]},
             "copies": [{"present": bool(row[f"present_{c}"]), "type": row[f"type_{c}"] if row[f"present_{c}"] else None,
                         "found": bool(row[f"found_{c}"]), "p_tumor": float(row[f"p_tumor_{c}"])} for c in range(len(self.copies))],
             "mask": to_orig(main256), "uncertain": to_orig((votes > 0) & (votes < len(self.copies)) & present),
             "head": to_orig(head256), "core": None, "edema": None, "core_applied": False,
             "mode": mode, "mode_ru": MODE_RU.get(mode, "не указан"), "mode_warning": mode != "t1c",
             "image_size": (int(meta.orig_w), int(meta.orig_h)), "system": self.info, "heatmap": None, "heatmap_class": None}

        if present and typ == "glioma" and self.core is not None:
            out = (torch.sigmoid(self.core(x).float())[0] > 0.5).cpu().numpy()
            core, whole = out[0], out[1] | out[0]
            if whole.sum() > self.min_area:
                r.update({"core": to_orig(core), "edema": to_orig(whole & ~core), "core_applied": True})
            else:
                r["core_applied"] = True  # модель применена, но области не нашла

        r["measure"] = measure(r["mask"], r["head"], r["core"], r["edema"], main256, head256)
        if heatmap and self.cam is not None:
            from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
            target = CLASSES.index(typ)
            with torch.enable_grad():
                cam = self.cam(input_tensor=x, targets=[ClassifierOutputTarget(target)])[0]
            r["heatmap"], r["heatmap_class"] = np.clip(map_to_original(cam, meta, binary=False), 0, 1), typ
        r["time_s"] = time.perf_counter() - t0
        return r


def measure(mask: np.ndarray, head: np.ndarray, core: np.ndarray | None, edema: np.ndarray | None,
            mask256: np.ndarray | None = None, head256: np.ndarray | None = None) -> dict:
    """Измерения области в пикселях исходного снимка. Доля головы — по кадру модели, если он дан (точнее), иначе по исходному."""
    area = int(mask.sum())
    if not area:
        return {"area_px": 0, "n_regions": 0}
    n_reg = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)[0] - 1
    frac = (float(mask256.sum() / max(1, head256.sum())) if mask256 is not None and head256 is not None
            else float(area / max(1, head.sum())))
    n_all, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    largest = lab == 1 + int(stats[1:, cv2.CC_STAT_AREA].argmax())  # положение — по самому крупному участку
    out = {"area_px": area, "head_fraction": frac, "max_diam_px": max_diameter(mask), "n_regions": int(n_reg),
           "position": position(largest, head)}
    if core is not None:
        out.update({"core_px": int(core.sum()), "edema_px": int(edema.sum())})
    return out


_default: SystemPredictor | None = None


def load_system(**kw) -> SystemPredictor:
    return SystemPredictor(**kw)


def run_check(batch: int = 16) -> dict:
    """Движок на кэше препроцессинга против записанных предсказаний системы на val (наш и внешний val)."""
    from src.confidence import signals as sigf
    from src.dataset import DataConfig, PreprocCache
    from src.eval_external import tables, typed

    cfg = DataConfig.from_yaml("configs/base.yaml")
    sp = SystemPredictor(core=False, cam=False)
    rec = typed(pd.read_csv(ROOT / "reports/eval_external/run3_val/predictions.csv", keep_default_na=False, low_memory=False))
    out = {}
    for table, cache in tables(cfg, ["val", "ext"], splits=("val",)):
        name = table["set"].iloc[0]
        rows = []
        for i in range(0, len(table), batch):
            paths = table["path"].iloc[i:i + batch].tolist()
            rows += sp.forward(torch.stack([to_tensor(cache.image(p)) for p in paths]))[0]
        got = pd.DataFrame(rows).assign(path=table["path"].to_numpy())
        got["check"] = sp.check(rows)["check"].to_numpy()
        ref = rec[rec["set"] == name].set_index("path").loc[got["path"]]
        ref_check = sigf(ref.reset_index(), sp.rule).any(axis=1).to_numpy()
        same = {f"{k}_{c}": int((got[f"{k}_{c}"].to_numpy() == ref[f"{k}_{c}"].to_numpy()).sum()) for k in ("present", "type", "found") for c in range(3)}
        out[name] = {"n": len(got), "same_present_0": same["present_0"], "same_type_0": same["type_0"], "same_found_0": same["found_0"],
                     "same_all_copies": int(min(same.values())), "same_check": int((got["check"].to_numpy() == ref_check).sum()),
                     "max_p_tumor_diff": float(np.abs(got["p_tumor_0"].to_numpy() - ref["p_tumor_0"].to_numpy()).max())}
    return out


def run_core_check() -> dict:
    """Модель ядра и отека на глиомах нашего val (срезы Figshare с обводкой). Обводка Figshare — одна область опухоли
    на T1 с контрастом, без деления на ядро и отек, поэтому сравнивается с ней и ядро, и вся опухоль модели."""
    from src.dataset import DataConfig, PreprocCache
    from src.eval_external import dice

    cfg = DataConfig.from_yaml("configs/base.yaml")
    sp = SystemPredictor(cam=False)
    pool = pd.read_csv(ROOT / "data/processed/pool.csv", keep_default_na=False)
    v = pool[(pool["split"] == "val") & (pool["cls"] == "glioma") & pool["has_mask"].astype(str).str.lower().eq("true")]
    cache = PreprocCache(cfg.cache_dir, "pool", cfg.size)
    rows = []
    with torch.no_grad():
        for p, g in zip(v["path"], v["group"]):
            x = to_tensor(cache.image(p)).unsqueeze(0)
            out = (torch.sigmoid(sp.core(x).float())[0] > 0.5).numpy()
            core, whole, gt = out[0], out[1] | out[0], cache.mask(p) > 0
            rows.append({"group": g, "dice_core": dice(core, gt), "dice_whole": dice(whole, gt),
                         "found": bool(whole.sum() > sp.min_area), "whole_over_gt": float(whole.sum() / max(1, gt.sum()))})
    d = pd.DataFrame(rows)
    pp = d.groupby("group")[["dice_core", "dice_whole"]].mean()
    return {"slices": len(d), "patients": int(d["group"].nunique()), "found_share": float(d["found"].mean()),
            "dice_core_by_patient": float(pp["dice_core"].mean()), "dice_whole_by_patient": float(pp["dice_whole"].mean()),
            "dice_core_slices": float(d["dice_core"].mean()), "dice_whole_slices": float(d["dice_whole"].mean()),
            "whole_area_over_gt_median": float(d["whole_over_gt"].median())}


def run_raw_check() -> dict:
    """Путь прототипа на исходных файлах: байты файла → декодирование в сером, как в app/app.py → predict(...) целиком
    (препроцессинг, три копии, «проверьте») против записанных предсказаний системы на val. Наш val — JPG/PNG кейса и
    PNG Figshare, внешний val — PNG срезов UPENN-GBM и BraTS-MEN-RT. В отличие от --check, кэш препроцессинга не используется."""
    from src.dataset import DataConfig
    from src.eval_external import tables, typed

    cfg = DataConfig.from_yaml("configs/base.yaml")
    data_root = Path(cfg.data_root) if Path(cfg.data_root).is_absolute() else ROOT / cfg.data_root
    roots = {"case": data_root, "figshare": ROOT / "data/external/figshare/png", "ext": ROOT / "data/external/slices"}
    sp = SystemPredictor(core=False, cam=False)
    rec = typed(pd.read_csv(ROOT / "reports/eval_external/run3_val/predictions.csv", keep_default_na=False, low_memory=False))
    out = {}
    for table, _ in tables(cfg, ["val", "ext"], splits=("val",)):
        name = table["set"].iloc[0]
        ref = rec[rec["set"] == name].reset_index(drop=True)
        ref_check = pd.Series(conf_signals(ref, sp.rule).astype(bool).any(axis=1).to_numpy(), index=ref["path"])
        ref = ref.set_index("path")
        diff = {"present": 0, "type": 0, "found": 0, "check": 0}
        max_dp = 0.0
        for row in table.itertuples():
            data = (roots[row.root] / row.path).read_bytes()
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)  # как decode() в app/app.py
            r = sp.predict(img, heatmap=False)
            f = ref.loc[row.path]
            for c, cp in enumerate(r["copies"]):
                diff["present"] += cp["present"] != bool(f[f"present_{c}"])
                diff["found"] += cp["found"] != bool(f[f"found_{c}"])
                diff["type"] += cp["present"] and cp["type"] != f[f"type_{c}"]
                max_dp = max(max_dp, abs(cp["p_tumor"] - float(f[f"p_tumor_{c}"])))
            diff["check"] += r["check"] != bool(ref_check[row.path])
        out[name] = {"n": len(table), "copies": len(sp.copies), "mismatch": {k: int(v) for k, v in diff.items()},
                     "max_p_tumor_diff": max_dp}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="сверить движок с записанными предсказаниями на val")
    ap.add_argument("--check-raw", action="store_true", help="то же, но из исходных файлов через predict(), как в прототипе")
    ap.add_argument("--check-core", action="store_true", help="модель ядра и отека на глиомах нашего val")
    ap.add_argument("image", nargs="?", type=Path)
    ap.add_argument("--mode", default=None, choices=[None, "t1c", "t1", "t2", "flair"])
    args = ap.parse_args()
    if args.check:
        print(json.dumps(run_check(), ensure_ascii=False, indent=1))
        return
    if args.check_core:
        print(json.dumps(run_core_check(), ensure_ascii=False, indent=1))
        return
    if args.check_raw:
        print(json.dumps(run_raw_check(), ensure_ascii=False, indent=1))
        return
    from src.io_utils import read_image
    r = load_system().predict(read_image(args.image), mode=args.mode)
    keep = {k: v for k, v in r.items() if not isinstance(v, np.ndarray) and k not in ("system",)}
    print(json.dumps(keep, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
