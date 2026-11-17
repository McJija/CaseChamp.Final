"""Обучение вспомогательных моделей этапа 3 подготовки к финалу.

kind:
  core — ядро и отек при глиобластоме (`smp.Unet`, два выхода с сигмоидой: ядро = метки 1 + 3 UPENN-GBM,
         вся опухоль = 1 + 2 + 3; отек = вся опухоль минус ядро). Данные — срезы UPENN-GBM всех плоскостей,
         с опухолью и без (пустая маска). Вход (`model.inputs`): `t1c` — T1 с контрастом (как у основной системы),
         `t1c_flair` — каналы T1c, FLAIR, T1c. T1 с контрастом и ядро — кэш `ext_256`, FLAIR и вся опухоль —
         кэш `upennf_256` (python -m src.build_core_cache).
         Эпоха — наибольшее среднее Dice ядра и Dice отека на срезах UPENN val, где они есть в обводке.
  mode — определитель режима МРТ (классификатор timm, 4 класса: t1c, t1, t2, flair). Данные — аксиальные срезы
         UPENN-GBM, один и тот же срез во всех четырех режимах (P11-7: корональные и сагиттальные T2/FLAIR размыты).
         T1c — кэш `ext_256`, остальные — `extm_256` (python -m src.build_mode_slices).
         Эпоха — наибольший macro-F1 по 4 режимам на UPENN val.
Деление по пациентам — `external_split.csv` (часть в колонке split); holdout и check сюда не попадают.
Двухфазное дообучение, оптимизатор, расписание, смешанная точность — как в src/train.py.

    python -m src.train_aux --config configs/run3_core.yaml [--seed 43]
    python -m src.train_aux --config configs/run3_mode.yaml --limit-train 64 --limit-val 48 --no-pretrained --device cpu
Результат — папка <out_dir>/<run>_s<seed>_<время>/ (best.pt, history.csv, val_predictions.csv, config.yaml)
и строка в reports/experiments.csv.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import fields, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import timm
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset

from src.config import load_config
from src.dataset import (_MEAN, _STD, CACHE_VERSION, DataConfig, MultiCache, PreprocCache, _worker_init, build_transforms,
                         seed_everything)
from src.metrics import dice_iou
from src.models import set_encoder_frozen, set_train_mode, split_params
from src.train import get_commit, seg_loss

MODES = ["t1c", "t1", "t2", "flair"]
CORE_OUT = ["core", "whole"]
BASE_KIND = {"core": "segmenter", "mode": "classifier"}  # как замораживать энкодер и делить параметры
SPLITS = ("train", "val")


def tables(kind: str, cfg: DataConfig) -> tuple[pd.DataFrame, object]:
    ext = PreprocCache(cfg.cache_dir, "ext", cfg.size)
    if kind == "core":
        fl = PreprocCache(cfg.cache_dir, "upennf", cfg.size)
        t = fl.table[fl.table["split"].isin(SPLITS)].copy()
        t["has_tumor"] = t["has_tumor"].astype(str).str.lower().eq("true")
        t["label"] = 0
        return t.reset_index(drop=True), (ext, fl)
    extm = PreprocCache(cfg.cache_dir, "extm", cfg.size)
    parts = []
    for c in (ext, extm):
        x = c.table
        parts.append(x[(x["source"] == "upenn") & (x["plane"] == "ax") & x["split"].isin(SPLITS) & x["mode"].isin(MODES)])
    t = pd.concat(parts, ignore_index=True)[["path", "mode", "split", "patient", "group", "plane", "slice"]]
    counts = t.groupby(["split", "mode"]).size().unstack()
    if counts.nunique(axis=1).max() != 1:
        raise ValueError(f"у режимов разное число срезов:\n{counts}")
    t["label"] = [MODES.index(m) for m in t["mode"]]
    return t, MultiCache([ext, extm])


class AuxDataset(Dataset):
    def __init__(self, table: pd.DataFrame, kind: str, cache, transform, flair: bool):
        self.table, self.kind, self.cache, self.transform, self.flair = table.reset_index(drop=True), kind, cache, transform, flair

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, i: int) -> dict:
        r = self.table.iloc[i]
        if self.kind == "mode":
            img = self.cache.image(r["path"])
            if self.transform is not None:
                img = self.transform(image=img)["image"]
            x = torch.from_numpy(np.ascontiguousarray(img)).float().div_(255.0)
            return {"image": (x.unsqueeze(0).expand(3, -1, -1) - _MEAN) / _STD, "label": int(r["label"]), "path": r["path"]}
        ext, fl = self.cache
        t1, core = ext.image(r["t1c_path"]), ext.mask(r["t1c_path"])
        f, whole = fl.image(r["path"]), fl.mask(r["path"])
        if self.transform is not None:
            o = self.transform(image=t1, image2=f, mask=core, mask2=whole)
            t1, f, core, whole = o["image"], o["image2"], o["mask"], o["mask2"]
        chans = [t1, f, t1] if self.flair else [t1, t1, t1]
        x = torch.from_numpy(np.ascontiguousarray(np.stack(chans))).float().div_(255.0)
        y = torch.from_numpy(np.ascontiguousarray(np.stack([core > 0, whole > 0]))).float()
        return {"image": (x - _MEAN) / _STD, "mask": y, "label": 0, "path": r["path"]}


def make_loader(table: pd.DataFrame, kind: str, cache, cfg: DataConfig, train: bool, flair: bool) -> DataLoader:
    transform = build_transforms(train, cfg.domain_randomization, cfg.seed)
    if transform is not None and kind == "core":
        transform.add_targets({"image2": "image", "mask2": "mask"})  # та же геометрия и та же яркостная правка
    g = torch.Generator()
    g.manual_seed(cfg.seed)
    return DataLoader(AuxDataset(table, kind, cache, transform, flair), batch_size=cfg.batch_size, shuffle=train, drop_last=train,
                      num_workers=cfg.num_workers, worker_init_fn=_worker_init, generator=g, pin_memory=torch.cuda.is_available(),
                      persistent_workers=cfg.num_workers > 0)


def build(kind: str, encoder: str, pretrained: bool) -> torch.nn.Module:
    if kind == "mode":
        return timm.create_model(encoder, pretrained=pretrained, num_classes=len(MODES))
    return smp.Unet(f"tu-{encoder}", encoder_weights="imagenet" if pretrained else None, in_channels=3, classes=len(CORE_OUT))


def loss_fn(out: torch.Tensor, batch: dict, kind: str, device) -> torch.Tensor:
    if kind == "mode":
        return F.cross_entropy(out.float(), batch["label"].to(device))
    y = batch["mask"].to(device)
    ok = torch.ones(len(y), dtype=torch.bool, device=device)
    return sum(seg_loss(out[:, c : c + 1], y[:, c : c + 1], ok) for c in range(y.shape[1])) / y.shape[1]


@torch.no_grad()
def evaluate(model, kind: str, loader, table: pd.DataFrame, device, min_area: int) -> tuple[dict, pd.DataFrame]:
    model.eval()
    rows = []
    for batch in loader:
        out = model(batch["image"].to(device, non_blocking=True)).float()
        if kind == "mode":
            for p in torch.softmax(out, 1).cpu().numpy():
                rows.append({f"p_{m}": float(v) for m, v in zip(MODES, p)})
            continue
        pr = (torch.sigmoid(out) > 0.5).cpu().numpy()
        gt = batch["mask"].numpy() > 0.5
        for k in range(len(pr)):
            core_p, whole_p = pr[k, 0], pr[k, 1] | pr[k, 0]  # ядро всегда часть всей опухоли
            core_t, whole_t = gt[k, 0], gt[k, 1]
            ed_p, ed_t = whole_p & ~core_p, whole_t & ~core_t
            rows.append({"dice_core": dice_iou(core_p, core_t)[0] if core_t.any() else np.nan,
                         "dice_edema": dice_iou(ed_p, ed_t)[0] if ed_t.any() else np.nan,
                         "dice_whole": dice_iou(whole_p, whole_t)[0] if whole_t.any() else np.nan,
                         "area_core": int(core_p.sum()), "area_whole": int(whole_p.sum()), "true_whole": int(whole_t.sum())})
    frame = pd.concat([table.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    if kind == "mode":
        p = frame[[f"p_{m}" for m in MODES]].to_numpy()
        frame["pred"] = [MODES[k] for k in p.argmax(1)]
        y, yp = frame["label"].to_numpy(), p.argmax(1)
        m = {"acc": float((y == yp).mean()), "macro_f1": float(f1_score(y, yp, labels=list(range(len(MODES))), average="macro"))}
        for k, mode in enumerate(MODES):
            called = yp == k
            m[f"recall_{mode}"] = float((yp[y == k] == k).mean())
            m[f"precision_{mode}"] = float((y[called] == k).mean()) if called.any() else float("nan")
        return m, frame
    neg = frame[~frame["has_tumor"]]
    m = {"dice_core": float(frame["dice_core"].mean()), "dice_edema": float(frame["dice_edema"].mean()),
         "dice_whole": float(frame["dice_whole"].mean()),
         "false_region_rate": float((neg["area_whole"] > min_area).mean()) if len(neg) else float("nan")}
    m["dice_mean"] = (m["dice_core"] + m["dice_edema"]) / 2
    return m, frame


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--limit-train", type=int, default=None)
    ap.add_argument("--limit-val", type=int, default=None)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--processed-dir", default=None)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("models"))
    ap.add_argument("--experiments", type=Path, default=Path("reports/experiments.csv"))
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    cfg = load_config(args.config)
    kind, mc, tc, ec = cfg["kind"], cfg["model"], cfg["train"], cfg["eval"]
    if kind not in BASE_KIND:
        raise ValueError(f"src.train_aux учит только {list(BASE_KIND)}, а в конфиге {kind!r} — используйте src.train")
    for key, val in (("seed", args.seed), ("epochs", args.epochs)):
        if val is not None:
            tc[key] = val
    if args.no_pretrained:
        mc["pretrained"] = False
    flair = mc.get("inputs", "t1c") == "t1c_flair"
    known = {f.name for f in fields(DataConfig)}
    over = {"batch_size": args.batch_size, "num_workers": args.num_workers, "processed_dir": args.processed_dir, "cache_dir": args.cache_dir}
    data = DataConfig(**{**{k: v for k, v in cfg["data"].items() if k in known}, **{k: v for k, v in over.items() if v is not None}})
    data = replace(data, seed=tc["seed"])
    cfg["data"] = {f.name: getattr(data, f.name) for f in fields(DataConfig)}
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    use_amp = bool(tc["amp"]) and device.type == "cuda"
    seed_everything(tc["seed"])
    commit = get_commit()

    table, cache = tables(kind, data)
    train, val = table[table["split"] == "train"], table[table["split"] == "val"]
    if args.limit_train:
        train = train.groupby("label" if kind == "mode" else "has_tumor", group_keys=False).head(max(1, args.limit_train // 2))
    if args.limit_val:
        val = val.groupby("label" if kind == "mode" else "has_tumor", group_keys=False).head(max(1, args.limit_val // 2))
    train, val = train.reset_index(drop=True), val.reset_index(drop=True)
    train_loader = make_loader(train, kind, cache, data, True, flair)
    val_loader = make_loader(val, kind, cache, replace(data, domain_randomization=False), False, flair)

    base = BASE_KIND[kind]
    model = build(kind, mc["encoder"], mc["pretrained"]).to(device)
    enc_params, head_params = split_params(model, base)
    opt = torch.optim.AdamW([{"params": enc_params, "lr": tc["lr_encoder"]}, {"params": head_params, "lr": tc["lr_head"]}],
                            weight_decay=tc["weight_decay"])
    steps = len(train_loader) * tc["epochs"]
    warm = max(1, len(train_loader) * tc["warmup_epochs"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(s, steps) / max(steps, 1))))
    scaler = torch.amp.GradScaler(enabled=use_amp)
    key = "macro_f1" if kind == "mode" else "dice_mean"

    run_dir = args.out_dir / f"{cfg['run']}_s{tc['seed']}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"запуск {run_dir.name}: {kind} ({'T1c+FLAIR' if flair else 'T1c'}), {mc['encoder']}, устройство {device}, коммит {commit[:12]}, "
          f"train {len(train)}, val {len(val)}, батчей в эпохе {len(train_loader)}", flush=True)

    history, best, best_val, t_start = [], None, -np.inf, time.perf_counter()
    for epoch in range(1, tc["epochs"] + 1):
        frozen = epoch <= tc["epochs_frozen"]
        set_encoder_frozen(model, base, frozen)
        set_train_mode(model, base, frozen)
        t0, total, n = time.perf_counter(), 0.0, 0
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = loss_fn(model(batch["image"].to(device, non_blocking=True)), batch, kind, device)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += float(loss.detach()) * len(batch["label"])
            n += len(batch["label"])
        t1 = time.perf_counter()
        metrics, _ = evaluate(model, kind, val_loader, val, device, ec["min_mask_area_px"])
        row = {"epoch": epoch, "phase": "головы" if frozen else "вся сеть", "train_loss": total / max(n, 1),
               "sec_train": t1 - t0, "sec_val": time.perf_counter() - t1, **metrics}
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False, encoding="utf-8")
        if metrics[key] > best_val:  # при равенстве остается более ранняя эпоха
            best, best_val = epoch, metrics[key]
            torch.save(model.state_dict(), run_dir / "best_state.pt")
        shown = {k: v for k, v in metrics.items() if k in ("macro_f1", "precision_t1c", "dice_core", "dice_edema", "false_region_rate")}
        print(f"эпоха {epoch:3d} ({row['phase']}): loss {row['train_loss']:.4f} | " + ", ".join(f"{k} {v:.3f}" for k, v in shown.items())
              + f" | {row['sec_train']:.0f} с + {row['sec_val']:.0f} с", flush=True)
    train_sec = time.perf_counter() - t_start

    state = torch.load(run_dir / "best_state.pt", map_location=device)
    model.load_state_dict(state)
    metrics, frame = evaluate(model, kind, val_loader, val, device, ec["min_mask_area_px"])
    if abs(metrics[key] - best_val) > 1e-4:
        raise RuntimeError(f"веса эпохи {best} после загрузки дают {key} = {metrics[key]:.6f}, а при обучении было {best_val:.6f}")
    frame.to_csv(run_dir / "val_predictions.csv", index=False, encoding="utf-8")
    torch.save({"model": state, "kind": kind, "encoder": mc["encoder"], "classes": MODES if kind == "mode" else CORE_OUT,
                "inputs": "t1c_flair" if flair else "t1c", "config": cfg, "epoch": best, "metrics": metrics, "commit": commit,
                "seed": tc["seed"], "preprocess_version": CACHE_VERSION}, run_dir / "best.pt")
    (run_dir / "best_state.pt").unlink()

    sec_epoch = float(np.mean([r["sec_train"] + r["sec_val"] for r in history]))
    exp = {"date": datetime.now().strftime("%Y-%m-%d %H:%M"), "run": cfg["run"], "kind": kind, "encoder": mc["encoder"],
           "pretrained": mc["pretrained"], "seed": tc["seed"], "commit": commit, "config": args.config.as_posix(),
           "epochs": tc["epochs"], "epochs_frozen": tc["epochs_frozen"], "lr_head": tc["lr_head"], "lr_encoder": tc["lr_encoder"],
           "batch_size": data.batch_size, "domain_randomization": data.domain_randomization, "table": f"{kind}:{'t1c_flair' if flair else 't1c'}",
           "n_train": len(train), "n_val": len(val), "best_epoch": best,
           **{f"val_{k}": (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()},
           "sec_per_epoch": round(sec_epoch, 1), "train_min": round(train_sec / 60, 1), "device": str(device),
           "weights": (run_dir / "best.pt").as_posix(), "note": args.note}
    args.experiments.parent.mkdir(parents=True, exist_ok=True)
    out = pd.concat([pd.read_csv(args.experiments), pd.DataFrame([exp])]) if args.experiments.exists() else pd.DataFrame([exp])
    out.to_csv(args.experiments, index=False, encoding="utf-8")
    print(f"выбрана эпоха {best}: " + ", ".join(f"{k} {v:.3f}" for k, v in metrics.items() if isinstance(v, float) and not np.isnan(v)), flush=True)
    print(f"веса: {run_dir / 'best.pt'}; время {train_sec / 60:.1f} мин ({sec_epoch:.0f} с на эпоху)", flush=True)


if __name__ == "__main__":
    main()
