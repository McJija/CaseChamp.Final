"""Модели: классификатор, сегментатор и многозадачная сеть на одном и том же энкодере.

kind:
  classifier — timm, 4 класса;
  segmenter  — smp.Unet, бинарная маска опухоли;
  multitask  — smp.Unet + классификационная голова на признаках энкодера (aux_params).
В smp энкодер берется с префиксом `tu-` — это та же сеть timm с теми же предобученными
весами, что и у классификатора.

Двухфазное дообучение: сначала энкодер заморожен (учатся только головы), затем учится вся
сеть, у энкодера скорость обучения ниже.

Выравнивание резкости (круг 4, `harmonize` = R > 0): перед сетью снимок S×S уменьшается до R×R (среднее по площади)
и увеличивается обратно (билинейно). Делается крючком перед forward — одинаково в обучении (после аугментаций),
при проверке и в прототипе; на веса и их имена не влияет. Интерполяция линейна, поэтому порядок с нормализацией
ImageNet не важен.
"""
from __future__ import annotations

import segmentation_models_pytorch as smp
import timm
import torch
import torch.nn.functional as F
from torch import nn

from src.labels import CLASSES

KINDS = ("classifier", "segmenter", "multitask")


def harmonize_tensor(x: torch.Tensor, res: int) -> torch.Tensor:
    """N×C×S×S → то же, но с «эффективной резкостью» R×R (R ≤ 0 или R ≥ S — без изменений)."""
    size = tuple(x.shape[-2:])
    if res <= 0 or res >= min(size):
        return x
    y = F.interpolate(x, size=(res, res), mode="area")
    return F.interpolate(y, size=size, mode="bilinear", align_corners=False)


class _Harmonize:
    def __init__(self, res: int):
        self.res = int(res)

    def __call__(self, module: nn.Module, args: tuple) -> tuple:
        return (harmonize_tensor(args[0], self.res), *args[1:])


def build_model(kind: str, encoder: str, pretrained: bool = True, dropout: float = 0.2, harmonize: int = 0) -> nn.Module:
    if kind not in KINDS:
        raise ValueError(f"неизвестный вид модели: {kind!r}")
    if kind == "classifier":
        model = timm.create_model(encoder, pretrained=pretrained, num_classes=len(CLASSES))
    else:
        aux = {"classes": len(CLASSES), "pooling": "avg", "dropout": dropout} if kind == "multitask" else None
        model = smp.Unet(f"tu-{encoder}", encoder_weights="imagenet" if pretrained else None, in_channels=3, classes=1, aux_params=aux)
    model.harmonize = int(harmonize or 0)
    if model.harmonize > 0:
        model.register_forward_pre_hook(_Harmonize(model.harmonize))
    return model


def split_params(model: nn.Module, kind: str) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Параметры энкодера и параметры голов (у сегментатора голова — декодер и выходные слои)."""
    if kind == "classifier":
        head_ids = {id(p) for p in model.get_classifier().parameters()}
        enc = [p for p in model.parameters() if id(p) not in head_ids]
        head = [p for p in model.parameters() if id(p) in head_ids]
    else:
        enc = list(model.encoder.parameters())
        enc_ids = {id(p) for p in enc}
        head = [p for p in model.parameters() if id(p) not in enc_ids]
    return enc, head


def set_encoder_frozen(model: nn.Module, kind: str, frozen: bool) -> None:
    enc, _ = split_params(model, kind)
    for p in enc:
        p.requires_grad = not frozen


def set_train_mode(model: nn.Module, kind: str, encoder_frozen: bool) -> None:
    """Режим обучения; у замороженного энкодера слои нормализации остаются в режиме оценки,
    чтобы их накопленные статистики не сдвигались."""
    model.train()
    if not encoder_frozen:
        return
    if kind == "classifier":
        head_modules = set(model.get_classifier().modules())
        for m in model.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm) and m not in head_modules:
                m.eval()
    else:
        model.encoder.eval()


def forward(model: nn.Module, kind: str, x: torch.Tensor) -> dict[str, torch.Tensor]:
    """Единый выход: logits (B×4) и/или mask_logits (B×1×H×W)."""
    if kind == "classifier":
        return {"logits": model(x)}
    out = model(x)
    if kind == "multitask":
        return {"mask_logits": out[0], "logits": out[1]}
    return {"mask_logits": out}
