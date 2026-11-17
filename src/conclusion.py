"""Проект заключения по шаблону из результата анализа (без языковой модели) и его автоматическая сверка (П8).

    from src.conclusion import build, verify
    text, facts = build(result, file_name)      # result — src.system.SystemPredictor.predict(...)
    report = verify(text, result, file_name)    # report["ok"] — все числа и утверждения совпадают с результатом

Правило П8 (`configs/eval_external.yaml`): каждое число и утверждение текста сверяется с результатом анализа; требуется 100%
совпадений, иначе это ошибка кода. Сверка независима от построения текста: ожидаемые значения заново берутся из
результата (маски, ответы копий), а не из переменных, по которым строился текст. Все числа текста должны найтись среди
ожидаемых — посторонних чисел быть не должно. Текст, исправленный врачом, сверяется так же: расхождения показываются
как предупреждения (врач вправе менять текст), а не как ошибка.

Ограничение «ложные тревоги на незнакомых данных» — свойство системы, а не снимка: число берется из замера
(reports/run4/candidates_compare.md, `run3`, оба экзамена вместе: 617 из 2883 срезов = 21,4%; хотя бы одна тревога —
у 255 из 292 пациентов, reports/run3/review_numbers.md).

Режимы, отличные от T1 с контрастом (reports/run4/candidates_compare.md, раздел 7; reports/run3/review_numbers.md):
система проверена на них только на глиобластомах UPENN-GBM и находит опухоль плохо, поэтому ответ «опухоли нет» на
таких режимах не выдается как вывод, а тип опухоли помечается как ненадежный.
"""
from __future__ import annotations

import re

import cv2
import numpy as np

from src.system import max_diameter

FALSE_ALARM = {"k": 617, "n": 2883, "pk": 255, "pn": 292}  # run3, срезы T1 с контрастом без опухоли, два экзамена
# run3 на глиобластомах UPENN-GBM, аксиальные срезы val + отложенные пациенты (57 пациентов, 222 среза с опухолью):
# найдено опухолей и верный тип среди найденных (тип у всех — глиома)
MODE_FOUND = {"t1": (53, 222), "t2": (98, 222), "flair": (203, 222)}
MODE_TYPE = {"t1": (19, 53), "t2": (20, 98), "flair": (41, 203)}
TYPE_RU_GEN = {"glioma": "глиома", "meningioma": "менингиома", "pituitary": "опухоль гипофиза"}
NUM = re.compile(r"\d+(?:[.,]\d+)?")


def pct(x: float, nd: int = 0) -> str:
    return f"{x * 100:.{nd}f}".replace(".", ",") + "%"


def false_alarm_note() -> str:
    fa = FALSE_ALARM
    return (f"На срезах T1 с контрастом без опухоли у пациентов с другими болезнями (рассеянный склероз, шванномы) из больниц, "
            f"снимков которых не было в обучении, система при проверке ошибочно видела опухоль на {fa['k']} из {fa['n']} срезов "
            f"({pct(fa['k'] / fa['n'], 1)}), хотя бы на одном срезе — у {fa['pk']} из {fa['pn']} пациентов")


def absence_note(mode: str | None, mode_ru: str) -> str:
    """Почему «опухоли нет» на этом режиме — не вывод (пустая строка для T1 с контрастом)."""
    if mode in MODE_FOUND:
        k, n = MODE_FOUND[mode]
        return (f"на режиме «{mode_ru}» отсутствие находки не оценивается: при проверке на глиобластомах одной больницы "
                f"система на этом режиме нашла опухоль на {k} из {n} срезов ({pct(k / n)}), другие типы опухоли на нем не проверялись")
    if not mode:
        return ("это имеет смысл только для T1 с контрастом: режим не указан, а на других режимах отсутствие находки "
                "не оценивается")
    return ""


def type_note(mode: str | None, mode_ru: str) -> str:
    """Тип опухоли на режиме, отличном от T1 с контрастом (пустая строка, если проверки нет)."""
    if mode not in MODE_TYPE:
        return ""
    k, n = MODE_TYPE[mode]
    return (f"На режиме «{mode_ru}» тип опухоли ненадежен: при проверке на глиобластомах тип был верен на {k} из {n} "
            f"срезов с найденной опухолью ({pct(k / n)})")


def expected(r: dict, file_name: str) -> dict[str, str]:
    """Ожидаемые значения — заново из результата анализа (используются и в тексте, и в сверке)."""
    w, h = r["image_size"]
    e = {"file": file_name, "width": str(w), "height": str(h), "mode": r["mode_ru"],
         "fa_k": str(FALSE_ALARM["k"]), "fa_n": str(FALSE_ALARM["n"]), "fa_pct": pct(FALSE_ALARM["k"] / FALSE_ALARM["n"], 1),
         "fa_pk": str(FALSE_ALARM["pk"]), "fa_pn": str(FALSE_ALARM["pn"])}
    if r["present"]:
        e["type"] = TYPE_RU_GEN[r["type"]]
        e["share"] = pct(r["type_share"])
        if r["mode"] in MODE_TYPE:
            k, n = MODE_TYPE[r["mode"]]
            e["mt_k"], e["mt_n"], e["mt_pct"] = str(k), str(n), pct(k / n)
    elif r["mode"] in MODE_FOUND:
        k, n = MODE_FOUND[r["mode"]]
        e["mf_k"], e["mf_n"], e["mf_pct"] = str(k), str(n), pct(k / n)
    m = r["mask"]
    if m.any():
        meas = r["measure"]
        pos = meas.get("position") or {"v": "—", "h": "—"}
        e["area"] = str(int(m.sum()))
        e["head_frac"] = pct(meas["head_fraction"], 1)
        e["diam"] = str(int(round(max_diameter(m))))
        e["n_regions"] = str(cv2.connectedComponents(m.astype(np.uint8), connectivity=8)[0] - 1)
        e["pos_v"], e["pos_h"] = pos["v"], pos["h"]
        if r["core"] is not None:
            e["core"], e["edema"] = str(int(r["core"].sum())), str(int(r["edema"].sum()))
    if r["uncertain"].any():
        e["unc_px"] = str(int(r["uncertain"].sum()))
    return e


def build(r: dict, file_name: str) -> tuple[str, dict[str, str]]:
    e = expected(r, file_name)
    L = ["ПРОЕКТ ЗАКЛЮЧЕНИЯ",
         "Сформирован автоматически системой поддержки врача. Требует проверки и подписи врача; не является медицинским заключением.",
         "",
         f"Исследование: МРТ головного мозга, один срез — файл «{e['file']}», {e['width']}×{e['height']} пикс.",
         f"Режим МРТ: {e['mode']}{' (указан врачом)' if r['mode'] else ''}.",
         "", "Находки:"]
    if not r["present"]:
        note = absence_note(r["mode"], e["mode"])
        L.append("— Признаков опухоли (глиомы, менингиомы, опухоли гипофиза) на срезе система не выявила"
                 + (f", но {note}." if r["mode"] in MODE_FOUND else f". {note[:1].upper() + note[1:]}." if note else "."))
        if "area" in e:
            L.append("— Врач отметил область вручную.")
    else:
        L.append(f"— На срезе выявлена область, подозрительная на опухоль. Предполагаемый тип: {e['type']} "
                 f"(доля этого типа среди трех типов — {e['share']}).")
    if "area" in e:
        L.append(f"— Расположение на изображении{' (самый крупный участок)' if e['n_regions'] != '1' else ''}: по вертикали — {e['pos_v']} треть головы, по горизонтали — {e['pos_h']} треть "
                 f"(сторона указана по изображению, не анатомически). Отдельных участков: {e['n_regions']}.")
        L.append(f"— Измерения: площадь области — {e['area']} пикс. ({e['head_frac']} площади головы на срезе), "
                 f"наибольший размер — {e['diam']} пикс. Размеры в миллиметрах не указаны: в файле нет сведений о масштабе.")
        if "core" in e:
            L.append(f"— Ядро опухоли (контрастируемая часть и некроз) — {e['core']} пикс.; окружающая зона изменений "
                     f"(отек, инфильтрация) — {e['edema']} пикс. Оценка ориентировочная: модель ядра и отека обучена на "
                     "глиобластомах одного набора данных, на снимках из других наборов ее точность не подтверждена. Ядро и зону "
                     "изменений выделяет отдельная модель, поэтому их площади не обязаны сходиться с площадью основной области.")
        elif r["core_applied"]:
            L.append("— Ядро опухоли и окружающую зону изменений выделить не удалось.")
    elif r["present"]:
        L.append("— Модель контура область не выделила: вывод «опухоль есть» сделан только по модели типа, "
                 "расположение и размеры не определены.")
    if "area" in e:
        legend = ("Соответствие снимку: красный контур — выделенная область; желтым отмечены участки, где граница неуверенна"
                  + ("; фиолетовым — ядро опухоли, голубым — зона изменений." if "core" in e else "."))
    elif "unc_px" in e:
        legend = (f"Соответствие снимку: основная копия системы область не выделила; желтым отмечен участок {e['unc_px']} пикс., "
                  "который выделили другие копии, — проверьте его.")
    else:
        legend = "Соответствие снимку: выделенных областей нет."
    L += ["", legend, ""]
    if r["check"]:
        L.append("Уверенность: ПРОВЕРЬТЕ — " + "; ".join(r["check_reasons"]) + ".")
    elif r["mode"] in MODE_FOUND:
        L.append(f"Уверенность: не оценивается — три копии системы согласны между собой, но на режиме «{e['mode']}» "
                 "система проверена мало и часто ошибается.")
    else:
        L.append("Уверенность: уверенно — три независимые копии системы согласны в ответе и типе, модели внутри системы не расходятся.")
    L += ["", "Сомнения и ограничения:"]
    if r["mode_warning"]:
        L.append(f"— Режим МРТ {'не указан' if not r['mode'] else 'отличается от T1 с контрастом'}: система рассчитана на T1 с "
                 "контрастом, результат требует тщательной перепроверки.")
    if r["present"] and type_note(r["mode"], e["mode"]):
        L.append(f"— {type_note(r['mode'], e['mode'])}.")
    if "area" in e and "unc_px" in e:
        L.append(f"— Граница области неуверенна на участке {e['unc_px']} пикс.: там копии системы расходятся.")
    L += ["— Проанализирован один двумерный срез; другие срезы исследования не учитывались.",
          "— Система различает только три типа опухоли; другие изменения (метастазы, шванномы, очаги рассеянного склероза и др.) "
          "не распознаются и могут быть приняты за опухоль.",
          f"— {false_alarm_note()}; пометка «проверьте» такие ошибки надежно не ловит.",
          "— Сторона поражения указана по изображению; анатомическая сторона не определяется."]
    return "\n".join(L), e


def _numbers(text: str, e: dict) -> list[str]:
    t = text.replace(e["file"], " ")
    t = t.replace("T1", " ").replace("2D", " ")  # обозначения режимов — не числа
    return [m.group(0) for m in NUM.finditer(t)]


def verify(text: str, r: dict, file_name: str, edited: bool = False) -> dict:
    """Сверка текста с результатом. Для текста системы — ошибки (П8: нужно 100%), для текста врача — предупреждения."""
    e = expected(r, file_name)
    problems, checked = [], 0
    allowed = set()
    for v in e.values():
        allowed |= {m.group(0) for m in NUM.finditer(v)} if v != e["file"] else set()
    allowed |= {"60"}  # порог доли типа в правиле «проверьте» (configs/confidence_run3.yaml)
    for x in _numbers(text, e):
        checked += 1
        if x not in allowed and x.rstrip("%") not in allowed:
            problems.append(f"число {x} не следует из результата анализа")
    claims = [("выявлена область, подозрительная на опухоль" in text, r["present"], "наличие опухоли"),
              ("не выявила" in text, not r["present"], "отсутствие опухоли"),
              (f"на режиме «{e['mode']}» отсутствие находки не оценивается" in text, not r["present"] and r["mode"] in MODE_FOUND,
               "отсутствие находки на этом режиме не оценивается"),
              (f"На режиме «{e['mode']}» тип опухоли ненадежен" in text, r["present"] and r["mode"] in MODE_TYPE,
               "тип опухоли на этом режиме ненадежен"),
              ("ПРОВЕРЬТЕ" in text, r["check"], "пометка «проверьте»"),
              (f"Режим МРТ: {e['mode']}" in text, True, "режим МРТ")]
    if r["present"]:
        claims.append((f"Предполагаемый тип: {e['type']}" in text, True, "тип опухоли"))
        for k in ("area", "diam", "head_frac", "core", "edema"):
            if k in e:
                claims.append((e[k] in text, True, f"значение «{k}»"))
    for said, truth, what in claims:
        checked += 1
        if said != truth:
            problems.append(f"утверждение «{what}» не совпадает с результатом анализа")
    return {"ok": not problems, "checked": checked, "problems": problems, "edited": edited}
