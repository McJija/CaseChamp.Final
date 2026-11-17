"""Прототип системы поддержки врача (версия финала, `run3`): снимок → выделение опухоли → уверенность →
проект заключения → проверка и правка врачом → сохранение.

Запуск — двойной щелчок по start_windows.bat (macOS и Linux: bash start_mac_linux.sh) или вручную:
    python app/launch.py            # сам выберет порт и откроет браузер
    streamlit run app/app.py        # то же через Streamlit

Вся логика модели — в src/system.py, текст заключения и его сверка — в src/conclusion.py; здесь только показ и правка.
Параметры адреса (для показа и скриншотов): ?example=<файл из app/examples>, ?mode=t1c|t1|t2|flair, ?limits=1.
Сохраненные результаты — в data/annotations/<время>_<имя снимка>/.
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.conclusion import MODE_FOUND, absence_note, build, false_alarm_note, pct, type_note, verify  # noqa: E402
from src.labels import CLASSES, CLS_RU  # noqa: E402
from src.system import MODE_RU, load_system, measure  # noqa: E402

EXAMPLES = Path(__file__).resolve().parent / "examples"
ANNOT = ROOT / "data" / "annotations"
DISCLAIMER = "Результат работы системы не является медицинским заключением. Решение принимает врач."
UNKNOWN_WARNING = f"{false_alarm_note()}. Отсутствие пометки «проверьте» не гарантирует, что ответ верен."
COLORS = {"mask": (255, 60, 60), "uncertain": (255, 205, 0), "core": (170, 70, 255), "edema": (0, 190, 255), "system": (200, 200, 200)}
MODES = [None, "t1c", "t1", "t2", "flair"]
DISPLAY_MAX = 512  # сторона картинки для правки


@st.cache_resource(show_spinner="Загрузка моделей (около 10 секунд)…")
def get_system():
    return load_system()


@st.cache_data(show_spinner=False, max_entries=16)
def analyze(data: bytes) -> dict:
    """Ответ моделей не зависит от выбранного режима — считается один раз на снимок."""
    return get_system().predict(decode(data), mode=None)


def decode(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("файл не удалось прочитать как изображение")
    return img


def with_mode(r: dict, mode: str | None) -> dict:
    return {**r, "mode": mode, "mode_ru": MODE_RU.get(mode, "не указан"), "mode_warning": mode != "t1c"}


def with_edits(r: dict, mask: np.ndarray) -> dict:
    """Результат с областью, исправленной врачом: измерения пересчитываются по новой области."""
    out = {**r, "mask": mask}
    out["measure"] = measure(mask, r["head"], r["core"], r["edema"])
    return out


def compose(gray: np.ndarray, r: dict, mask: np.ndarray, show: dict, alpha: float, edited: bool) -> np.ndarray:
    rgb = np.repeat(gray[..., None], 3, 2).astype(np.float32)

    def fill(m, color, a):
        if m is not None and m.any():
            rgb[m] = (1 - a) * rgb[m] + a * np.array(color, np.float32)

    if show["edema"]:
        fill(r["edema"], COLORS["edema"], alpha * 0.8)
    if show["core"]:
        fill(r["core"], COLORS["core"], alpha)
    if show["uncertain"]:
        fill(r["uncertain"] & ~mask, COLORS["uncertain"], alpha * 0.9)
        fill(r["uncertain"] & mask, COLORS["uncertain"], alpha * 0.5)
    out = np.ascontiguousarray(np.clip(rgb, 0, 255).astype(np.uint8))
    thick = max(1, round(max(gray.shape) / 400))
    if edited and r["mask"].any():
        cnt, _ = cv2.findContours(r["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, cnt, -1, COLORS["system"], thick)
    if show["mask"] and mask.any():
        cnt, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, cnt, -1, COLORS["mask"], thick + 1)
    return out


def figure(img: np.ndarray, scale: float) -> go.Figure:
    """Снимок для правки: обводка мышью (лассо) → многоугольник в координатах снимка. Невидимая сетка точек нужна,
    чтобы у графика был режим выделения."""
    h, w = img.shape[:2]
    step = max(4, int(max(h, w) / 64))
    gx, gy = np.meshgrid(np.arange(0, w, step), np.arange(0, h, step))
    fig = go.Figure(go.Image(z=img, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=gx.ravel(), y=gy.ravel(), mode="markers", marker=dict(size=3, opacity=0), hoverinfo="skip",
                             showlegend=False))
    fig.update_layout(dragmode="lasso", margin=dict(l=0, r=0, t=0, b=0), height=min(640, int(h * 1.0) + 10),
                      xaxis=dict(visible=False, range=[0, w]), yaxis=dict(visible=False, range=[h, 0], scaleanchor="x"),
                      newselection=dict(line=dict(color="#22c55e", width=2)), activeselection=dict(fillcolor="#22c55e", opacity=0.15))
    return fig


def polygon_mask(sel: dict, shape: tuple[int, int], scale: float) -> np.ndarray | None:
    """Обводка или прямоугольник из выделения plotly → маска в координатах исходного снимка."""
    m = np.zeros(shape, np.uint8)
    got = False
    for las in sel.get("lasso", []) or []:
        xs, ys = las.get("x", []), las.get("y", [])
        if len(xs) >= 3:
            pts = np.round(np.stack([np.array(xs, float), np.array(ys, float)], 1) / scale).astype(np.int32)
            cv2.fillPoly(m, [pts], 1)
            got = True
    for box in sel.get("box", []) or []:
        xs, ys = box.get("x", []), box.get("y", [])
        if len(xs) >= 2:
            x0, x1 = sorted(int(round(v / scale)) for v in xs[:2])
            y0, y1 = sorted(int(round(v / scale)) for v in ys[:2])
            m[max(0, y0):y1 + 1, max(0, x0):x1 + 1] = 1
            got = True
    return m.astype(bool) if got else None


def save_bundle(name: str, data: bytes, r: dict, mask: np.ndarray, text_sys: str, text_final: str, edits: list,
                check_sys: dict, check_final: dict, overlay: np.ndarray) -> tuple[Path, bytes]:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    folder = ANNOT / f"{stamp}_{Path(name).stem}"
    folder.mkdir(parents=True, exist_ok=True)
    png = lambda a: cv2.imencode(".png", a)[1].tobytes()  # noqa: E731
    files = {
        f"снимок{Path(name).suffix.lower() or '.png'}": data,
        "область_система.png": png(r["mask"].astype(np.uint8) * 255),
        "область_итог.png": png(mask.astype(np.uint8) * 255),
        "неуверенная_граница.png": png(r["uncertain"].astype(np.uint8) * 255),
        "наложение.png": png(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)),
        "заключение_проект_системы.txt": text_sys.encode("utf-8"),
        "заключение_итог.txt": text_final.encode("utf-8"),
    }
    if r["core"] is not None:
        files["ядро.png"] = png(r["core"].astype(np.uint8) * 255)
        files["отек.png"] = png(r["edema"].astype(np.uint8) * 255)
    meta = {"сохранено": stamp, "снимок": name, "режим_МРТ": r["mode_ru"], "система": r["system"],
            "ответ_системы": {"опухоль": r["present"], "тип": r["type"], "доля_типа": r["type_share"], "p_опухоли": r["p_tumor"],
                              "проверьте": r["check"], "причины": r["check_reasons"], "копии": r["copies"],
                              "ядро_и_отек_применены": r["core_applied"]},
            "измерения_система": r["measure"], "измерения_итог": measure(mask, r["head"], r["core"], r["edema"]),
            "правки_областей": edits, "текст_изменен_врачом": text_final.strip() != text_sys.strip(),
            "сверка_П8_проект": check_sys, "сверка_итогового_текста": check_final}
    files["результат.json"] = json.dumps(meta, ensure_ascii=False, indent=1, default=str).encode("utf-8")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for fn, content in files.items():
            (folder / fn).write_bytes(content)
            z.writestr(fn, content)
    return folder, buf.getvalue()


# ---------------------------------------------------------------- экран

st.set_page_config(page_title="МРТ: поддержка врача", layout="wide")
st.title("Опухоли головного мозга на МРТ — система поддержки врача")
st.warning(DISCLAIMER)

qp = st.query_params
with st.sidebar:
    st.header("1. Снимок")
    uploaded = st.file_uploader("Загрузите срез МРТ (JPG или PNG)", type=["jpg", "jpeg", "png"])
    examples = sorted(p.name for p in EXAMPLES.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")) if EXAMPLES.exists() else []
    options = ["—", *examples]
    wanted = qp.get("example", "—")
    example = st.selectbox("или выберите пример", options, index=options.index(wanted) if wanted in options else 0) if examples else "—"
    st.header("2. Режим МРТ")
    q_mode = qp.get("mode")
    mode = st.radio("Выберите режим снимка", MODES, index=MODES.index(q_mode) if q_mode in MODES else 0,
                    format_func=lambda m: MODE_RU[m] if m else "не указан")
    st.caption("Система рассчитана на T1 с контрастом. Для других режимов и если режим не указан, результат выдается "
               "с предупреждением о тщательной перепроверке; на других режимах отсутствие опухоли не оценивается — система "
               "находит на них опухоль плохо. Автоматически режим не определяется: на проверке определитель режима "
               "ошибался чаще, чем был прав.")
    st.header("3. Отображение")
    show = {"mask": st.checkbox("Контур опухоли", True), "uncertain": st.checkbox("Неуверенная граница", True),
            "core": st.checkbox("Ядро опухоли (глиома)", True), "edema": st.checkbox("Зона изменений вокруг (глиома)", True)}
    alpha = st.slider("Яркость заливки", 0.1, 1.0, 0.45, 0.05)

if uploaded is not None:
    name, data = uploaded.name, uploaded.getvalue()
elif example != "—":
    name, data = example, (EXAMPLES / example).read_bytes()
else:
    st.info("Загрузите снимок или выберите пример в панели слева.")
    st.stop()

try:
    image = decode(data)
except ValueError as e:
    st.error(f"{name}: {e}")
    st.stop()

with st.spinner("Анализ снимка: три копии системы, ядро и отек…"):
    r = with_mode(analyze(data), mode)

key = hashlib.md5(data).hexdigest()[:12]
ss = st.session_state
if ss.get("img_key") != key:  # новый снимок — новые правки и новый текст
    for k_, v_ in dict(img_key=key, mask=r["mask"].copy(), edits=[], history=[], text=None, text_basis=None, text_sys_at=None, saved=None).items():
        ss[k_] = v_

mask = ss.mask
edited = bool(ss.edits)
re_ = with_edits(r, mask) if edited else r

left, right = st.columns([5, 4])
with right:
    st.subheader("Ответ системы")
    if r["present"]:
        st.markdown(f"### Подозрение на опухоль — **{r['type_ru']}**")
        st.caption(f"Доля этого типа среди трех типов — {r['type_share']:.0%}. Вероятность опухоли у модели типа — {r['p_tumor']:.1%}.")
        if type_note(mode, r["mode_ru"]):
            st.warning(type_note(mode, r["mode_ru"]) + ".")
    elif mode in MODE_FOUND:
        st.markdown(f"### Находок нет — на режиме «{r['mode_ru']}» это не оценивается")
        note = absence_note(mode, r["mode_ru"])
        st.warning(note[:1].upper() + note[1:] + ". Оцените снимок самостоятельно или загрузите срез T1 с контрастом.")
    else:
        st.markdown("### Признаков опухоли не выявлено")
        if not mode:
            note = absence_note(mode, r["mode_ru"])
            st.caption(note[:1].upper() + note[1:] + ".")
    if r["check"]:
        why = "; ".join(r["check_reasons"])
        st.error("**ПРОВЕРЬТЕ.** " + why[:1].upper() + why[1:] + ".")
    elif mode in MODE_FOUND:
        st.info(f"**Уверенность не оценивается:** три копии системы согласны между собой, но на режиме «{r['mode_ru']}» "
                "система проверена мало и часто ошибается.")
    else:
        st.success("**Уверенно:** три копии системы согласны, модели внутри системы не расходятся.")
    if r["mode_warning"]:
        st.warning(f"Режим МРТ {'не указан' if not mode else '«' + MODE_RU[mode] + '»'}: система рассчитана на T1 с контрастом — "
                   "результат требует тщательной перепроверки.")
    st.info(UNKNOWN_WARNING)
    m = re_["measure"]
    if m.get("area_px"):
        pos = m["position"]
        st.markdown(f"**Расположение на изображении:** по вертикали — {pos.get('v', '—')} треть головы, "
                    f"по горизонтали — {pos.get('h', '—')} треть  \n"
                    f"**Площадь:** {m['area_px']} пикс. ({pct(m['head_fraction'], 1)} площади головы)  \n"
                    f"**Наибольший размер:** {round(m['max_diam_px'])} пикс.  \n"
                    + (f"**Ядро / зона изменений:** {m['core_px']} / {m['edema_px']} пикс. (ориентировочно)" if "core_px" in m else ""))
        st.caption("Миллиметры не указаны: в файле JPG/PNG нет сведений о масштабе." + (" Измерения — по области после правки врача." if edited else ""))
    elif r["present"]:
        st.caption("Модель контура область не выделила: вывод «опухоль есть» сделан только по модели типа.")
    with st.expander("Подробности: ответы трех копий"):
        for i, c in enumerate(r["copies"]):
            st.markdown(f"- Копия {i + 1}: {'опухоль — ' + CLS_RU[c['type']] if c['present'] else 'нет опухоли'}; "
                        f"область {'найдена' if c['found'] else 'не найдена'}; вероятность опухоли {c['p_tumor']:.1%}")
        for cls in CLASSES:
            st.progress(float(r["probs"][cls]), text=f"{CLS_RU[cls].capitalize()} — {r['probs'][cls]:.1%} (основная копия)")
        st.caption(f"Время анализа: {r['time_s']:.1f} с. Версия системы: финал (`run3`: три копии, порог "
                   f"{r['system']['threshold']:.6f}, пометка «проверьте» — {r['system']['check_rule']}).")

with left:
    st.subheader("Снимок и выделенные области")
    overlay = compose(image, r, mask, show, alpha, edited)
    scale = DISPLAY_MAX / max(image.shape)  # маленькие снимки увеличиваются для удобной правки
    shown = cv2.resize(overlay, (round(image.shape[1] * scale), round(image.shape[0] * scale)), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR) if scale != 1 else overlay
    legend = ["<span style='color:#ff3c3c'>■</span> контур", "<span style='color:#ffcd00'>■</span> неуверенная граница"]
    if r["core"] is not None:
        legend += ["<span style='color:#aa46ff'>■</span> ядро", "<span style='color:#00beff'>■</span> зона изменений"]
    if edited:
        legend.append("<span style='color:#aaaaaa'>■</span> контур системы до правки")
    st.markdown(" &nbsp; ".join(legend), unsafe_allow_html=True)
    event = st.plotly_chart(figure(shown, scale), key=f"fig_{key}_{len(ss.edits)}", on_select="rerun", selection_mode=("lasso", "box"),
                            config={"displaylogo": False, "modeBarButtonsToRemove": ["toImage"]})
    if r["core"] is not None:
        st.caption("Ядро и зона изменений — ориентировочно: модель обучена на глиобластомах одного набора (там совпадение ядра "
                   "с обводкой врача 0,89). На глиомах нашего набора совпадение всей опухоли с обводкой — только 0,35 (7 пациентов).")
    st.caption("Правка области: обведите участок мышью (лассо) и нажмите «Добавить» или «Убрать».")
    sel = event.selection if event else {}
    poly = polygon_mask(sel, image.shape[:2], scale) if sel else None
    c1, c2, c3, c4 = st.columns(4)
    if c1.button("Добавить обводку", disabled=poly is None, width="stretch"):
        ss.history.append(ss.mask.copy()); ss.mask = ss.mask | poly
        ss.edits.append({"действие": "добавить", "пикселей": int(poly.sum()), "время": datetime.now().isoformat(timespec="seconds")})
        st.rerun()
    if c2.button("Убрать обводку", disabled=poly is None, width="stretch"):
        ss.history.append(ss.mask.copy()); ss.mask = ss.mask & ~poly
        ss.edits.append({"действие": "убрать", "пикселей": int(poly.sum()), "время": datetime.now().isoformat(timespec="seconds")})
        st.rerun()
    if c3.button("Отменить правку", disabled=not ss.history, width="stretch"):
        ss.mask = ss.history.pop(); ss.edits.pop()
        st.rerun()
    if c4.button("Вернуть ответ системы", disabled=not ss.edits, width="stretch"):
        ss.mask, ss.edits, ss.history = r["mask"].copy(), [], []
        st.rerun()
    if edited:
        st.caption(f"Правок области: {len(ss.edits)}. Площадь: система {int(r['mask'].sum())} → после правки {int(mask.sum())} пикс.")

st.divider()
st.subheader("Проект заключения")
text_sys, _ = build(re_, name)
if edited:
    text_sys = text_sys.replace("Находки:", "Находки (область исправлена врачом, измерения — по исправленной области):")
basis = (key, mode, len(ss.edits), int(mask.sum()))
ta = f"ta_{key}"
if ss.text is None:
    ss.text, ss.text_basis, ss.text_sys_at = text_sys, basis, text_sys
if ss.text_basis != basis:
    if ss.text.strip() == (ss.text_sys_at or "").strip():  # врач текст не правил — проект обновляется сам
        ss.text, ss.text_basis, ss.text_sys_at = text_sys, basis, text_sys
        ss[ta] = text_sys
    else:
        st.warning("Режим МРТ или области изменились после правки текста. Нажмите «Пересобрать проект», чтобы обновить числа "
                   "(ваши правки текста будут заменены), или оставьте текст как есть.")
cols = st.columns([1, 1, 3])
if cols[0].button("Пересобрать проект", width="stretch"):
    ss.text, ss.text_basis, ss.text_sys_at = text_sys, basis, text_sys
    ss[ta] = text_sys
    st.rerun()
if ta not in ss:
    ss[ta] = ss.text
text = st.text_area("Текст можно править: врач видит снимок и текст одновременно.", height=430, key=ta)
ss.text = text

check_sys = verify(text_sys, re_, name)
check_now = verify(text, re_, name, edited=text.strip() != text_sys.strip())
if check_sys["ok"]:
    st.caption(f"Сверка проекта системы с результатом анализа: проверено {check_sys['checked']} чисел и утверждений, расхождений нет.")
else:
    st.error("Ошибка сверки проекта системы (ошибка кода): " + "; ".join(check_sys["problems"]))
if check_now["edited"]:
    if check_now["problems"]:
        st.warning("Текст изменен врачом. Не совпадает с результатом анализа: " + "; ".join(check_now["problems"])
                   + ". Если это сознательная правка — оставьте как есть.")
    else:
        st.caption("Текст изменен врачом; числа и утверждения по-прежнему совпадают с результатом анализа.")

if cols[1].button("Сохранить", type="primary", width="stretch"):
    folder, zipped = save_bundle(name, data, r, mask, text_sys, text, ss.edits, check_sys, check_now, overlay)
    ss.saved = (str(folder.relative_to(ROOT)), zipped)
if ss.get("saved"):
    cols[2].success(f"Сохранено: {ss.saved[0]}")
    cols[2].download_button("Скачать архив (снимок, области, заключение)", ss.saved[1], file_name=f"{Path(name).stem}_заключение.zip",
                            mime="application/zip")

with st.expander("На что смотрела модель типа (тепловая карта)"):
    st.warning("Карта показывает, какие участки сильнее всего повлияли на ответ модели типа, а не где опухоль. Если на "
               "снимке закрасить опухоль, ответ модели типа не меняется (проверка на 391 снимке, у всех трех копий): она "
               "опирается на общий вид снимка. Область опухоли выделяет модель контура.")
    if r["heatmap"] is not None:
        color = cv2.cvtColor(cv2.applyColorMap((r["heatmap"] * 255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
        st.image(cv2.addWeighted(np.repeat(image[..., None], 3, 2), 0.55, color, 0.45, 0), width=420,
                 caption=f"Тепловая карта для типа «{CLS_RU[r['heatmap_class']]}»")

with st.expander("Ограничения системы", expanded=qp.get("limits") == "1"):
    st.markdown(
        "- Анализируется один двумерный срез; другие срезы исследования не учитываются.\n"
        "- Различает три типа опухоли (глиома, менингиома, опухоль гипофиза) и их отсутствие; другие изменения "
        "(метастазы, шванномы, очаги рассеянного склероза) не распознает и может принять за опухоль.\n"
        f"- {UNKNOWN_WARNING}\n"
        "- Другие режимы МРТ проверены только на глиобластомах одной больницы: опухоль найдена на "
        + ", ".join(f"{pct(k / n)} срезов {MODE_RU[m]}" for m, (k, n) in MODE_FOUND.items())
        + " (на T1 с контрастом — 98%). Поэтому на других режимах «опухоли нет» не выдается как вывод, а тип помечается как ненадежный.\n"
        "- Ядро и зона изменений выделяются только у глиомы и ориентировочно: модель обучена на глиобластомах одного набора; "
        "на глиомах нашего набора совпадение с обводкой слабое (0,35).\n"
        "- Размеры — в пикселях: у JPG/PNG нет сведений о масштабе.\n"
        "- Сторона указывается по изображению, анатомическая сторона не определяется.\n"
        "- Пометка «проверьте» на проверке помогала находить ошибки на знакомых данных, но не на снимках из незнакомых больниц."
    )
