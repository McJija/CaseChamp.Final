"""Запуск прототипа одной командой: поднимает экран на этом компьютере и открывает его в браузере.

Его вызывают скрипты start_windows.bat и start_mac_linux.sh; вручную:
    python app/launch.py
Экран доступен только с этого компьютера (адрес 127.0.0.1). Остановить — закрыть окно или Ctrl+C.
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def free_port(start: int = 8501) -> int:
    for port in range(start, start + 50):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise SystemExit("Не нашлось свободного порта 8501–8550.")


def system_runs() -> list[str]:
    """Шаблоны папок с весами версии финала (configs/system_run3.yaml) — те же, что читает src/system.py."""
    import yaml
    cfg = yaml.safe_load((ROOT / "configs" / "system_run3.yaml").read_text(encoding="utf-8"))
    pats = [c[k] for c in cfg["copies"] for k in ("classifier", "segmenter")]
    return pats + [cfg["core_edema"]["t1c"][0]]


def main() -> None:
    missing = [p for p in system_runs() if not any((d / "best.pt").exists() for d in (ROOT / "models").glob(p))]
    if missing:
        raise SystemExit("Нет весов моделей в папке models/: " + ", ".join(missing))
    try:  # тяжелые библиотеки — заранее: ошибку установки видно здесь, а не на странице
        import pytorch_grad_cam, segmentation_models_pytorch, timm, torch  # noqa: E401, F401
    except (ImportError, OSError) as e:
        hint = ("\nВозможная причина — слишком длинный путь к папке проекта ({} символов): Windows не открывает файлы "
                "с путем длиннее 260 символов. Перенесите папку ближе к корню диска (например, C:\\mri), удалите в ней "
                "папку .venv-app и запустите снова.".format(len(str(ROOT))) if os.name == "nt" and len(str(ROOT)) > 80 else "")
        raise SystemExit(f"Библиотеки установлены не полностью: {e}{hint}")
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    cmd = [sys.executable, "-m", "streamlit", "run", str(ROOT / "app" / "app.py"), "--server.headless", "true",
           "--server.address", "127.0.0.1", "--server.port", str(port), "--browser.gatherUsageStats", "false"]
    proc = subprocess.Popen(cmd, cwd=ROOT)
    for _ in range(240):
        if proc.poll() is not None:
            raise SystemExit("Прототип не запустился — подробности в сообщениях выше.")
        try:
            urllib.request.urlopen(url + "/_stcore/health", timeout=1)
            break
        except OSError:
            time.sleep(0.5)
    else:
        proc.terminate()
        raise SystemExit("Прототип не ответил за 2 минуты — подробности в сообщениях выше.")
    print(f"\nПрототип работает: {url}\nЕсли браузер не открылся, откройте этот адрес вручную. "
          "Чтобы остановить, закройте это окно или нажмите Ctrl+C.\n", flush=True)
    if os.environ.get("PROTOTYPE_NO_BROWSER") != "1":  # для автоматических проверок
        webbrowser.open(url)
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()


if __name__ == "__main__":
    main()
