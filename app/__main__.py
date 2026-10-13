"""Запуск сервиса: python -m app"""

import logging
import os
import threading
import webbrowser
from pathlib import Path
from typing import IO

import uvicorn

from .config import Settings
from .web import create_app


def acquire_instance_lock(path: Path) -> IO | None:
    """Блокировка «сервис уже запущен» на файл в папке данных. None — её держит другая копия сервиса.

    Вторая копия (повторный запуск start.bat) иначе успела бы вернуть в очередь и взять в работу звонки
    первой и закрыться, не сумев занять порт, — а звонок остался бы «в расшифровке» навсегда.
    Блокировку снимает система, когда процесс завершается, в том числе аварийно.
    """
    f = open(path, "a+b")
    try:
        if os.name == "nt":
            import msvcrt

            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        return None
    return f


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = Settings.from_env()
    url = f"http://localhost:{settings.port}"
    settings.ensure_dirs()
    lock = acquire_instance_lock(settings.data_dir / "service.lock")
    if lock is None:
        print(f"\n  Сервис уже запущен в другом окне: {url}\n  Второй раз запускать не нужно — откройте адрес в браузере.\n")
        if os.environ.get("OPEN_BROWSER") == "1":
            webbrowser.open(url)
        return
    app = create_app(settings)
    print(f"\n  Сервис оценки звонков запущен: {url}\n  Не закрывайте это окно, пока работаете с сервисом.\n")
    if os.environ.get("OPEN_BROWSER") == "1":
        threading.Timer(3.0, webbrowser.open, [url]).start()
    try:
        uvicorn.run(app, host=settings.host, port=settings.port, log_level="warning")
    finally:
        lock.close()


if __name__ == "__main__":
    main()
