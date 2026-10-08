"""Запуск сервиса: python -m app"""

import logging
import os
import threading
import webbrowser

import uvicorn

from .config import Settings
from .web import create_app


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = Settings.from_env()
    app = create_app(settings)
    url = f"http://localhost:{settings.port}"
    print(f"\n  Сервис оценки звонков запущен: {url}\n  Не закрывайте это окно, пока работаете с сервисом.\n")
    if os.environ.get("OPEN_BROWSER") == "1":
        threading.Timer(3.0, webbrowser.open, [url]).start()
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="warning")


if __name__ == "__main__":
    main()
