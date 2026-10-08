#!/usr/bin/env sh
# Запуск на Linux / macOS
set -e
cd "$(dirname "$0")"
[ -x .venv/bin/python ] || python3 -m venv .venv
[ -f .env ] || cp .env.example .env
.venv/bin/python -m pip install --disable-pip-version-check -q -r requirements.txt
exec .venv/bin/python -m app
