"""Разбор имён файлов записей Mango Office.

Формат: «2026-10-07__12-19-24__79084642037__Регистратура_1.mp3» — дата, время, кто звонил, кому звонили.
Если первым идёт номер телефона — звонок входящий, если линия/сотрудник клиники — исходящий.
Время в именах файлов — по часовому поясу аккаунта Mango (обычно московское).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

_STAMP = re.compile(r"(\d{4})-(\d{2})-(\d{2})__(\d{2})-(\d{2})-(\d{2})__(.+)$")
_PHONE = r"\+?\d{10,13}"
_INCOMING = re.compile(rf"^({_PHONE})__(.+)$")
_OUTGOING = re.compile(rf"^(.+?)__({_PHONE})$")


@dataclass
class MangoFile:
    started_at: datetime
    direction: str | None  # in / out / None — не удалось понять
    phone: str | None
    line: str | None  # линия или сотрудник в Mango


def parse_mango_filename(filename: str, shift_hours: int = 0) -> MangoFile | None:
    m = _STAMP.search(Path(filename).stem)
    if not m:
        return None
    y, mo, d, hh, mm, ss, rest = m.groups()
    try:
        started = datetime(int(y), int(mo), int(d), int(hh), int(mm), int(ss)) + timedelta(hours=shift_hours)
    except ValueError:
        return None
    if inc := _INCOMING.match(rest):
        return MangoFile(started, "in", inc.group(1), inc.group(2).strip("_") or None)
    if out := _OUTGOING.match(rest):
        return MangoFile(started, "out", out.group(2), out.group(1).strip("_") or None)
    return MangoFile(started, None, None, None)
