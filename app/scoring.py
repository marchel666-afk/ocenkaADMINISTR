"""Расчёт средневзвешенной оценки звонка."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from .checklists import Criterion

YES, NO, NA = "yes", "no", "na"
RESULT_LABELS = {YES: "Да", NO: "Нет", NA: "Не применимо"}


def weighted_score(criteria: Iterable[Criterion], results: Mapping[str, str]) -> float | None:
    """Процент выполнения: сумма весов пунктов «Да» / сумма весов применимых пунктов.

    Пункты «не применимо» и неоцениваемые по записи в расчёт не входят.
    Возвращает None, если применимых пунктов нет.
    """
    earned = 0.0
    possible = 0.0
    for c in criteria:
        if not c.evaluable:
            continue
        result = results.get(c.id, NA)
        if result == YES:
            earned += c.weight
            possible += c.weight
        elif result == NO:
            possible += c.weight
    if possible == 0:
        return None
    return round(100.0 * earned / possible, 1)
