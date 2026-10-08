from app.checklists import load_checklists
from app.scoring import NA, NO, YES, weighted_score

from .conftest import ROOT


def test_checklists_load():
    cl = load_checklists(ROOT / "config" / "checklists.yaml")
    assert len(cl.general.criteria) == 27
    assert set(cl.scenarios) == {"incoming", "out_request", "out_online", "out_reminder", "out_reschedule", "out_noshow"}
    assert cl.scenarios["incoming"].direction == "in"
    assert all(s.direction == "out" for k, s in cl.scenarios.items() if k != "incoming")
    # пункты, которые нельзя оценить по записи, помечены
    skipped = {c.id for c in cl.general.criteria if not c.evaluable}
    assert skipped == {"G09", "G19", "G20"}
    assert not cl.criterion("IN01").evaluable  # «поднимает трубку за 2-3 звонка» — только из данных телефонии
    assert cl.criterion("IN04").weight == 3.0 and cl.criterion("IN17").weight == 1.0
    assert "Варикоза нет" in cl.clinic_context


def test_weighted_score():
    cl = load_checklists(ROOT / "config" / "checklists.yaml")
    crit = [cl.criterion(i) for i in ("IN04", "IN05", "IN17", "IN01")]  # веса 3, 2, 1 и неоцениваемый
    assert weighted_score(crit, {"IN04": YES, "IN05": YES, "IN17": YES}) == 100.0
    assert weighted_score(crit, {"IN04": YES, "IN05": NO, "IN17": YES}) == round(100 * 4 / 6, 1)
    # «не применимо» не влияет на процент
    assert weighted_score(crit, {"IN04": YES, "IN05": NA, "IN17": NO}) == 75.0
    # неоцениваемый пункт игнорируется, даже если результат есть
    assert weighted_score(crit, {"IN04": NO, "IN01": YES}) == 0.0
    assert weighted_score(crit, {}) is None
