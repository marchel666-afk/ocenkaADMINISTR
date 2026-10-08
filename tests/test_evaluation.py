import pytest

from app.checklists import load_checklists
from app.evaluation import Evaluation, Evaluator, Usage, extract_json, normalize_result
from app.llm import LLMError

from .conftest import ROOT, FakeLLM, sample_transcript


@pytest.fixture
def checklists():
    return load_checklists(ROOT / "config" / "checklists.yaml")


def test_extract_json_variants():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Вот результат: {"a": {"b": 2}} — готово') == {"a": {"b": 2}}
    with pytest.raises(ValueError):
        extract_json("нет json")


def test_normalize_result():
    assert normalize_result("Да") == "yes"
    assert normalize_result("partial") == "no"
    assert normalize_result(None) == "na"
    assert normalize_result(False) == "no"
    assert normalize_result("что-то") is None


def test_evaluate_full(checklists):
    llm = FakeLLM()
    ev = Evaluator(checklists, llm)
    usage = Usage()
    result = ev.evaluate(sample_transcript(), "incoming", "in", None, usage)

    expected = [c.id for c in checklists.general.evaluable] + [c.id for c in checklists.scenario("incoming").evaluable]
    assert list(result.verdicts) == expected
    assert result.verdicts["IN11"].result == "no"
    assert result.verdicts["G26"].result == "na"
    assert result.verdicts["IN04"].result == "yes"
    assert not result.warnings
    assert result.improvements[0].criterion_id == "IN11"
    assert usage.cost_usd == pytest.approx(0.001) and usage.prompt_tokens == 1000

    system_parts, user = llm.requests[0]
    assert len(system_parts) == 2
    assert "IN01 [" not in system_parts[1]  # неоцениваемые пункты в запрос не попадают
    assert "Администратор: Добрый день" in user

    # сериализация туда-обратно
    again = Evaluation.from_json(result.to_json())
    assert again.verdicts["IN11"].result == "no" and again.summary == result.summary


def test_evaluate_missing_and_retry(checklists):
    partial = '{"criteria": [{"id": "G01", "result": "да"}], "summary": "s"}'
    llm = FakeLLM(raw=["это не json", partial])
    result = Evaluator(checklists, llm).evaluate(sample_transcript(), "out_reminder", "out", None, Usage())
    assert len(llm.requests) == 2  # повтор после неразобранного ответа
    assert result.verdicts["G01"].result == "yes"
    assert result.verdicts["RM08"].result == "na"
    assert result.warnings and "RM08" in result.warnings[0]


def test_evaluate_bad_json_twice(checklists):
    llm = FakeLLM(raw=["ошибка", "снова ошибка"])
    with pytest.raises(LLMError):
        Evaluator(checklists, llm).evaluate(sample_transcript(), "incoming", "in", None, Usage())


def test_classify(checklists):
    llm = FakeLLM(call_type="out_reminder", admin_name="Мария")
    cls = Evaluator(checklists, llm).classify(sample_transcript(), "out", None, Usage())
    assert cls.call_type == "out_reminder" and cls.admin_name == "Мария"
    system = llm.requests[0][0][0]
    assert "out_reminder" in system and "incoming:" not in system  # для исходящих — только исходящие сценарии


def test_classify_admin_speaker(checklists):
    llm = FakeLLM(raw=['{"call_type": "incoming", "admin_name": "Анна", "admin_speaker": 2, "reason": "т"}'])
    cls = Evaluator(checklists, llm).classify(sample_transcript(), "in", None, Usage())
    assert cls.admin_speaker == "s2"


def test_classify_rejects_unknown(checklists):
    llm = FakeLLM(call_type="incoming")
    with pytest.raises(LLMError):
        Evaluator(checklists, llm).classify(sample_transcript(), "out", None, Usage())


def test_call_facts_roles_note():
    from app.evaluation import _call_facts

    t = sample_transcript()
    t.stereo = False
    assert "по голосу" in _call_facts(t, "in", None)
    t.stereo = True
    assert "стерео" in _call_facts(t, "in", None)
