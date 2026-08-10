"""T23 · UC-1 레벨 측정 (H2).

이 카드의 값은 셋이다. ① 프리셋이 **문서와 어긋나지 않는가**(마커를 다시 지으면
"행동 기준만"이라는 첫 원칙이 조용히 무너진다) ② "해당 없음"이 **레벨 0이 아니라
커버리지에서 빠지는가** ③ 사전 채움이 **임베딩 API를 오프라인에서 안 타는가**.
"""

from __future__ import annotations

from datetime import date

import pytest
import yaml

from app.hitl import normalize_prompt
from contracts.enums import Category, Importance, Level
from contracts.models import CompetencyRecord, ProfileJSON
from nodes.level_survey import (
    AXES_PATH,
    CATEGORIES_PATH,
    EVIDENCE_SUFFIX,
    KIND,
    bind_axes,
    build_payload,
    build_profile,
    collect_evidence,
    key_of,
    level_survey,
    load_survey,
    score,
)


@pytest.fixture(scope="module")
def survey():
    return load_survey()


def _owned(comp_id: str, name: str, category: Category, level: Level | None = None):
    return CompetencyRecord(
        comp_id=comp_id,
        category=category,
        name=name,
        importance=Importance.REQUIRED,
        level=level,
    )


# --------------------------------------------------------------------------
# 프리셋 — 문서가 정본이고 이 파일은 그 사본이다
# --------------------------------------------------------------------------
def test_대분류_다섯_개를_순서대로_읽는다(survey):
    assert [category for category, _ in survey.categories] == [
        Category.D1_SW_FOUNDATION,
        Category.D2_BACKEND,
        Category.D3_CLOUD_INFRA,
        Category.D4_ORCHESTRATION,
        Category.D5_AI_INFRA,
    ]


def test_인재상은_레벨_측정_대상이_아니다(survey):
    """4단계 사다리는 '무엇을 해봤나'의 깊이 좌표인데 인재상은 그런 축이 아니다."""
    assert Category.C_CULTURE_FIT not in {c for c, _ in survey.categories}
    assert all(axis.category is not Category.C_CULTURE_FIT for axis in survey.axes)


def test_문항_수가_문서의_범위_안이다(survey):
    """질문세트 §3 — 대분류 5개 × 축 5~6개 ≈ 25~30문항."""
    assert 25 <= len(survey.axes) <= 30
    for category, _ in survey.categories:
        assert 5 <= len(survey.axes_of(category)) <= 6


def test_모든_축이_네_레벨_마커를_갖는다(survey):
    for axis in survey.axes:
        assert set(axis.markers) == {
            Level.LEARNED,
            Level.USED,
            Level.OPERATED,
            Level.LED,
        }, axis.axis_id


def test_축_id가_유일하다(survey):
    ids = [axis.axis_id for axis in survey.axes]
    assert len(ids) == len(set(ids))


def test_같은_축_안에서_마커가_서로_다르다(survey):
    """레벨이 달라도 문구가 같으면 사용자가 무엇을 고르는지 알 수 없다."""
    for axis in survey.axes:
        markers = list(axis.markers.values())
        assert len(markers) == len(set(markers)), axis.axis_id


def test_마커가_자기평가_척도가_아니다(survey):
    """질문세트 §1 첫 번째 원칙 — '얼마나 잘하냐(1~5)' 금지, 행동 마커만.

    부분 문자열로 한글을 거르면 "취약**점**" 같은 멀쩡한 말이 걸린다. 자기평가로
    읽히는 표현만 정확히 짚는다.
    """
    import re

    banned_words = ("잘함", "보통 수준", "매우 잘", "상급", "중급", "초급", "숙련도", "자신있")
    scale = re.compile(r"\d\s*(점|단계|~\s*\d)")

    for axis in survey.axes:
        for marker in axis.markers.values():
            assert not any(word in marker for word in banned_words), (axis.axis_id, marker)
            assert not scale.search(marker), (axis.axis_id, marker)


def test_프리셋_오타는_로드에서_죽는다(tmp_path, survey):
    """조용히 넘어가면 그 축만 폼에서 사라져 커버리지가 틀린다."""
    raw = yaml.safe_load(AXES_PATH.read_text(encoding="utf-8"))
    raw["axes"][0]["category"] = "D9_없는대분류"
    broken = tmp_path / "axes.yaml"
    broken.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")

    with pytest.raises(ValueError):
        load_survey(broken, CATEGORIES_PATH)


def test_대분류가_프리셋에_없으면_죽는다(tmp_path):
    raw = yaml.safe_load(CATEGORIES_PATH.read_text(encoding="utf-8"))
    raw["categories"] = [c for c in raw["categories"] if c["id"] != "D5_AI인프라"]
    trimmed = tmp_path / "categories.yaml"
    trimmed.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")

    with pytest.raises(ValueError):
        load_survey(AXES_PATH, trimmed)


# --------------------------------------------------------------------------
# 사전 채움 — 임베딩은 주입점 뒤에 있어야 한다
# --------------------------------------------------------------------------
def test_보유_역량이_없으면_임베딩을_아예_안_부른다(survey):
    def boom(*args, **kwargs):
        raise AssertionError("보유 역량이 없는데 임베딩을 불렀다")

    assert bind_axes(survey, [], matcher=boom) == {}


def test_사전_채움은_유사도_1순위를_그_축의_주인으로_삼는다(survey):
    owned = [
        _owned("pf-01", "Kubernetes 클러스터 운영", Category.D4_ORCHESTRATION, Level.OPERATED),
        _owned("pf-02", "Docker 이미지 관리", Category.D4_ORCHESTRATION, Level.USED),
    ]
    calls: list[int] = []

    def matcher(axis_records, owned_records, top_k, *, embed=None):
        calls.append(top_k)
        return [("d4-02", "pf-01"), ("d4-02", "pf-02"), ("d4-01", "pf-02")]

    bindings = bind_axes(survey, owned, matcher=matcher)

    assert bindings["d4-02"].comp_id == "pf-01", "먼저 온 후보가 주인이어야 한다"
    assert bindings["d4-01"].comp_id == "pf-02"
    assert calls == [1], "축 하나당 후보를 여럿 받으면 역량명 병합의 빌미가 된다"


def test_모르는_역량_id는_무시한다(survey):
    owned = [_owned("pf-01", "쿠버네티스", Category.D4_ORCHESTRATION)]

    bindings = bind_axes(
        survey, owned, matcher=lambda *a, **k: [("d4-02", "없는id")]
    )

    assert bindings == {}


def test_묶인_축은_역량_id를_키로_쓴다(survey):
    """D06 — level_coordinates는 보유 역량 id로 키를 잡는다."""
    axis = next(a for a in survey.axes if a.axis_id == "d4-02")
    bound = {"d4-02": _owned("pf-01", "쿠버네티스", Category.D4_ORCHESTRATION)}

    assert key_of(axis, bound) == "pf-01"
    assert key_of(axis, {}) == "d4-02", "안 묶인 축은 축 id가 그대로 키다"


# --------------------------------------------------------------------------
# 폼 페이로드
# --------------------------------------------------------------------------
def test_축마다_라디오_하나를_만든다(survey):
    payload = build_payload(survey, {})

    assert payload["kind"] == KIND
    radios = [q for q in payload["questions"] if q.get("options")]
    assert len(radios) == len(survey.axes)


def test_선택지는_마커_넷_더하기_해당없음이다(survey):
    payload = build_payload(survey, {})
    question = next(q for q in payload["questions"] if q["question_id"] == "d4-02")
    axis = next(a for a in survey.axes if a.axis_id == "d4-02")

    assert question["options"] == [
        axis.markers[Level.LEARNED],
        axis.markers[Level.USED],
        axis.markers[Level.OPERATED],
        axis.markers[Level.LED],
        survey.none_label,
    ]


def test_사전_채움이_없으면_아무것도_안_고른다(survey):
    payload = build_payload(survey, {})
    assert all(q.get("default") is None for q in payload["questions"])


def test_추출된_레벨이_미리_선택된다(survey):
    bindings = {
        "d4-02": _owned("pf-01", "쿠버네티스", Category.D4_ORCHESTRATION, Level.OPERATED)
    }
    axis = next(a for a in survey.axes if a.axis_id == "d4-02")

    payload = build_payload(survey, bindings)
    question = next(q for q in payload["questions"] if q["question_id"] == "d4-02")

    assert question["default"] == axis.markers[Level.OPERATED]


def test_레벨을_모르는_역량은_미리_고르지_않는다(survey):
    """없는 근거로 미리 골라 두면 손대지 않은 항목이 '답한 것'으로 넘어간다(D28)."""
    bindings = {"d4-02": _owned("pf-01", "쿠버네티스", Category.D4_ORCHESTRATION, None)}

    payload = build_payload(survey, bindings)
    question = next(q for q in payload["questions"] if q["question_id"] == "d4-02")

    assert question.get("default") is None


@pytest.mark.parametrize("level", [Level.OPERATED, Level.LED])
def test_높은_레벨_사전채움에는_근거란이_붙는다(survey, level):
    bindings = {"d4-02": _owned("pf-01", "쿠버네티스", Category.D4_ORCHESTRATION, level)}

    payload = build_payload(survey, bindings)
    keys = [q["question_id"] for q in payload["questions"]]

    assert "d4-02" + EVIDENCE_SUFFIX in keys


@pytest.mark.parametrize("level", [Level.LEARNED, Level.USED, None])
def test_낮은_레벨에는_근거란을_안_붙인다(survey, level):
    """26문항에 근거란까지 전부 붙이면 폼이 두 배가 된다 — 값은 위쪽 두 관문에 있다."""
    bindings = {"d4-02": _owned("pf-01", "쿠버네티스", Category.D4_ORCHESTRATION, level)}

    payload = build_payload(survey, bindings)
    keys = [q["question_id"] for q in payload["questions"]]

    assert "d4-02" + EVIDENCE_SUFFIX not in keys


def test_섹션은_대분류_라벨이고_순서를_지킨다(survey):
    payload = build_payload(survey, {})
    sections = []
    for question in payload["questions"]:
        if question["section"] not in sections:
            sections.append(question["section"])

    assert sections == [label for _, label in survey.categories]


def test_대분류_enum_값이_화면에_새지_않는다(survey):
    payload = build_payload(survey, {})
    for question in payload["questions"]:
        assert not question["section"].startswith("D"), question["section"]


def test_hitl이_사전채움과_섹션을_보존한다(survey):
    """`app/hitl.py`가 새 필드를 흘리면 화면에서 조용히 사라진다 — 폼은 그려지는데
    미리 고른 것도, 섹션 구분도 없어진다."""
    bindings = {
        "d4-02": _owned("pf-01", "쿠버네티스", Category.D4_ORCHESTRATION, Level.OPERATED)
    }
    axis = next(a for a in survey.axes if a.axis_id == "d4-02")

    prompt = normalize_prompt(build_payload(survey, bindings))
    question = next(q for q in prompt.questions if q.key == "d4-02")

    assert question.default == axis.markers[Level.OPERATED]
    assert question.section == "컨테이너·오케스트레이션·DevOps/SRE"
    assert not question.multi


def test_페이로드가_hitl_폼으로_정규화된다(survey):
    """T12의 재개 루프를 그대로 재사용한다 — 새 HITL 메커니즘을 만들지 않았다."""
    prompt = normalize_prompt(build_payload(survey, {}))

    assert prompt.kind == KIND
    assert len(prompt.questions) == len(survey.axes)
    first = prompt.questions[0]
    assert first.section == survey.categories[0][1]
    assert first.options


# --------------------------------------------------------------------------
# 위젯 렌더 — 페이로드가 맞아도 그리는 쪽이 무시하면 화면에서 조용히 사라진다
# --------------------------------------------------------------------------
class FakeStreamlit:
    """`app/hitl.py`가 부르는 것만 흉내낸다. 위젯 인자를 그대로 모은다."""

    def __init__(self, submitted: bool = True):
        self.radio_calls: list[dict] = []
        self.markdown_calls: list[str] = []
        self._submitted = submitted

    # 폼 컨텍스트
    def form(self, key):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def subheader(self, text):
        pass

    def caption(self, text):
        pass

    def markdown(self, text):
        self.markdown_calls.append(text)

    def form_submit_button(self, label, type=None):
        return self._submitted

    def radio(self, label, *, options, index, key):
        self.radio_calls.append({"label": label, "options": options, "index": index})
        return options[index] if index is not None else None

    def multiselect(self, label, *, options, default, key, placeholder):
        return []

    def text_area(self, label, *, key, height, placeholder):
        return ""


@pytest.fixture
def fake_st(monkeypatch):
    import app.hitl as hitl_module

    fake = FakeStreamlit()
    monkeypatch.setattr(hitl_module, "st", fake)
    return fake


def test_사전채움이_라디오에서_실제로_선택된다(survey, fake_st):
    """페이로드에 default가 실려도 위젯이 index=None으로 그리면 아무 소용이 없다."""
    from app.hitl import render_prompt_form

    bindings = {
        "d4-02": _owned("pf-01", "쿠버네티스", Category.D4_ORCHESTRATION, Level.OPERATED)
    }
    axis = next(a for a in survey.axes if a.axis_id == "d4-02")
    prompt = normalize_prompt(build_payload(survey, bindings))

    answers = render_prompt_form(prompt, form_key="t23")

    picked = next(c for c in fake_st.radio_calls if c["label"] == axis.name)
    assert picked["index"] == picked["options"].index(axis.markers[Level.OPERATED])
    assert answers["d4-02"] == axis.markers[Level.OPERATED], "제출값에 안 실렸다"


def test_사전채움이_없는_축은_아무것도_안_고른_채로_뜬다(survey, fake_st):
    from app.hitl import render_prompt_form

    prompt = normalize_prompt(build_payload(survey, {}))

    answers = render_prompt_form(prompt, form_key="t23")

    assert all(call["index"] is None for call in fake_st.radio_calls)
    assert answers == {}, "아무것도 안 골랐는데 답변이 실렸다"


def test_섹션_제목이_대분류마다_한_번씩_찍힌다(survey, fake_st):
    from app.hitl import render_prompt_form

    render_prompt_form(normalize_prompt(build_payload(survey, {})), form_key="t23")

    assert fake_st.markdown_calls == [f"**{label}**" for _, label in survey.categories]


# --------------------------------------------------------------------------
# 채점 — "해당 없음"은 레벨 0이 아니다
# --------------------------------------------------------------------------
def _answer_all(survey, level: Level) -> dict[str, str]:
    return {axis.axis_id: axis.markers[level] for axis in survey.axes}


def test_전부_응답하면_커버리지가_1이다(survey):
    coordinates, coverage = score(survey, _answer_all(survey, Level.USED), {})

    assert len(coordinates) == len(survey.axes)
    assert set(coverage.values()) == {1.0}


def test_해당_없음은_좌표에_안_들어가고_커버리지에서_빠진다(survey):
    answers = _answer_all(survey, Level.USED)
    d5_axes = survey.axes_of(Category.D5_AI_INFRA)
    for axis in d5_axes:
        answers[axis.axis_id] = survey.none_label

    coordinates, coverage = score(survey, answers, {})

    assert all(axis.axis_id not in coordinates for axis in d5_axes)
    assert coverage[Category.D5_AI_INFRA] == 0.0
    assert coverage[Category.D1_SW_FOUNDATION] == 1.0


def test_무응답은_해당_없음과_같다(survey):
    coordinates, coverage = score(survey, {}, {})

    assert coordinates == {}
    assert set(coverage.values()) == {0.0}


def test_커버리지는_대분류별_비율이다(survey):
    d2 = survey.axes_of(Category.D2_BACKEND)
    answers = {d2[0].axis_id: d2[0].markers[Level.LED]}

    _, coverage = score(survey, answers, {})

    assert coverage[Category.D2_BACKEND] == pytest.approx(1 / len(d2))


def test_모르는_문구는_응답으로_안_센다(survey):
    """라디오 밖의 값이 흘러들어와도 좌표를 오염시키지 않는다."""
    coordinates, coverage = score(survey, {"d1-01": "아무 말"}, {})

    assert coordinates == {}
    assert coverage[Category.D1_SW_FOUNDATION] == 0.0


def test_묶인_축의_좌표는_역량_id로_기록된다(survey):
    bindings = {"d4-02": _owned("pf-01", "쿠버네티스 운영", Category.D4_ORCHESTRATION)}
    axis = next(a for a in survey.axes if a.axis_id == "d4-02")

    coordinates, _ = score(survey, {"d4-02": axis.markers[Level.LED]}, bindings)

    assert coordinates == {"pf-01": Level.LED}
    assert "d4-02" not in coordinates


# --------------------------------------------------------------------------
# 근거
# --------------------------------------------------------------------------
def test_근거_한_줄을_모은다(survey):
    answers = {
        "d4-02" + EVIDENCE_SUFFIX: "사내 배포 플랫폼에서 클러스터를 운영했습니다",
        "d1-01" + EVIDENCE_SUFFIX: "   ",
    }

    assert collect_evidence(survey, answers) == {
        "d4-02": "사내 배포 플랫폼에서 클러스터를 운영했습니다"
    }


def test_근거는_역량의_evidence로_붙는다(survey):
    axis = next(a for a in survey.axes if a.axis_id == "d4-02")
    answers = {
        "d4-02": axis.markers[Level.OPERATED],
        "d4-02" + EVIDENCE_SUFFIX: "장애 대응과 튜닝을 직접 했습니다",
    }

    profile = build_profile(survey, answers, {}, [], built_at=date(2026, 8, 11))

    record = next(c for c in profile.competencies if c.comp_id == "d4-02")
    assert [e.quote for e in record.evidence] == ["장애 대응과 튜닝을 직접 했습니다"]


# --------------------------------------------------------------------------
# 완료 조건 — 폼 제출 → ProfileJSON이 채워진다
# --------------------------------------------------------------------------
def test_제출하면_ProfileJSON이_채워진다(survey):
    profile = build_profile(
        survey, _answer_all(survey, Level.USED), {}, [], built_at=date(2026, 8, 11)
    )

    assert isinstance(profile, ProfileJSON)
    assert len(profile.level_coordinates) == len(survey.axes)
    assert set(profile.coverage.values()) == {1.0}
    assert profile.built_at == date(2026, 8, 11)
    # 계약대로 직렬화·역직렬화가 된다
    assert ProfileJSON.model_validate_json(profile.model_dump_json()) == profile


def test_설문이_새로_발견한_축은_역량으로_추가된다(survey):
    """폭 보완 — 이력서에 안 적혔지만 해봤을 수 있는 것을 줍는다(질문세트 §1)."""
    axis = next(a for a in survey.axes if a.axis_id == "d5-01")

    profile = build_profile(survey, {"d5-01": axis.markers[Level.USED]}, {}, [])

    record = next(c for c in profile.competencies if c.comp_id == "d5-01")
    assert record.name == axis.name
    assert record.category is Category.D5_AI_INFRA
    assert record.level is Level.USED


def test_추출된_역량은_확정_레벨로_갱신된다(survey):
    axis = next(a for a in survey.axes if a.axis_id == "d4-02")
    owned = [_owned("pf-01", "쿠버네티스 운영", Category.D4_ORCHESTRATION, Level.USED)]
    bindings = {"d4-02": owned[0]}

    profile = build_profile(survey, {"d4-02": axis.markers[Level.LED]}, bindings, owned)

    record = next(c for c in profile.competencies if c.comp_id == "pf-01")
    assert record.level is Level.LED, "좌표와 레코드가 어긋나면 어느 쪽이 맞는지 모른다"
    assert record.name == "쿠버네티스 운영", "원문 역량명은 축 이름으로 덮이지 않는다"
    assert profile.level_coordinates["pf-01"] is Level.LED


def test_역량_id가_중복되지_않는다(survey):
    owned = [_owned("pf-01", "쿠버네티스 운영", Category.D4_ORCHESTRATION, Level.USED)]
    bindings = {"d4-02": owned[0]}

    profile = build_profile(survey, _answer_all(survey, Level.USED), bindings, owned)

    ids = [c.comp_id for c in profile.competencies]
    assert len(ids) == len(set(ids))
    assert "d4-02" not in ids, "묶인 축이 새 역량으로 또 생기면 안 된다"


def test_해당_없음만_고른_축은_역량으로_안_생긴다(survey):
    profile = build_profile(survey, {}, {}, [])

    assert profile.competencies == []
    assert profile.level_coordinates == {}


# --------------------------------------------------------------------------
# 노드 — interrupt 앞은 순수 계산이어야 한다
# --------------------------------------------------------------------------
def test_노드가_중단하고_재개하면_프로필을_돌려준다(survey, monkeypatch):
    import nodes.level_survey as module

    captured: list[dict] = []

    def fake_interrupt(payload):
        captured.append(payload)
        return {survey.axes[0].axis_id: survey.axes[0].markers[Level.OPERATED]}

    monkeypatch.setattr(module, "interrupt", fake_interrupt)

    result = level_survey({"required": []})

    assert captured and captured[0]["kind"] == KIND
    profile = result["profile"]
    assert isinstance(profile, ProfileJSON)
    assert profile.level_coordinates == {survey.axes[0].axis_id: Level.OPERATED}


def test_노드가_보유_역량을_required에서_읽는다(survey, monkeypatch):
    """`GraphState`에 역량 목록을 담는 칸이 그것뿐이다 — 다른 칸을 보면 사전 채움이
    통째로 죽는데, 폼은 멀쩡히 떠서 아무도 모른다."""
    import nodes.level_survey as module

    owned = [
        _owned("pf-01", "쿠버네티스 운영", Category.D4_ORCHESTRATION, Level.OPERATED)
    ]
    seen: list[list] = []

    def matcher(axis_records, owned_records, top_k, *, embed=None):
        seen.append(owned_records)
        return [("d4-02", "pf-01")]

    monkeypatch.setattr(module, "retrieve_candidates", matcher)
    monkeypatch.setattr(module, "interrupt", lambda payload: {})

    level_survey({"required": owned})

    assert seen == [owned], "보유 역량이 사전 채움까지 안 갔다"


def test_기존_근거를_지우지_않는다(survey):
    """추출 단계가 붙여 둔 인용을 설문이 덮으면 UC-2가 근거를 잃는다."""
    from datetime import date as _date

    from contracts.models import Evidence

    axis = next(a for a in survey.axes if a.axis_id == "d4-02")
    original = _owned("pf-01", "쿠버네티스 운영", Category.D4_ORCHESTRATION, Level.USED)
    original = original.model_copy(
        update={
            "evidence": [
                Evidence(
                    source_name="이력서",
                    quote="EKS 클러스터를 운영했다",
                    collected_at=_date(2026, 8, 1),
                )
            ]
        }
    )
    answers = {
        "d4-02": axis.markers[Level.LED],
        "d4-02" + EVIDENCE_SUFFIX: "표준을 제가 정했습니다",
    }

    profile = build_profile(survey, answers, {"d4-02": original}, [original])

    record = next(c for c in profile.competencies if c.comp_id == "pf-01")
    assert [e.quote for e in record.evidence] == [
        "EKS 클러스터를 운영했다",
        "표준을 제가 정했습니다",
    ]


def test_노드는_부분_상태만_돌려준다(survey, monkeypatch):
    import nodes.level_survey as module

    monkeypatch.setattr(module, "interrupt", lambda payload: {})

    result = level_survey({"required": [], "company": "카카오"})

    assert set(result) == {"profile"}, "노드는 전체 상태를 반환하지 않는다"


def test_노드가_보유_역량_없이도_임베딩을_안_탄다(monkeypatch):
    """오프라인 테스트가 조용히 과금되는 것을 구조로 막는다(D66)."""
    import nodes.level_survey as module

    def boom(*args, **kwargs):
        raise AssertionError("임베딩 API를 탔다")

    monkeypatch.setattr(module, "retrieve_candidates", boom)
    monkeypatch.setattr(module, "interrupt", lambda payload: {})

    assert "profile" in level_survey({"required": []})
