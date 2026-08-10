"""UC-1 레벨 측정 (H2) — 추출된 역량 원자에 깊이 좌표를 찍는다 (T23).

`docs/UC1_레벨측정_질문세트.md`의 구현이다. 두 가지를 동시에 한다:
**깊이 보정**(이력서의 "K8s 경험"이 튜토리얼인지 운영인지 확정)과 **폭 보완**(이력서에
안 적혔지만 해봤을 수 있는 축을 프리셋으로 훑어 줍기).

세 가지 원칙이 이 파일 전체를 지배한다(질문세트 §1):

1. **행동 기준만.** 라디오 옵션은 "얼마나 잘하냐"가 아니라 **무엇을 해봤냐**이며,
   문구는 `presets/level_questions.yaml`의 마커를 **그대로** 쓴다. 여기서 문구를
   다시 지으면 원칙이 조용히 무너지므로 이 파일에는 마커 문자열이 하나도 없다.
2. **사전 채움.** 이미 추출된 역량은 시스템이 레벨을 미리 골라 두고 사용자는 확인만 한다.
3. **"해당 없음"도 정보다.** 레벨 0이 아니라 **커버리지에서 빠지는** 것으로 다룬다.

키 규약 — `level_coordinates`는 **역량 id로 키를 잡는다**
------------------------------------------------------
질문세트 §6은 "역량명"이라고 적었지만, `fixtures/profile_sample.json`과 **D06이 이미
`pf-01` 같은 보유 역량 id를 전제로 결정을 내려 뒀다.** 골든 픽스처와 기록된 결정이
정본이라 그쪽을 따른다. 축이 추출 역량과 묶이면 **그 역량의 comp_id**를, 안 묶이면
**축 id**(`d4-02`)를 키로 쓴다 — 후자는 설문이 새로 발견한 역량이라 id가 없기 때문이다.

상태 계약에 설문 전용 칸이 없다
------------------------------
`GraphState`에는 `level_coordinates`도 `coverage`도 담을 자리가 없고 R1이라 못 늘린다.
카드의 완료 조건이 "**ProfileJSON**의 …가 채워짐"이므로 `profile` 필드로 돌려준다 —
계약 안에서 유일하게 맞는 자리다. T18이 `GateStatus`에 칸이 없어 `missing` 라벨을
재사용한 것과 같은 판단이다(D59).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml
from langgraph.types import interrupt

from contracts.enums import Category, Importance, Level
from contracts.models import CompetencyRecord, Evidence, ProfileJSON
from contracts.state import GraphState
from tools.retrieve import retrieve_candidates

PRESETS = Path(__file__).resolve().parent.parent / "presets"
AXES_PATH = PRESETS / "level_questions.yaml"
CATEGORIES_PATH = PRESETS / "categories.yaml"

KIND = "level_survey"

# 근거 한 줄을 받을 레벨. 질문세트 §3 — 여기서 받아 두면 UC-2 델타 인터뷰가 짧아진다.
# 부풀려지기 쉬운 두 관문("써봄→실무운영", "실무운영→설계주도")의 위쪽이기도 하다(§2).
EVIDENCE_LEVELS = frozenset({Level.OPERATED, Level.LED})

EVIDENCE_SUFFIX = "__evidence"
EVIDENCE_SOURCE = "레벨 측정(UC-1)"

# 축 하나당 후보 1건만 본다 — 가장 가까운 역량이 그 축의 주인이다. 여럿을 받아
# 병합하면 "역량명 원문 보존"(§7)이 깨진다.
TOP_K = 1


@dataclass(frozen=True)
class Axis:
    """물어볼 역량 축 하나."""

    axis_id: str
    category: Category
    name: str
    markers: dict[Level, str]     # 레벨 → 행동 마커(라디오 옵션 텍스트)

    def option_labels(self, order: list[Level]) -> list[str]:
        return [self.markers[level] for level in order if level in self.markers]

    def level_of(self, label: str) -> Level | None:
        """고른 마커 문구를 레벨로 되돌린다. 모르는 문구면 None."""
        for level, marker in self.markers.items():
            if marker == label:
                return level
        return None

    def marker_of(self, level: Level | None) -> str | None:
        return self.markers.get(level) if level is not None else None


@dataclass(frozen=True)
class Survey:
    axes: list[Axis]
    order: list[Level]
    none_label: str
    categories: list[tuple[Category, str]]   # 폼 섹션 순서대로

    def axes_of(self, category: Category) -> list[Axis]:
        return [axis for axis in self.axes if axis.category is category]


# --- 프리셋 로드 -------------------------------------------------------------


def load_survey(
    axes_path: Path | str = AXES_PATH,
    categories_path: Path | str = CATEGORIES_PATH,
) -> Survey:
    """프리셋 두 장을 읽어 설문을 조립한다.

    **enum에 없는 값은 여기서 죽는다.** 프리셋은 사람이 손으로 고치는 파일이라
    오타가 나기 쉬운데, 조용히 넘어가면 그 축만 폼에서 사라져 커버리지가 틀린다.
    """
    axes_raw = yaml.safe_load(Path(axes_path).read_text(encoding="utf-8"))
    categories_raw = yaml.safe_load(Path(categories_path).read_text(encoding="utf-8"))

    categories = [
        (Category(entry["id"]), str(entry["label"]))
        for entry in sorted(categories_raw["categories"], key=lambda e: e["order"])
    ]
    known = {category for category, _ in categories}

    order = [Level(value) for value in axes_raw["order"]]

    axes: list[Axis] = []
    for entry in axes_raw["axes"]:
        category = Category(entry["category"])
        if category not in known:
            raise ValueError(
                f"축 {entry['id']!r}의 대분류 {category.value!r}가 "
                f"{Path(categories_path).name}에 없다"
            )
        axes.append(
            Axis(
                axis_id=str(entry["id"]),
                category=category,
                name=str(entry["name"]),
                markers={Level(k): str(v) for k, v in entry["markers"].items()},
            )
        )

    return Survey(
        axes=axes,
        order=order,
        none_label=str(axes_raw["none_label"]),
        categories=categories,
    )


# --- 사전 채움 ---------------------------------------------------------------


def bind_axes(
    survey: Survey,
    owned: list[CompetencyRecord],
    *,
    matcher=None,
    embed=None,
) -> dict[str, CompetencyRecord]:
    """축 → 그 축을 대표하는 추출 역량. 임베딩 유사도로 잇는다(T14 재사용).

    `matcher`는 주입점이다 — 오프라인 테스트가 임베딩 API를 타지 않게 하는 유일한
    통로이며, 없으면 이 노드를 부르는 모든 테스트가 조용히 과금된다(D66의 반복 방지).

    **기본값을 `matcher=retrieve_candidates`로 박지 않는 이유**가 여기 있다. 기본 인자는
    함수 정의 시점에 묶이므로 그렇게 쓰면 `monkeypatch.setattr(module, "retrieve_candidates", …)`가
    **먹지 않는다** — 주입점이 있는데도 테스트가 실 API를 탄다. 실제로 그렇게 만들었다가
    잡혔다(D71). 호출 시점에 모듈 전역을 본다.

    보유 역량이 없으면 **호출 자체를 하지 않는다** — `retrieve_candidates`도 빈 입력에
    같은 계약이지만, 여기서 먼저 끊어 두면 "왜 안 불렀나"가 이 파일에서 읽힌다.
    """
    if not owned:
        return {}

    matcher = matcher or retrieve_candidates

    axis_records = [
        CompetencyRecord(
            comp_id=axis.axis_id,
            category=axis.category,
            name=axis.name,
            importance=Importance.REQUIRED,
        )
        for axis in survey.axes
    ]

    pairs = matcher(axis_records, owned, TOP_K, embed=embed)

    by_id = {record.comp_id: record for record in owned}
    bindings: dict[str, CompetencyRecord] = {}
    for axis_id, comp_id in pairs:
        # 유사도 내림차순이므로 먼저 온 것이 그 축의 주인이다.
        if axis_id not in bindings and comp_id in by_id:
            bindings[axis_id] = by_id[comp_id]
    return bindings


def key_of(axis: Axis, bindings: dict[str, CompetencyRecord]) -> str:
    """이 축이 `level_coordinates`에서 가질 키."""
    bound = bindings.get(axis.axis_id)
    return bound.comp_id if bound is not None else axis.axis_id


# --- 폼 페이로드 -------------------------------------------------------------


def build_payload(survey: Survey, bindings: dict[str, CompetencyRecord]) -> dict:
    """`interrupt()`에 실어 보낼 페이로드. `app/hitl.py`가 이대로 폼을 그린다.

    축 하나 = 라디오 하나. 사전 채움된 레벨은 `default`로 실어 보내되, **묶인 역량의
    레벨이 없으면 아무것도 고르지 않은 채로 둔다** — 없는 근거로 미리 골라 두면
    사용자가 손대지 않은 항목이 "답한 것"으로 넘어간다(D28의 원칙).

    근거란은 **사전 채움이 실무운영·설계주도인 축에만** 붙는다. 폼이 배치라서 사용자가
    폼 안에서 올린 선택은 제출 전까지 알 수 없다 — 그건 이 화면의 한계이고, 어차피
    부풀림이 잦은 자리는 "시스템이 이미 높게 본 축"이라 값은 대부분 여기서 나온다.
    """
    labels = {category: label for category, label in survey.categories}
    questions: list[dict] = []

    for category, _ in survey.categories:
        for axis in survey.axes_of(category):
            bound = bindings.get(axis.axis_id)
            default = axis.marker_of(bound.level if bound else None)

            questions.append(
                {
                    "question_id": axis.axis_id,
                    "text": axis.name,
                    "section": labels[category],
                    "options": [*axis.option_labels(survey.order), survey.none_label],
                    "default": default,
                }
            )

            if bound is not None and bound.level in EVIDENCE_LEVELS:
                questions.append(
                    {
                        "question_id": axis.axis_id + EVIDENCE_SUFFIX,
                        "text": f"{axis.name} — 어떤 프로젝트에서였나요? (한 줄, 선택)",
                        "section": labels[category],
                    }
                )

    return {"kind": KIND, "questions": questions}


# --- 채점 -------------------------------------------------------------------


def score(
    survey: Survey,
    answers: dict[str, str],
    bindings: dict[str, CompetencyRecord],
) -> tuple[dict[str, Level], dict[Category, float]]:
    """응답 → (`level_coordinates`, `coverage`).

    "해당 없음"과 무응답은 같게 다룬다 — 둘 다 `level_coordinates`에 안 들어가고
    커버리지 분자에서 빠진다. 레벨 0이 아니라 **모르는 축**이라는 뜻이다(질문세트 §1).
    """
    coordinates: dict[str, Level] = {}
    answered: dict[Category, int] = {}
    total: dict[Category, int] = {}

    for axis in survey.axes:
        total[axis.category] = total.get(axis.category, 0) + 1
        level = axis.level_of(answers.get(axis.axis_id, ""))
        if level is None:
            continue
        coordinates[key_of(axis, bindings)] = level
        answered[axis.category] = answered.get(axis.category, 0) + 1

    coverage = {
        category: (answered.get(category, 0) / count if count else 0.0)
        for category, count in total.items()
    }
    return coordinates, coverage


def collect_evidence(survey: Survey, answers: dict[str, str]) -> dict[str, str]:
    """축 id → 사용자가 적은 근거 한 줄. 빈 값은 담지 않는다."""
    found: dict[str, str] = {}
    for axis in survey.axes:
        text = (answers.get(axis.axis_id + EVIDENCE_SUFFIX) or "").strip()
        if text:
            found[axis.axis_id] = text
    return found


def build_profile(
    survey: Survey,
    answers: dict[str, str],
    bindings: dict[str, CompetencyRecord],
    owned: list[CompetencyRecord],
    *,
    built_at: date | None = None,
) -> ProfileJSON:
    """설문 결과를 `ProfileJSON`으로 조립한다.

    역량 목록은 **추출된 것 + 설문이 새로 발견한 것**이다. 확정된 레벨은 레코드의
    `level`에도 반영한다 — 좌표와 레코드가 어긋나면 어느 쪽이 맞는지 아무도 모른다.

    근거 한 줄은 `Evidence`로 붙인다. `ProfileJSON`에 근거를 담을 칸이 따로 없기도
    하고, **"모든 판정에 evidence"라는 이 저장소의 규약과도 맞는다** — UC-2의
    `verify_criteria`가 프로필의 인용문을 읽으므로 델타 인터뷰가 실제로 짧아진다.
    """
    stamp = built_at or date.today()
    coordinates, coverage = score(survey, answers, bindings)
    evidence_texts = collect_evidence(survey, answers)

    bound_ids = {record.comp_id for record in bindings.values()}
    axis_by_key = {key_of(axis, bindings): axis for axis in survey.axes}

    competencies: list[CompetencyRecord] = []

    for record in owned:
        level = coordinates.get(record.comp_id, record.level)
        axis = axis_by_key.get(record.comp_id) if record.comp_id in bound_ids else None
        extra = evidence_texts.get(axis.axis_id) if axis else None
        competencies.append(_with_survey_result(record, level, extra, stamp))

    # 설문이 새로 발견한 역량 — 추출 결과에 없던 축이다 (폭 보완).
    known = {record.comp_id for record in owned}
    for axis in survey.axes:
        key = key_of(axis, bindings)
        if key in known or key not in coordinates:
            continue
        competencies.append(
            _with_survey_result(
                CompetencyRecord(
                    comp_id=key,
                    category=axis.category,
                    name=axis.name,
                    importance=Importance.REQUIRED,
                ),
                coordinates[key],
                evidence_texts.get(axis.axis_id),
                stamp,
            )
        )

    return ProfileJSON(
        competencies=competencies,
        level_coordinates=coordinates,
        coverage=coverage,
        built_at=stamp,
    )


def _with_survey_result(
    record: CompetencyRecord,
    level: Level | None,
    evidence_text: str | None,
    stamp: date,
) -> CompetencyRecord:
    evidence = list(record.evidence)
    if evidence_text:
        evidence.append(
            Evidence(source_name=EVIDENCE_SOURCE, quote=evidence_text, collected_at=stamp)
        )
    return record.model_copy(update={"level": level, "evidence": evidence})


# --- 노드 -------------------------------------------------------------------


def level_survey(state: GraphState) -> dict:
    """H2 — 레벨 측정 폼을 띄우고 답을 받아 `ProfileJSON`을 만든다.

    `interrupt()`는 이 함수 중간에서 멈춘다. 재개되면 **노드 처음부터 다시 실행**되고
    `interrupt()`만 답을 반환하므로(D27), 위쪽은 전부 순수 계산이어야 한다 — 여기서
    유일하게 비용이 있는 것은 `bind_axes`의 임베딩 왕복 1회이고, 재개 시 한 번 더
    나간다. 프로필 구축은 **1회성**이라 감수한다(대안은 상태에 캐시하는 것인데
    `GraphState`에 칸이 없다).

    보유 역량은 `required`에서 읽는다. 이름이 어색하지만 `GraphState`에 역량 목록을
    담는 칸이 그것뿐이고, 프로필 모드에서는 `extract`가 그 칸을 채운다.
    """
    survey = load_survey()
    owned = list(state.get("required") or [])
    bindings = bind_axes(survey, owned)

    answers = interrupt(build_payload(survey, bindings))

    return {"profile": build_profile(survey, answers or {}, bindings, owned)}
