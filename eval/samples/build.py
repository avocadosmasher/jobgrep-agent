"""평가 단위 생성기 — `python -m eval.samples.build` (T27).

**손으로 적은 JSON을 두지 않는다.** 평가 단위는 골든 픽스처(`fixtures/`)와 실제
실행 산출(`brief_filled.json` — `fill_brief_slots`를 한 번 돌려 얼린 것)에서
코드로 파생된다. 그래야 픽스처가 바뀌면 평가 단위도 같이 바뀌고, 무엇이 원본이고
무엇이 변형인지가 **표 하나**(아래 `_units()`)에 남는다.

왜 변형(음성 대조군)을 섞는가
-----------------------------
전부 정상 산출이면 채점표가 `적합` 일색이 되고, 그때 일치율은 "둘 다 만점을 줬다"는
뜻밖에 없다 — 우연 일치를 뺀 κ가 0으로 떨어져 그 사실이 드러난다. 판별력을 재려면
**틀린 산출이 섞여 있어야** 한다. 이 저장소가 테스트를 뮤테이션으로 검증해 온 것과
같은 논리다.

변형은 여기서 **명시적으로** 만든다(`provenance`에 이유를 적는다). 다만 그 문구는
채점자에게도 심사모델에게도 보이지 않는다 — 정답을 알려주면 일치율이 무의미해진다.

**기대 점수는 어디에도 적지 않는다.** 정답은 사람이 매긴 라벨이지 생성자의 의도가
아니다. "이건 부적합일 것"이라고 적어 두면 그 순간 사람 라벨이 그 문구를 따라간다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from contracts.models import (
    CompetencyRecord,
    Criterion,
    CriterionVerdict,
    ProfileJSON,
    SourceDocument,
    StrategyBrief,
)
from eval.rubric import Dimension, EvalUnit, Label, render_worksheet

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "fixtures"
SAMPLES = Path(__file__).resolve().parent

UNITS_PATH = SAMPLES / "units.json"
HUMAN_PATH = SAMPLES / "human_labels.json"
WORKSHEET_PATH = SAMPLES / "LABELING.md"
FILLED_BRIEF_PATH = SAMPLES / "brief_filled.json"

# `aggregate_states`의 집계 규칙 (contracts/tools.py 독스트링 = 정본, 설계도 §7-2).
# 채점자와 심사모델에게 **같은 규칙 문장**을 준다 — 규칙을 모르면 집계 타당성을 못 잰다.
AGGREGATE_RULE = (
    "집계 규칙(§7-2): 필수 기준을 전부 충족하면 '충족', 필수 기준의 절반 이상을 "
    "충족하면 '인접', 그 외는 '미보유'. 근거 없이는 충족/부분/미충족 판정을 내리지 "
    "않고 '판단보류'로 두거나 질문으로 승격한다."
)


# ── 픽스처 로딩 ───────────────────────────────────────────────────────────────
def _load(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _load_fixtures() -> dict:
    jd = SourceDocument.model_validate(_load("jd_sample_backend.json"))
    profile = ProfileJSON.model_validate(_load("profile_sample.json"))
    comps = {
        record["comp_id"]: CompetencyRecord.model_validate(record)
        for record in _load("competencies_required.json")
    }
    criteria = {
        comp_id: [Criterion.model_validate(item) for item in items]
        for comp_id, items in _load("criteria_sample.json").items()
    }
    verdicts = {
        name: [CriterionVerdict.model_validate(item) for item in _load(f"verdicts_{name}.json")]
        for name in ("all_met", "half", "mostly_unmet")
    }
    if not FILLED_BRIEF_PATH.exists():
        raise SystemExit(
            f"{FILLED_BRIEF_PATH.name}이 없다. ◇ 슬롯이 채워진 실제 산출이 있어야 "
            "실행 가능성·전략 타당성 단위를 만들 수 있다 — "
            "`fill_brief_slots(brief_expected)` 결과를 이 경로에 얼려 둘 것."
        )
    brief = StrategyBrief.model_validate_json(FILLED_BRIEF_PATH.read_text(encoding="utf-8"))
    return {
        "jd": jd,
        "profile": profile,
        "comps": comps,
        "criteria": criteria,
        "verdicts": verdicts,
        "brief": brief,
    }


# ── 항목 렌더 (subject/context 문자열) ────────────────────────────────────────
def _claim(name: str, required: str | None, mine: str | None, state: str) -> str:
    return (
        f"[주장] {name}\n"
        f"       요구 레벨 {required or '미상'} / 내 레벨 {mine or '없음'} → 판정 '{state}'"
    )


def _quote_line(quote: str, source: str, collected: str) -> str:
    return f'[근거] "{quote}"  (출처: {source}, 수집일 {collected})'


def _profile_context(profile: ProfileJSON) -> str:
    lines = ["[내 프로필 — 이력서에서 추출된 보유 역량]"]
    for record in profile.competencies:
        quote = record.evidence[0].quote if record.evidence else "(근거 없음)"
        lines.append(f'- {record.name} (레벨 {record.level.value if record.level else "미상"}) : "{quote}"')
    return "\n".join(lines)


def _jd_context(jd: SourceDocument) -> str:
    return f"[채용공고 원문 — {jd.company} {jd.title}]\n{jd.raw_text}"


def _competency_subject(name: str, category: str, importance: str) -> str:
    return f"[추출된 요구역량] {name}\n[대분류] {category}   [중요도] {importance}"


def _match_subject(
    name: str,
    criteria: list[Criterion],
    verdicts: list[CriterionVerdict],
    aggregated: str,
    *,
    drop_evidence: bool = False,
) -> str:
    verdict_map = {verdict.criterion_id: verdict for verdict in verdicts}
    lines = [f"[역량] {name}", "[기준별 판정]"]
    for index, criterion in enumerate(criteria, start=1):
        verdict = verdict_map.get(criterion.criterion_id)
        state = verdict.state.value if verdict else "판단보류"
        lines.append(f"  {index}. {criterion.text}")
        lines.append(f"     → {state}")
        quote = ""
        if verdict and verdict.evidence and not drop_evidence:
            quote = verdict.evidence[0].quote
        lines.append(f'     근거: {chr(34) + quote + chr(34) if quote else "(없음)"}')
    lines.append(f"[집계 결과] {aggregated}")
    return "\n".join(lines)


def _card_subject(card, remaining: int, label: str, body: str | None = None) -> str:
    priority = f" · 우선순위 {card.priority}" if card.priority else ""
    return (
        f"[{label}] {card.name}\n"
        f"       상태 {card.state.value} · 요구 {card.required_level.value if card.required_level else '미상'}"
        f" · 내 레벨 {card.my_level.value if card.my_level else '없음'}{priority}\n"
        f"[남은 기간] {remaining}일\n"
        f"[제안]\n{body if body is not None else card.body}"
    )


def _counts(brief: StrategyBrief) -> dict[str, int]:
    """집계 카운트를 한글 값으로 다시 색인한다.

    `summary_counts`의 키는 `MatchState` 멤버이고 **str 열거형이어도 해시가 이름 기준**이라
    `counts["충족"]`은 걸리지 않는다. 값으로 찾을 일이 여기서만 있어 여기서 푼다.
    """
    return {
        (key.value if hasattr(key, "value") else str(key)): value
        for key, value in brief.summary_counts.items()
    }


def _triage_subject(brief: StrategyBrief, *, track1, track2, track3) -> str:
    def render(cards) -> str:
        if not cards:
            return "  (없음)"
        return "\n".join(
            f"  - {name} (상태 {state}, 요구 {required}, 내 레벨 {mine})"
            for name, state, required, mine in cards
        )

    counts = _counts(brief)
    return (
        f"[남은 기간] {brief.meta.days_remaining}일 (목표일 {brief.meta.target_date})\n"
        f"[집계] 충족 {counts.get('충족', 0)} · "
        f"인접 {counts.get('인접', 0)} · "
        f"미보유 {counts.get('미보유', 0)}\n"
        f"[트랙1 지금 내세울 것]\n{render(track1)}\n"
        f"[트랙2 기간 내 채울 것]\n{render(track2)}\n"
        f"[트랙3 이번엔 포기할 것]\n{render(track3)}"
    )


def _card_tuple(card):
    return (
        card.name,
        card.state.value,
        card.required_level.value if card.required_level else "미상",
        card.my_level.value if card.my_level else "없음",
    )


# ── 평가 단위 표 ──────────────────────────────────────────────────────────────
def _units() -> list[EvalUnit]:  # noqa: C901 — 표 하나로 읽히는 편이 낫다
    data = _load_fixtures()
    jd: SourceDocument = data["jd"]
    profile: ProfileJSON = data["profile"]
    comps: dict[str, CompetencyRecord] = data["comps"]
    criteria: dict[str, list[Criterion]] = data["criteria"]
    verdicts: dict[str, list[CriterionVerdict]] = data["verdicts"]
    brief: StrategyBrief = data["brief"]

    jd_context = _jd_context(jd)
    profile_context = _profile_context(profile)
    remaining = brief.meta.days_remaining

    k8s_card = brief.track1[0]
    api_card = brief.track2[0]
    aws_card = brief.track3[0]

    pf = {record.comp_id: record for record in profile.competencies}
    units: list[EvalUnit] = []

    def add(unit_id: str, dimension: Dimension, title: str, subject: str, context: str, provenance: str):
        units.append(
            EvalUnit(
                unit_id=unit_id,
                dimension=dimension,
                title=title,
                subject=subject,
                context=context,
                provenance=provenance,
            )
        )

    # ── 근거 타당성 ──────────────────────────────────────────────────────────
    k8s_evidence = k8s_card.evidence[0]
    add(
        "E01",
        Dimension.EVIDENCE,
        "충족 판정에 붙은 근거",
        _claim(k8s_card.name, k8s_card.required_level.value, k8s_card.my_level.value, k8s_card.state.value)
        + "\n"
        + _quote_line(k8s_evidence.quote, k8s_evidence.source_name, str(k8s_evidence.collected_at)),
        profile_context,
        "원본: brief track1 카드의 근거",
    )
    add(
        "E02",
        Dimension.EVIDENCE,
        "충족 판정에 붙은 근거",
        _claim(k8s_card.name, k8s_card.required_level.value, k8s_card.my_level.value, k8s_card.state.value)
        + "\n"
        + _quote_line(pf["pf-07"].evidence[0].quote, "이력서", "2026-07-15"),
        profile_context,
        "변형: 인용을 같은 프로필의 다른 역량(Docker)으로 교체 — 실재하나 주장과 무관",
    )
    add(
        "E03",
        Dimension.EVIDENCE,
        "충족 판정에 붙은 근거",
        _claim(comps["req-be-07"].name, "실무운영", "실무운영", "충족")
        + "\n"
        + _quote_line(pf["pf-05"].evidence[0].quote, "이력서", "2026-07-15"),
        profile_context,
        "변형: 인용은 실재하지만 그 인용이 받치는 레벨은 '써봄'이다 — 범위 초과 주장",
    )
    add(
        "E04",
        Dimension.EVIDENCE,
        "요구역량 추출에 붙은 근거",
        f"[주장] 이 공고는 '{comps['req-be-06'].name}'을 요구한다\n"
        + _quote_line(comps["req-be-06"].evidence[0].quote, jd.title, str(jd.collected_at)),
        jd_context,
        "원본: competencies_required.json req-be-06",
    )
    add(
        "E05",
        Dimension.EVIDENCE,
        "요구역량 추출에 붙은 근거",
        f"[주장] 이 공고는 '{comps['req-be-06'].name}'을 요구한다\n"
        + _quote_line("300노드 이상 규모의 Kubernetes 클러스터 운영 경험 필수", jd.title, str(jd.collected_at)),
        jd_context,
        "변형: 원문에 없는 인용(환각) — 규모 조건을 지어냈다",
    )
    add(
        "E06",
        Dimension.EVIDENCE,
        "요구역량 추출에 붙은 근거",
        f"[주장] 이 공고는 '{comps['req-be-04'].name}'을 요구한다\n"
        + _quote_line("CI/CD 파이프라인 구축 및 배포 자동화", jd.title, str(jd.collected_at)),
        jd_context,
        "변형: 인용은 원문에 실재하나 담당업무 줄이라 이 주장(메시지 큐)을 받치지 못한다",
    )

    # ── 요구역량 정확성 ──────────────────────────────────────────────────────
    add(
        "C01",
        Dimension.COMPETENCY,
        "공고에서 뽑아낸 요구역량",
        _competency_subject(comps["req-be-02"].name, comps["req-be-02"].category.value, comps["req-be-02"].importance.value),
        jd_context,
        "원본: req-be-02",
    )
    add(
        "C02",
        Dimension.COMPETENCY,
        "공고에서 뽑아낸 요구역량",
        _competency_subject("API 개발 능력", comps["req-be-02"].category.value, comps["req-be-02"].importance.value),
        jd_context,
        "변형: 원문 표현을 일반화·병합 (저장소 규약 위반: 역량명은 원문 그대로)",
    )
    add(
        "C03",
        Dimension.COMPETENCY,
        "공고에서 뽑아낸 요구역량",
        _competency_subject("Rust 기반 시스템 프로그래밍 경험 3년 이상", "D1_SW기반", "필수"),
        jd_context,
        "변형: 원문에 없는 항목(환각)",
    )
    add(
        "C04",
        Dimension.COMPETENCY,
        "공고에서 뽑아낸 요구역량",
        _competency_subject(comps["req-be-11"].name, comps["req-be-11"].category.value, "필수"),
        jd_context,
        "변형: 원문에서 '우대사항'인 항목을 '필수'로 표기 — 중요도 오분류",
    )
    add(
        "C05",
        Dimension.COMPETENCY,
        "공고에서 뽑아낸 요구역량",
        _competency_subject(comps["req-be-13"].name, comps["req-be-13"].category.value, comps["req-be-13"].importance.value),
        jd_context,
        "원본: req-be-13 (인재상 항목)",
    )
    add(
        "C06",
        Dimension.COMPETENCY,
        "공고에서 뽑아낸 요구역량",
        _competency_subject("Adobe Photoshop을 활용한 배너 디자인 경험", "D1_SW기반", "필수"),
        jd_context,
        "변형: 이 직무와 무관한 항목",
    )

    # ── 매칭 판정 타당성 ─────────────────────────────────────────────────────
    def for_comp(comp_id: str, bucket: str) -> list[CriterionVerdict]:
        ids = {criterion.criterion_id for criterion in criteria[comp_id]}
        return [verdict for verdict in verdicts[bucket] if verdict.criterion_id in ids]

    add(
        "M01",
        Dimension.MATCH,
        "기준 분해와 집계 결과",
        _match_subject(comps["req-be-06"].name, criteria["req-be-06"], for_comp("req-be-06", "all_met"), "충족"),
        AGGREGATE_RULE,
        "원본: criteria_sample + verdicts_all_met (필수 4/4 충족 → 충족)",
    )
    add(
        "M02",
        Dimension.MATCH,
        "기준 분해와 집계 결과",
        _match_subject(comps["req-be-02"].name, criteria["req-be-02"], for_comp("req-be-02", "half"), "인접"),
        AGGREGATE_RULE,
        "원본: verdicts_half (필수 2/4 충족 → 인접, 경계값)",
    )
    add(
        "M03",
        Dimension.MATCH,
        "기준 분해와 집계 결과",
        _match_subject(comps["req-be-07"].name, criteria["req-be-07"], for_comp("req-be-07", "mostly_unmet"), "미보유"),
        AGGREGATE_RULE,
        "원본: verdicts_mostly_unmet (필수 1/4 충족 → 미보유)",
    )
    add(
        "M04",
        Dimension.MATCH,
        "기준 분해와 집계 결과",
        _match_subject(comps["req-be-02"].name, criteria["req-be-02"], for_comp("req-be-02", "half"), "충족"),
        AGGREGATE_RULE,
        "변형: 판정 패턴은 절반인데 집계만 '충족'으로 올렸다",
    )
    vague = [
        Criterion(criterion_id="cr-vague-01", comp_id="req-be-06", text="Kubernetes를 잘 다룰 수 있다", is_required=True),
        Criterion(criterion_id="cr-vague-02", comp_id="req-be-06", text="컨테이너 기술에 대한 이해가 깊다", is_required=True),
        Criterion(criterion_id="cr-vague-03", comp_id="req-be-06", text="운영 경험이 충분하다", is_required=True),
        Criterion(criterion_id="cr-vague-04", comp_id="req-be-06", text="클러스터 관리에 자신이 있다", is_required=True),
    ]
    vague_verdicts = [
        CriterionVerdict(criterion_id=criterion.criterion_id, state="충족", rationale="프로필상 충족", evidence=[])
        for criterion in vague
    ]
    add(
        "M05",
        Dimension.MATCH,
        "기준 분해와 집계 결과",
        _match_subject(comps["req-be-06"].name, vague, vague_verdicts, "충족"),
        AGGREGATE_RULE,
        "변형: 기준 문장이 예/아니오로 판정 불가능하게 뭉뚱그려졌다",
    )
    add(
        "M06",
        Dimension.MATCH,
        "기준 분해와 집계 결과",
        _match_subject(
            comps["req-ai-02"].name,
            criteria["req-ai-02"],
            for_comp("req-ai-02", "all_met"),
            "충족",
            drop_evidence=True,
        ),
        AGGREGATE_RULE,
        "변형: 근거를 모두 제거한 채 전 기준 충족 — 근거 없는 판정은 판단보류/질문 승격이어야 한다",
    )

    # ── 실행 가능성 ──────────────────────────────────────────────────────────
    add(
        "A01",
        Dimension.ACTIONABILITY,
        "트랙2 카드의 보완 방법",
        _card_subject(api_card, remaining, "트랙2 기간 내 채울 것"),
        profile_context,
        "원본: fill_brief_slots가 실제로 채운 ◇ 슬롯",
    )
    add(
        "A02",
        Dimension.ACTIONABILITY,
        "트랙3 카드의 권고 사유",
        _card_subject(aws_card, remaining, "트랙3 이번엔 포기할 것"),
        profile_context,
        "원본: fill_brief_slots가 실제로 채운 ◇ 슬롯",
    )
    add(
        "A03",
        Dimension.ACTIONABILITY,
        "트랙2 카드의 보완 방법",
        _card_subject(
            api_card,
            remaining,
            "트랙2 기간 내 채울 것",
            body="부족한 부분을 열심히 공부하고 관련 기술을 꾸준히 익히면 좋겠습니다. "
            "자신감을 갖고 준비하시면 충분히 좋은 결과가 있을 것입니다.",
        ),
        profile_context,
        "변형: 다음 행동이 지목되지 않는 격려문",
    )
    add(
        "A04",
        Dimension.ACTIONABILITY,
        "트랙2 카드의 보완 방법",
        _card_subject(
            aws_card,
            remaining,
            "트랙2 기간 내 채울 것",
            body="AWS Solutions Architect Associate·SysOps·Developer 자격증 3종을 먼저 취득하고, "
            "이후 6개월간 실무 프로젝트에서 EC2·VPC·IAM 운영을 경험한 뒤 지원하는 것을 권합니다.",
        ),
        profile_context,
        "변형: 남은 43일 안에 불가능한 분량 — 기간 비현실",
    )
    add(
        "A05",
        Dimension.ACTIONABILITY,
        "트랙2 카드의 보완 방법",
        _card_subject(
            api_card,
            remaining,
            "트랙2 기간 내 채울 것",
            body="gRPC protobuf 스키마 작성, API 버저닝 전략 수립, OpenAPI 문서화, 리소스 설계 리뷰, "
            "인증·인가 설계, 에러 응답 규격 정의를 모두 보완 대상으로 둡니다.",
        ),
        profile_context,
        "변형: 항목은 구체적이나 무엇을 먼저 할지가 없다 — 순서 부재",
    )
    add(
        "A06",
        Dimension.ACTIONABILITY,
        "트랙2 카드의 보완 방법",
        _card_subject(
            api_card,
            remaining,
            "트랙2 기간 내 채울 것",
            body="남은 기간에는 이 회사의 인적성·컬처핏 검사 대비를 병행하십시오. 기출 문항을 구해 "
            "'주도적으로 문제를 파고드는 사람'으로 보이는 선택지를 고르는 연습을 하고, 실제 성향과 "
            "다르더라도 인재상에 맞는 답변을 준비해 두면 통과율이 올라갑니다.",
        ),
        profile_context,
        "변형: 컬처핏 시험 공략법 — 게이트 위반 대상 (설계도 §4·§13)",
    )

    # ── 전략 타당성 ──────────────────────────────────────────────────────────
    t1, t2, t3 = _card_tuple(k8s_card), _card_tuple(api_card), _card_tuple(aws_card)
    add(
        "S01",
        Dimension.STRATEGY,
        "3트랙 트리아지",
        _triage_subject(brief, track1=[t1], track2=[t2], track3=[t3]),
        profile_context,
        "원본: brief_expected의 트랙 배치",
    )
    add(
        "S02",
        Dimension.STRATEGY,
        "3트랙 트리아지",
        _triage_subject(brief, track1=[t1, t3], track2=[t2], track3=[]),
        profile_context,
        "변형: 미보유 항목을 강점(트랙1)으로 재라벨링 — §11-6이 금지한 것",
    )
    add(
        "S03",
        Dimension.STRATEGY,
        "3트랙 트리아지",
        _triage_subject(brief, track1=[t1], track2=[t3, t2], track3=[]),
        profile_context,
        "변형: 남은 기간에 못 채울 미보유 항목을 트랙2 1순위로 올렸다",
    )
    add(
        "S04",
        Dimension.STRATEGY,
        "요약 판정 한 줄",
        f"[남은 기간] {remaining}일\n"
        f"[집계] 충족 {_counts(brief).get('충족', 0)} · 인접 {_counts(brief).get('인접', 0)} · "
        f"미보유 {_counts(brief).get('미보유', 0)}\n"
        f"[총평]\n{brief.summary_line}",
        profile_context,
        "원본: fill_brief_slots가 실제로 채운 ◇ summary_line",
    )
    add(
        "S05",
        Dimension.STRATEGY,
        "컬처핏 항목 서술",
        f"[남은 기간] {remaining}일\n"
        "[컬처핏] 이 회사의 인재상은 '문제의 근본 원인을 끝까지 파고드는 주도적 태도'입니다.\n"
        "[권고]\n면접과 자기소개서에서는 실제 성향과 무관하게 주도적·협업 지향으로 일관되게 답하십시오. "
        "혼자 일하는 것을 선호한다는 인상을 주는 경험은 빼고, 회사가 원하는 상을 먼저 만든 뒤 "
        "거기에 맞는 사례를 골라 배치하는 것이 통과에 유리합니다.",
        profile_context,
        "변형: 성향 위장 조언 — 게이트 위반 대상 (설계도 §4·§13)",
    )
    add(
        "S06",
        Dimension.STRATEGY,
        "3트랙 트리아지",
        _triage_subject(brief, track1=[], track2=[t1, t2, t3], track3=[]),
        profile_context,
        "변형: 충족 항목까지 '채울 것'으로 내려 강점이 사라졌다",
    )

    return units


# ── 파일 쓰기 ────────────────────────────────────────────────────────────────
def _merge_human_labels(units: list[EvalUnit], existing: list[Label]) -> list[Label]:
    """이미 매긴 사람 라벨은 **절대 덮지 않는다** — 다시 채점하는 비용이 이 프로젝트의 병목이다."""
    kept = {label.unit_id: label for label in existing}
    return [kept.get(unit.unit_id, Label(unit_id=unit.unit_id, score=None, rater="human")) for unit in units]


def write_samples(*, force: bool = False) -> list[EvalUnit]:
    units = _units()
    ids = [unit.unit_id for unit in units]
    if len(set(ids)) != len(ids):
        raise SystemExit("unit_id가 중복됐다 — 라벨이 뒤엉킨다")

    UNITS_PATH.write_text(
        json.dumps([unit.model_dump() for unit in units], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    existing: list[Label] = []
    if HUMAN_PATH.exists() and not force:
        existing = [Label.model_validate(item) for item in json.loads(HUMAN_PATH.read_text(encoding="utf-8"))]
    merged = _merge_human_labels(units, existing)
    HUMAN_PATH.write_text(
        json.dumps([label.model_dump() for label in merged], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    WORKSHEET_PATH.write_text(render_worksheet(units), encoding="utf-8")
    return units


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="평가 단위와 채점표를 픽스처에서 다시 만든다 (T27)")
    parser.add_argument("--force", action="store_true", help="이미 매긴 사람 라벨까지 비운다")
    args = parser.parse_args(argv)

    units = write_samples(force=args.force)
    labeled = sum(
        1
        for label in json.loads(HUMAN_PATH.read_text(encoding="utf-8"))
        if label.get("score")
    )
    print(f"평가 단위 {len(units)}건 → {UNITS_PATH.relative_to(ROOT)}")
    for dimension in Dimension:
        count = sum(1 for unit in units if unit.dimension is dimension)
        print(f"  {dimension.value:<12} {count}건")
    print(f"채점표 → {WORKSHEET_PATH.relative_to(ROOT)}")
    print(f"사람 라벨 {labeled}/{len(units)}건 채워짐 → {HUMAN_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
