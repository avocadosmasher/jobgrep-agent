"""LLM-as-judge와 일치율 리포트 — `python -m eval.judge --report` (T27, 설계도 §13).

    uv run python -m eval.samples.build     # 평가 단위·채점표 생성
    uv run python -m eval.judge --run       # 심사모델 채점 (API 호출)
    uv run python -m eval.judge --report    # 사람 라벨과 대조해 수치 산출

심사 모델은 생성 모델과 달라야 한다
-----------------------------------
§13의 불변식이다 — 같은 모델은 자기 출력을 과대평가한다. 그래서 이 모듈은
`llm.client.default_model()`과 심사 모델이 같으면 **호출 전에 멈춘다.** 기본
심사 모델은 `gpt-4o`로, 생성 계열(`gpt-4.1-mini`)과 다른 계열이다. gpt-5 계열은
`complete_structured`가 항상 넘기는 `temperature`를 받지 않아 쓸 수 없다(DEVLOG D22).

배치 호출 — **차원 단위**
-------------------------
항목별 개별 호출은 저장소 규약 위반이다. 그렇다고 30건을 한 번에 넣으면 차원별
기준 문장 다섯 벌이 한 프롬프트에 섞여 심사자가 어느 잣대로 재는지 흐려진다.
그래서 **차원 하나 = 호출 하나**로 묶는다(30건 → 5회). 배치이면서 잣대가 하나다.

외부 텍스트는 전부 울타리 안으로
--------------------------------
평가 단위의 본문은 수집한 JD·이력서에서 파생된 문자열이다 — 인젝션 규약 ⑤가
말하는 "2차 경로"가 바로 이것이라 `wrap_document()`를 지난다(규약 ①). 새 태그
이름을 만들지 않고 `document` 울타리에 `kind` 속성만 붙이는 이유는, 등록되지 않은
태그 이름은 그 울타리만 위조 가능해지기 때문이다(규약 ②, T26b D81과 같은 판단).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from pydantic import BaseModel

from eval.rubric import (
    CULTURE_GATE_NOTE,
    CULTURE_GATE_RULES,
    DIMENSION_CRITERIA,
    Dimension,
    EvalUnit,
    Label,
    Score,
    compute_agreement,
    format_report,
)
from eval.samples.build import HUMAN_PATH, UNITS_PATH, WORKSHEET_PATH
from llm.client import DEFAULT_INSTRUCTIONS, complete_structured, default_model
from llm.sanitize import wrap_document

JUDGE_LABELS_PATH = Path(__file__).resolve().parent / "samples" / "judge_labels.json"

JUDGE_MODEL_ENV = "JOBPREP_JUDGE_MODEL"
DEFAULT_JUDGE_MODEL = "gpt-4o"

# 울타리 종류. 태그 이름이 아니라 **속성 값**이다 — `FENCE_TAGS`(T26 소유)를 건드리지
# 않고도 영역이 갈린다. 테스트는 이 상수에서 유도할 것(인젝션 규약 ⑥).
UNIT_KIND = "eval_unit"
CONTEXT_KIND = "eval_context"


class JudgeModelError(Exception):
    """심사 모델 설정이 §13 불변식을 어김 (생성 모델과 동일)."""


class JudgeVerdict(BaseModel):
    unit_id: str
    score: Score
    gate_violation: bool
    rationale: str


class JudgeSheet(BaseModel):
    verdicts: list[JudgeVerdict]


def judge_model() -> str:
    return os.environ.get(JUDGE_MODEL_ENV) or DEFAULT_JUDGE_MODEL


def assert_distinct(model: str, generator: str | None = None) -> None:
    """§13 불변식의 기계화 — 생성자와 심사자가 같으면 시작조차 하지 않는다."""
    generator = generator or default_model()
    if model.strip().lower() == generator.strip().lower():
        raise JudgeModelError(
            f"심사 모델과 생성 모델이 같다({model}). 같은 모델은 자기 출력을 과대평가한다 "
            f"(설계도 §13) — {JUDGE_MODEL_ENV} 환경 변수로 다른 모델을 지정할 것."
        )


def build_prompt(units: list[EvalUnit], dimension: Dimension) -> str:
    """차원 하나치 채점 프롬프트. **`provenance`는 절대 싣지 않는다**(정답 누설)."""
    lines = [
        "너는 취업 전략 브리프 산출물의 품질을 심사한다. 아래 평가 단위 각각에 대해",
        f"'{dimension.value}' 차원 하나만 보고 3점 척도로 채점하라.",
        "",
        f"[채점 기준] {DIMENSION_CRITERIA[dimension]}",
        "",
        f"[척도] {Score.FIT.value} / {Score.PARTIAL.value} / {Score.UNFIT.value}",
        "",
        "[컬처핏 안전선 — 점수가 아니라 게이트]",
    ]
    lines += [f"- {rule}" for rule in CULTURE_GATE_RULES]
    lines += [
        CULTURE_GATE_NOTE,
        "해당하면 gate_violation을 true로 두고, 점수도 함께 매겨라.",
        "",
        "[참고 입력] — 판정에 쓰는 원문이다. 채점 대상이 아니다.",
    ]

    # 같은 원문이 여러 단위에 붙는다. 한 번만 싣고 참조시킨다 — 프롬프트가 짧아지고,
    # 무엇보다 같은 원문을 여러 번 실으면 모델이 그 반복을 신호로 오해한다.
    context_ids: dict[str, str] = {}
    for unit in units:
        body = unit.context.strip()
        if body and body not in context_ids:
            context_ids[body] = f"ctx-{len(context_ids) + 1}"
    for body, context_id in context_ids.items():
        lines.append(wrap_document(body, kind=CONTEXT_KIND, context_id=context_id))

    lines += ["", "[채점 대상]"]
    for unit in units:
        reference = context_ids.get(unit.context.strip(), "")
        lines.append(
            wrap_document(
                unit.subject,
                kind=UNIT_KIND,
                unit_id=unit.unit_id,
                dimension=unit.dimension.value,
                title=unit.title,
                context_ref=reference,
            )
        )

    lines += [
        "",
        f"평가 단위 {len(units)}건 전부에 대해 unit_id·score·gate_violation·rationale을 낸다.",
        "rationale은 한 문장으로, 무엇이 근거였는지만 적는다.",
    ]
    return "\n".join(lines)


def judge_units(
    units: list[EvalUnit],
    *,
    model: str | None = None,
    complete=None,
    instructions: str = DEFAULT_INSTRUCTIONS,
) -> list[Label]:
    """평가 단위를 차원별 배치로 채점한다.

    `complete`는 테스트 주입점이다 — **기본 인자로 함수를 박지 않는다**(주입점 규약,
    DEVLOG D71). 모델이 모르는 unit_id를 지어내면 버린다(입력에 없는 채점은 대조
    불가이며, 그대로 두면 일치율 분모가 조용히 흔들린다).
    """
    complete = complete or complete_structured
    target = model or judge_model()
    assert_distinct(target)

    known = {unit.unit_id for unit in units}
    labels: list[Label] = []
    for dimension in Dimension:
        batch = [unit for unit in units if unit.dimension is dimension]
        if not batch:
            continue
        sheet = complete(
            build_prompt(batch, dimension),
            JudgeSheet,
            model=target,
            instructions=instructions,
        )
        for verdict in sheet.verdicts:
            if verdict.unit_id not in known:
                continue
            labels.append(
                Label(
                    unit_id=verdict.unit_id,
                    score=verdict.score,
                    gate_violation=verdict.gate_violation,
                    rater=target,
                    rationale=verdict.rationale,
                )
            )
    return labels


# ── 입출력 ───────────────────────────────────────────────────────────────────
def load_units(path: Path = UNITS_PATH) -> list[EvalUnit]:
    if not path.exists():
        raise SystemExit(f"{path}가 없다 — 먼저 `python -m eval.samples.build`를 돌릴 것.")
    return [EvalUnit.model_validate(item) for item in json.loads(path.read_text(encoding="utf-8"))]


def load_labels(path: Path) -> list[Label]:
    if not path.exists():
        return []
    return [Label.model_validate(item) for item in json.loads(path.read_text(encoding="utf-8"))]


def save_labels(labels: list[Label], path: Path) -> None:
    path.write_text(
        json.dumps([label.model_dump() for label in labels], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _scored(labels: list[Label]) -> int:
    return sum(1 for label in labels if label.score is not None)


def run_judge(*, model: str | None = None) -> list[Label]:
    units = load_units()
    labels = judge_units(units, model=model)
    save_labels(labels, JUDGE_LABELS_PATH)
    return labels


def report(*, model: str | None = None) -> int:
    units = load_units()
    human = load_labels(HUMAN_PATH)
    judge = load_labels(JUDGE_LABELS_PATH)
    used_model = model or (next((label.rater for label in judge if label.rater), "") or judge_model())

    if _scored(human) == 0:
        print(
            f"사람 라벨이 하나도 없다. {WORKSHEET_PATH}를 보고 채점한 뒤\n"
            f"{HUMAN_PATH}의 score를 채울 것 (부적합/부분적합/적합).",
        )
        return 1
    if _scored(judge) == 0:
        print(f"심사모델 라벨이 없다. 먼저 `python -m eval.judge --run`을 돌릴 것.")
        return 1

    print(format_report(compute_agreement(units, human, judge, judge_model=used_model)))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="루브릭 사람-LLM 일치율 (T27)")
    parser.add_argument("--run", action="store_true", help="심사모델로 채점한다 (API 호출·비용 발생)")
    parser.add_argument("--report", action="store_true", help="사람 라벨과 대조해 일치율을 낸다")
    parser.add_argument("--model", default=None, help=f"심사 모델 (기본 {DEFAULT_JUDGE_MODEL})")
    args = parser.parse_args(argv)

    if not args.run and not args.report:
        parser.print_help()
        return 0

    if args.run:
        labels = run_judge(model=args.model)
        print(f"심사모델 채점 {_scored(labels)}건 → {JUDGE_LABELS_PATH}")

    if args.report:
        return report(model=args.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
