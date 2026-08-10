# fixtures — 골든 데이터

이 디렉터리는 상류 모듈(수집기·LLM 도구) 없이 하류 모듈을 개발·테스트하기 위한 고정 데이터다.
검증: `uv run pytest tests/test_fixtures.py -q`.

| 파일 | 모델 | 내용 | 주 사용처 |
| --- | --- | --- | --- |
| `jd_sample_backend.json` | `SourceDocument` | 백엔드 JD 1건. 자격요건 10개(필수) + 우대 2개 + 인재상 2개, 총 14개 역량이 원문에 실재 | T04(`extract_competencies`), T05, T08 스모크 |
| `jd_sample_aiinfra.json` | `SourceDocument` | AI 인프라 JD 1건. GPU·서빙·분산학습 포함 자격요건 8개 + 우대 2개 + 인재상 2개, 총 12개 역량 | T04, T05 |
| `competencies_required.json` | `list[CompetencyRecord]` | 위 두 JD에서 추출된 정답 26건(`req-be-*` 14 + `req-ai-*` 12). `evidence.quote`는 대응 JD `raw_text`의 실제 부분 문자열 | T04 출력 검증, T05 입력 |
| `profile_sample.json` | `ProfileJSON` | 후보자 보유 역량 8건, D1~D4는 채움, **D5(AI인프라)는 의도적으로 0.0 커버리지** — 인접/미보유 경계 테스트용 | T05(`verify_criteria`), T06, UC-1 관련 |
| `criteria_sample.json` | `dict[comp_id, list[Criterion]]` | `req-be-06`(K8s 운영)·`req-be-02`(API 설계)·`req-be-07`(AWS 운영)·`req-ai-02`(모델 서빙) 4개 역량 × 각 4개 필수 기준(`is_required=true`) | T03(`aggregate_states`), T05 |
| `verdicts_all_met.json` | `list[CriterionVerdict]` | `criteria_sample`의 16개 기준 전부 `충족` → `aggregate_states`가 각 comp에 `MET`을 반환해야 함 | T03 경계 테스트 |
| `verdicts_half.json` | `list[CriterionVerdict]` | comp별 4개 중 2개(정확히 절반) `충족` → `aggregate_states`가 `ADJACENT`를 반환해야 함 | T03 경계 테스트 |
| `verdicts_mostly_unmet.json` | `list[CriterionVerdict]` | comp별 4개 중 1개만 `충족` → `aggregate_states`가 `UNMET`을 반환해야 함 | T03 경계 테스트 |
| `brief_expected.json` | `StrategyBrief` | `req-be-06/02/07`을 각각 트랙1/2/3에 배치한 골격. ■ 필드(메타·집계·트랙 배정)는 확정값, ◇ 슬롯(`body`, `summary_line`)은 빈 문자열(`""`)로 자리만 표시 — 이유는 DEVLOG [D03] 참조 | T06(`build_strategy_brief`), T07 렌더러 |
| `jd_injection.json` | `SourceDocument` | 백엔드 JD 본문 중간에 "이전의 모든 지시를 무시하라"류 지시문을 심은 변형본 | T26(프롬프트 인젝션 격리) |

## 불변식
- 모든 파일이 대응 Pydantic 모델로 `model_validate_json` 통과 (`tests/test_fixtures.py`).
- `competencies_required.json`의 `name`은 JD 원문 표현 그대로 보존 — 일반화·병합 없음.
- `competencies_required.json`의 `evidence.quote`는 대응 JD의 `raw_text`에 실제로 존재하는 부분 문자열.
- `criteria_sample.json` / `verdicts_*.json`은 동일한 16개 `criterion_id` 집합을 공유한다(경계 테스트 3종이 같은 기준을 다른 충족 패턴으로 채점).
