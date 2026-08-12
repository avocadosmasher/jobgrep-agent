# eval/samples — 평가 단위와 라벨 (T27)

| 파일 | 내용 | 만든 것 |
| --- | --- | --- |
| `units.json` | 평가 단위 30건 (차원 5 × 6) | `python -m eval.samples.build` |
| `LABELING.md` | 사람 채점표 — 척도·기준·게이트 + 30문항 | 〃 |
| `human_labels.json` | **사람** 라벨 (정답) | 사람이 채운다 |
| `judge_labels.json` | **심사모델** 라벨 | `python -m eval.judge --run` |
| `brief_filled.json` | `fill_brief_slots`를 실제로 돌려 얼린 브리프 | 1회 실행 산출 |

## 순서

```bash
uv run python -m eval.samples.build     # 단위·채점표 생성 (사람 라벨은 보존된다)
#   → LABELING.md 를 읽고 human_labels.json 의 score 를 채운다
uv run python -m eval.judge --run       # 심사모델 채점 (API 호출)
uv run python -m eval.judge --report    # 사람-LLM 일치율
```

## 이 세트가 지키는 것

- **평가 단위 하나 = 출력 항목 하나** (설계도 §11-5 불변식). 브리프 전체를 통째로
  묻지 않는다 — 소판정 단위로 물어야 평가자 간 일치도가 올라간다(§13).
- **원본과 변형이 섞여 있다.** 전부 정상 산출이면 채점표가 `적합` 일색이 되고
  일치율은 "둘 다 만점을 줬다"는 뜻밖에 없다. 변형이 섞여야 판별력이 재진다.
- **정답을 적어 두지 않는다.** `units.json`의 `provenance`는 감사용이며 채점표와
  심사 프롬프트 어디에도 실리지 않는다(테스트가 그것을 건다).
- **사람 라벨은 재생성해도 지워지지 않는다.** `build`는 기존 `score`를 보존한다
  (`--force`를 줘야 비운다).
- `brief_filled.json`은 손으로 쓴 문장이 아니라 **실제 실행 산출**이다. 이력서·JD
  픽스처에서 나온 브리프의 ◇ 슬롯을 `fill_brief_slots`가 채운 결과를 얼린 것이다.
