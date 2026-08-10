from enum import Enum


class Category(str, Enum):          # 설계도 §7-4
    D1_SW_FOUNDATION = "D1_SW기반"
    D2_BACKEND       = "D2_백엔드"
    D3_CLOUD_INFRA   = "D3_클라우드"
    D4_ORCHESTRATION = "D4_오케스트레이션"
    D5_AI_INFRA      = "D5_AI인프라"
    C_CULTURE_FIT    = "C_인재상컬처핏"


class Level(str, Enum):             # 4단계 사다리
    LEARNED  = "학습함"
    USED     = "써봄"
    OPERATED = "실무운영"
    LED      = "설계주도"


class Importance(str, Enum):
    REQUIRED  = "필수"
    PREFERRED = "우대"


class SourceType(str, Enum):
    JD = "JD"
    TECH_BLOG = "기술블로그"
    VALUES = "인재상"
    TALK = "발표"
    OTHER = "기타"


class Confidence(str, Enum):
    HIGH = "상"
    MID = "중"
    LOW = "하"


class VerdictState(str, Enum):      # 기준 단위 판정
    MET = "충족"
    PARTIAL = "부분"
    UNMET = "미충족"
    UNKNOWN = "판단보류"


class MatchState(str, Enum):        # 역량 단위 집계 결과
    MET = "충족"
    ADJACENT = "인접"
    UNMET = "미보유"


class Track(str, Enum):
    SHOWCASE = "내세울것"
    FILL = "채울것"
    DROP = "포기할것"
