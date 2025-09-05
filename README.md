# SAI Challenge - RAG Pipeline

이 프로젝트는 SAI Challenge를 위한 RAG(Retrieval-Augmented Generation) 파이프라인입니다.

## 📁 프로젝트 구조

```
sai-challenge/
├── modules/                          # 핵심 모듈들
│   ├── __init__.py
│   ├── config.py                     # 설정 관리
│   ├── document_manager.py           # 문서 관리 (벡터 DB + 메타데이터)
│   ├── rag_pipeline.py              # RAG 파이프라인
│   ├── search_engine.py             # 검색 엔진
│   ├── reranking.py                 # 문서 재순위화
│   ├── answer_generator.py          # 답변 생성기
│   ├── prompting.py                 # 프롬프트 관리
│   ├── keyword_extractors/          # 키워드 추출기들
│   │   ├── __init__.py
│   │   ├── base_extractor.py
│   │   ├── basic_extractor.py
│   │   ├── domain_extractor.py
│   │   └── llm_extractor.py
│   ├── search_methods/              # 검색 방법들
│   │   ├── __init__.py
│   │   ├── base_method.py
│   │   ├── keyword_search.py
│   │   ├── semantic_search.py
│   │   └── hybrid_search.py
│   └── search_tools/                # 검색 도구들
│       ├── __init__.py
│       ├── base_tool.py
│       ├── scienceon_tool.py
│       └── arxiv_tool.py
├── gemini_client.py                 # Gemini API 클라이언트
├── scienceon_api_example.py         # ScienceON API 클라이언트
├── submission_pipeline_modular.py   # 메인 파이프라인
├── test.csv                         # 테스트 데이터
├── requirements.txt                 # 의존성 패키지
└── README.md                        # 이 파일
```

## 🚀 설치 및 실행

### 1. 의존성 설치
```bash
pip install -r requirements.txt
```

### 2. API 인증 설정
`configs/` 폴더에 다음 파일들을 생성하세요:

**configs/scienceon_api_credentials.json:**
```json
{
  "auth_key": "YOUR_32_CHARACTER_AUTH_KEY_HERE",
  "client_id": "YOUR_CLIENT_ID_HERE",
  "mac_address": "YOUR_MAC_ADDRESS_HERE"
}
```

**configs/gemini_api_credentials.json:**
```json
{
  "api_key": "YOUR_GEMINI_API_KEY_HERE"
}
```

### 3. 파이프라인 실행
```bash
python submission_pipeline_modular.py
```

## 🔧 주요 기능

### 📚 문서 관리
- **벡터 데이터베이스**: ChromaDB를 사용한 문서 임베딩 저장
- **메타데이터 관리**: SQLite를 사용한 문서 메타데이터 관리
- **중복 제거**: 해시 기반 중복 문서 제거

### 🔍 검색 시스템
- **키워드 검색**: ScienceON API를 통한 키워드 기반 검색
- **의미적 검색**: 벡터 유사도를 통한 의미적 검색
- **하이브리드 검색**: 키워드와 의미적 검색의 조합

### 🎯 답변 생성
- **Gemini API**: Google Gemini를 사용한 답변 생성
- **컨텍스트 관리**: 검색된 문서를 기반으로 한 컨텍스트 생성
- **프롬프트 엔지니어링**: 효과적인 프롬프트 설계

### 🔄 문서 재순위화
- **다양성 필터링**: 중복 문서 제거
- **관련성 점수**: 문서-질문 관련성 계산
- **최적화된 선택**: 상위 관련 문서 선택

## ⚙️ 설정

`modules/config.py`에서 다음 설정을 조정할 수 있습니다:

- **검색 설정**: 최소/최대 문서 수, 재시도 횟수
- **답변 설정**: 최소 답변 길이, 최대 컨텍스트 문서 수
- **테스트 설정**: 처리할 질문 수, 디버그 모드

## 📊 출력

파이프라인 실행 후 다음 파일들이 생성됩니다:

- `../submissions/submission_modular_v2_YYYYMMDD_HHMMSS.csv`: 답변 결과
- `../submissions/submission_modular_v2_YYYYMMDD_HHMMSS.md`: 상세 리포트
- `./outputs/elapsed_times.json`: 처리 시간 통계

## 🛠️ 개발

### 모듈 구조
각 모듈은 독립적으로 설계되어 있어 개별적으로 테스트하고 수정할 수 있습니다.

### 확장성
- 새로운 검색 도구 추가 가능
- 새로운 키워드 추출기 추가 가능
- 새로운 검색 방법 추가 가능

## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 🤝 기여

버그 리포트나 기능 제안은 GitHub Issues를 통해 해주세요.

---

**SAI Challenge 2024** - RAG Pipeline for Scientific Question Answering