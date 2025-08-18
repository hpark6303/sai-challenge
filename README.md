# 🚀 ScienceON AI (SAI) Challenge - 모듈화 RAG 파이프라인

> **Kaggle 대회**: ScienceON AI (SAI) Challenge  
> **목표**: 학술 논문 기반 질의응답 시스템 구축  
> **최종 성과**: 모듈화된 RAG 파이프라인으로 실제 50개 문서 보장

## 📋 목차

- [프로젝트 개요](#-프로젝트-개요)
- [🏗️ 아키텍처](#️-아키텍처)
- [📦 모듈 구조](#-모듈-구조)
- [🚀 주요 기능](#-주요-기능)
- [📊 성과](#-성과)
- [🛠️ 설치 및 실행](#️-설치-및-실행)
- [📁 프로젝트 구조](#-프로젝트-구조)
- [🔧 기술 스택](#-기술-스택)
- [📈 개발 과정](#-개발-과정)

## 🎯 프로젝트 개요

ScienceON AI Challenge에서 **모듈화된 RAG(Retrieval-Augmented Generation) 파이프라인**을 구축하여 학술 논문 기반 질의응답 시스템을 개발했습니다.

### 🎖️ 핵심 성과
- ✅ **실제 50개 문서 보장**: placeholder 없이 진짜 검색 결과만 사용
- ✅ **벡터 검색 최적화**: 개선된 유사도 계산으로 검색 품질 향상
- ✅ **모듈화 설계**: 유지보수성과 확장성 극대화
- ✅ **100% 성공률**: 50개 질문 모두 성공적으로 처리

## 🏗️ 아키텍처

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   사용자 질문   │───▶│  키워드 추출    │───▶│  문서 검색      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   답변 생성     │◀───│  프롬프트 엔지니어링 │◀───│  문서 재순위화   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                        │                        │
        ▼                        ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  벡터 DB 저장   │    │  컨텍스트 강화   │    │  유사도 검색    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📦 모듈 구조

### 🔧 핵심 모듈 (8개)

| 모듈 | 파일 | 역할 | 주요 클래스 |
|------|------|------|-------------|
| **설정 관리** | `config.py` | 하이퍼파라미터 중앙 관리 | 설정 딕셔너리들 |
| **벡터 DB** | `vector_db.py` | ChromaDB 기반 벡터 검색 | `VectorDatabase` |
| **문서 검색** | `retrieval.py` | 키워드 추출 및 검색 | `DocumentRetriever` |
| **재순위화** | `reranking.py` | 다중 기준 문서 재순위화 | `DocumentReranker` |
| **프롬프트** | `prompting.py` | 고품질 프롬프트 생성 | `PromptEngineer` |
| **답변 생성** | `answer_generator.py` | Gemini API 답변 생성 | `AnswerGenerator` |
| **메인 파이프라인** | `rag_pipeline.py` | 전체 워크플로우 관리 | `RAGPipeline` |
| **초기화** | `__init__.py` | 모듈 패키지 초기화 | 클래스 export |

### 🔄 워크플로우

1. **🔍 키워드 추출**: 한국어(명사), 영어(불용어 제거 + 전문용어 우선)
2. **📚 문서 검색**: ScienceON API + 재시도 로직으로 50개 문서 보장
3. **🗄️ 벡터 DB 저장**: SentenceTransformer 임베딩 생성 및 ChromaDB 저장
4. **🔍 유사도 검색**: 개선된 유사도 계산 (`1.0 / (1.0 + distance)`)
5. **📊 재순위화**: TF-IDF, 키워드 매칭, 품질, 컨텍스트 다중 기준
6. **📝 프롬프트 생성**: 전문가 수준 분석 보고서용 고품질 프롬프트
7. **🤖 답변 생성**: Gemini API + 품질 검증 + 재시도 로직
8. **📋 결과 변환**: Kaggle 제출 형식으로 변환

## 🚀 주요 기능

### 🎯 실제 50개 문서 보장
- **적극적인 키워드 확장**: 관련 용어 매핑, 학술 용어 추가
- **긴급 검색 시스템**: 50개 미만 시 자동 추가 검색
- **품질 필터링**: 중복 제거 및 품질 기준 적용

### 🔍 벡터 검색 최적화
- **개선된 유사도 계산**: `1.0 / (1.0 + distance)` 공식 적용
- **ChromaDB 연동**: 영구 저장소 기반 벡터 DB
- **임베딩 모델**: `sentence-transformers/all-MiniLM-L6-v2`

### 📊 고급 재순위화
- **다중 기준 점수**: TF-IDF(30%) + 키워드 매칭(25%) + 제목 관련성(20%) + 품질(15%) + 컨텍스트(10%)
- **다양성 필터링**: 중복 내용 제거 및 다양한 관점 보장
- **도메인 최적화**: 학술 논문 특성에 맞는 점수 계산

### 🤖 고품질 답변 생성
- **전문가 수준 프롬프트**: 제목, 서론, 본론, 결론 구조화
- **언어별 최적화**: 한국어/영어에 따른 프롬프트 조정
- **품질 검증**: 답변 길이, 내용 품질 확인 및 재시도

## 📊 성과

### 🏆 최종 성과
- **총 소요 시간**: 636초 (약 10분 36초)
- **평균 처리 시간**: 12.72초/질문
- **성공률**: 50/50 (100%)
- **벡터 DB 문서 수**: 384개
- **실제 문서 보장**: 50개/질문 (placeholder 없음)

### 🔧 기술적 성과
- **벡터 검색 문제 해결**: 음수 유사도 → 양수 유사도 정상화
- **모듈화 완성**: 8개 독립적 모듈로 분리
- **설정 중앙화**: 모든 하이퍼파라미터 통합 관리
- **오류 처리 강화**: 각 단계별 예외 처리 및 fallback

## 🛠️ 설치 및 실행

### 1. 환경 설정
```bash
# Conda 환경 생성 및 활성화
conda create -n llm2024 python=3.10
conda activate llm2024

# 의존성 설치 (루트 디렉토리에서)
pip install -r requirements.txt
```

### 2. API 설정
```bash
# ScienceON API 설정
cp configs/api_config_template.json configs/api_config.json
# api_config.json에 실제 API 키 입력
```

### 3. 파이프라인 실행
```bash
cd submission/final_pipeline
python submission_pipeline_modular.py
```

### 4. Kaggle 제출
```bash
kaggle competitions submit -c sai-challenge -f ../submissions/submission_modular_v2_YYYYMMDD_HHMMSS.csv -m "Modular RAG v2 with improved vector search"
```

## 📁 프로젝트 구조

```
sai-challenge/
├── 📁 submission/
│   ├── 📁 final_pipeline/          # 메인 파이프라인
│   │   ├── 📁 modules/             # 모듈화된 컴포넌트
│   │   │   ├── config.py           # 설정 관리
│   │   │   ├── vector_db.py        # 벡터 DB
│   │   │   ├── retrieval.py        # 문서 검색
│   │   │   ├── reranking.py        # 재순위화
│   │   │   ├── prompting.py        # 프롬프트 엔지니어링
│   │   │   ├── answer_generator.py # 답변 생성
│   │   │   ├── rag_pipeline.py     # 메인 파이프라인
│   │   │   └── __init__.py         # 모듈 초기화
│   │   └── submission_pipeline_modular.py  # 실행 스크립트
│   └── 📁 submissions/             # 제출 파일들
│       ├── submission_modular_v2_*.csv
│       └── submission_modular_v2_*.md
├── 📁 rdgenai-api-sample/          # 원본 샘플 코드
├── 📁 configs/                     # API 설정
├── README.md                       # 프로젝트 문서
├── DEVELOPMENT.md                  # 개발 과정
├── requirements.txt                # 메인 의존성
└── LICENSE                         # MIT 라이선스
```

## 🔧 기술 스택

### 🤖 AI/ML
- **임베딩 모델**: `sentence-transformers/all-MiniLM-L6-v2`
- **벡터 DB**: ChromaDB
- **LLM**: Google Gemini API
- **한국어 NLP**: KoNLPy (Okt)

### 🐍 Python 라이브러리
- **벡터 검색**: `chromadb`, `sentence-transformers`
- **텍스트 처리**: `konlpy`, `sklearn`
- **API 통신**: `requests`
- **데이터 처리**: `pandas`, `numpy`

### 🛠️ 개발 도구
- **환경 관리**: Conda
- **버전 관리**: Git
- **대회 플랫폼**: Kaggle

## 📈 개발 과정

### 🔄 주요 개선사항

1. **벡터 검색 문제 해결**
   - 문제: 음수 유사도로 인한 검색 결과 부족
   - 해결: `1.0 / (1.0 + distance)` 공식으로 유사도 정규화

2. **실제 50개 문서 보장**
   - 문제: placeholder로 채우는 방식
   - 해결: 적극적인 키워드 확장 + 긴급 검색 시스템

3. **모듈화 완성**
   - 문제: 단일 파일의 복잡한 구조
   - 해결: 8개 독립적 모듈로 분리

4. **설정 중앙화**
   - 문제: 하드코딩된 하이퍼파라미터
   - 해결: config.py에서 통합 관리

### 🎯 핵심 도전과제 해결

- **검색 품질**: 다중 기준 재순위화로 관련성 향상
- **답변 품질**: 고품질 프롬프트 + 품질 검증
- **안정성**: 각 단계별 예외 처리 및 fallback
- **성능**: 배치 처리 및 최적화된 API 호출

---

## 📄 라이선스

이 프로젝트는 ScienceON AI Challenge를 위한 교육 목적의 코드입니다.

## 👥 기여자

- **개발**: 모듈화 RAG 파이프라인 설계 및 구현
- **최적화**: 벡터 검색 및 문서 재순위화 알고리즘
- **테스트**: 50개 질문 전체 테스트 및 검증

---

**🏆 ScienceON AI Challenge - 모듈화 RAG 파이프라인으로 실제 50개 문서 보장 달성!**
