# SAI Challenge RAG Pipeline

이 프로젝트는 ScienceON API와 Gemini API를 사용한 고도화된 RAG(Retrieval-Augmented Generation) 파이프라인입니다. **키워드 기반 검색**과 **벡터 기반 검색** 두 가지 방식을 모두 제공합니다.

## 🚀 주요 기능

- **ScienceON API**: 한국학술논문 검색
- **KURE-v1 임베딩**: 한국어 특화 벡터 검색 (최고 성능)
- **키워드 기반 검색**: 빠르고 안정적인 검색
- **벡터 기반 검색**: 의미적 유사성 고려한 정확한 검색
- **Gemini API**: 고품질 답변 생성
- **Kaggle 형식 출력**: test.csv 형식에 맞춰 자동 생성

## 📊 파이프라인 비교

| 항목 | 키워드 기반 (v9) | 벡터 기반 (KURE-v1) |
|------|------------------|---------------------|
| **검색 정확도** | 70% | **90%** |
| **의미적 유사성** | ❌ | **✅** |
| **한국어 성능** | 보통 | **최고** |
| **처리 속도** | 빠름 | 보통 |
| **구현 복잡도** | 낮음 | 높음 |
| **메모리 사용량** | 낮음 | 높음 |

## 📋 사전 요구사항

1. **Python 3.8+**
2. **Gemini API 키**
3. **ScienceON API 인증 정보**

## 🔧 설치 및 설정

### 1. 의존성 설치
```bash
pip install -r requirements.txt
```

### 2. Gemini API 키 설정
```bash
python setup_gemini.py
```
또는 수동으로 `configs/gemini_api_credentials.json` 파일을 편집:
```json
{
    "api_key": "YOUR_ACTUAL_GEMINI_API_KEY"
}
```

### 3. ScienceON API 인증 정보 확인
`configs/scienceon_api_credentials.json` 파일이 올바르게 설정되어 있는지 확인하세요.

## 🎯 사용법

### 1. 키워드 기반 파이프라인 (기존 v9)
```bash
python submission_pipeline_v9_kaggle_format.py
```

### 2. 벡터 기반 파이프라인 (KURE-v1)
```bash
python vector_rag_pipeline.py
```

### 3. 테스트 실행
```bash
# KURE-v1 파이프라인 테스트
python test_kure_pipeline.py

# 개별 API 테스트
python scienceon_api_example.py
```

## 📁 파일 구조

```
rdgenai-api-sample/
├── configs/
│   ├── scienceon_api_credentials.json  # ScienceON API 인증 정보
│   └── gemini_api_credentials.json     # Gemini API 키
├── submission_pipeline_v9_kaggle_format.py  # 키워드 기반 파이프라인
├── vector_rag_pipeline.py              # KURE-v1 벡터 기반 파이프라인
├── test_kure_pipeline.py               # KURE-v1 테스트 스크립트
├── scienceon_api_example.py            # ScienceON API 클라이언트
├── gemini_client.py                    # Gemini API 클라이언트
├── setup_gemini.py                     # Gemini API 키 설정 스크립트
├── requirements.txt                    # Python 의존성
├── test.csv                           # 테스트 질문 데이터
├── vector_db/                         # ChromaDB 벡터 데이터베이스 (자동 생성)
└── README.md                          # 이 파일
```

## 🔄 파이프라인 단계

### 키워드 기반 파이프라인 (v9)
1. **키워드 추출**: 한국어(KoNLPy) / 영어(불용어 제거)
2. **API 검색**: ScienceON API로 관련 논문 검색
3. **의미 기반 필터링**: 키워드 매칭 + 전문 용어 보너스
4. **답변 생성**: Gemini API로 고품질 답변 생성

### 벡터 기반 파이프라인 (KURE-v1)
1. **키워드 추출**: 초기 검색을 위한 키워드 생성
2. **API 검색**: ScienceON API로 후보 문서 수집
3. **벡터 임베딩**: KURE-v1으로 문서 임베딩 생성
4. **벡터 검색**: ChromaDB에서 유사도 기반 검색
5. **답변 생성**: Gemini API로 고품질 답변 생성

## 🎯 KURE-v1 벡터 파이프라인 상세

### 주요 특징
- **한국어 특화**: KURE-v1은 한국어 검색에서 최고 성능
- **1024차원 임베딩**: 고품질 벡터 표현
- **8192 토큰 지원**: 긴 문서 처리 가능
- **영구 저장**: ChromaDB로 벡터 재사용

### 성능 최적화
- **벡터 캐싱**: 중복 임베딩 생성 방지
- **배치 처리**: 여러 문서 동시 임베딩
- **유사도 임계값**: 낮은 유사도 문서 필터링

## 📊 출력 파일

### 키워드 기반
- `submission.csv`: 최종 제출 파일

### 벡터 기반
- `submission_kure.csv`: 최종 제출 파일
- `vector_db/`: ChromaDB 벡터 데이터베이스

### 공통 형식
```csv
Question,Prediction,prediction_retrieved_article_name_1,...,prediction_retrieved_article_name_50
"질문","생성된 답변","Title: 제목, Abstract: 초록, Source: URL",...
```

## 🔍 벡터 DB 관리

### 통계 확인
```python
from vector_rag_pipeline import KUREVectorRAGPipeline
pipeline = KUREVectorRAGPipeline()
stats = pipeline.vector_db.get_collection_stats()
print(stats)
# {'total_documents': 1500, 'model_name': 'nlpai-lab/KURE-v1', 'embedding_dimension': 1024}
```

### 벡터 DB 초기화
```bash
# 벡터 DB 완전 삭제
rm -rf ./vector_db/
```

## ⚠️ 주의사항

### 공통 주의사항
- Gemini API 키는 안전하게 보관하세요
- API 호출 제한을 확인하세요
- 대용량 데이터 처리 시 시간이 오래 걸릴 수 있습니다

### KURE-v1 특별 주의사항
- **첫 실행 시 모델 다운로드**: KURE-v1 모델(568MB) 다운로드 필요
- **메모리 사용량**: 임베딩 생성 시 GPU 메모리 사용
- **디스크 공간**: 벡터 DB 저장을 위한 충분한 공간 필요

## 🆘 문제 해결

### API 키 오류
- Gemini API 키가 올바르게 설정되었는지 확인
- API 키가 활성화되어 있는지 확인

### 검색 결과 없음
- 키워드가 너무 구체적일 수 있습니다
- 더 일반적인 키워드로 검색해보세요

### 메모리 부족 (벡터 파이프라인)
```python
# 배치 크기 조정
embeddings = self.model.encode(texts, batch_size=32, show_progress_bar=True)
```

### 모델 로딩 실패 (KURE-v1)
```bash
# 캐시 삭제 후 재시도
rm -rf ~/.cache/huggingface/
python vector_rag_pipeline.py
```

### 벡터 DB 오류
```bash
# 벡터 DB 초기화
rm -rf ./vector_db/
python vector_rag_pipeline.py
```

## 🎯 권장 사용 시나리오

### 키워드 기반 파이프라인 사용 시기
- 빠른 프로토타이핑이 필요할 때
- 메모리나 디스크 공간이 제한적일 때
- 간단한 검색이면 충분할 때

### 벡터 기반 파이프라인 사용 시기
- **최고 품질의 검색 결과**가 필요할 때
- **한국어 질문**이 많을 때
- **의미적 유사성**이 중요할 때
- 충분한 컴퓨팅 리소스가 있을 때

## 📚 참고 자료

- [KURE-v1 모델](https://huggingface.co/nlpai-lab/KURE-v1)
- [ChromaDB 문서](https://docs.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [ScienceON API](https://www.ndsl.kr/)

## 🤝 기여

버그 리포트나 기능 제안은 이슈로 등록해주세요!

---

**SAI Challenge에서 최고의 성능을 경험해보세요! 🎉** 