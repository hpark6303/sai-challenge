# SAI Challenge RAG Pipeline

이 프로젝트는 ScienceON API와 Gemini API를 사용한 고도화된 RAG(Retrieval-Augmented Generation) 파이프라인입니다.

## 🚀 주요 기능

- **ScienceON API**: 한국학술논문 검색
- **Semantic Search**: 의미 기반 문서 검색
- **Cross-Encoder Re-ranking**: 정밀한 문서 재정렬
- **Gemini API**: 고품질 답변 생성

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

### 전체 파이프라인 실행
```bash
python submission_pipeline.py
```

### 개별 API 테스트
```bash
# ScienceON API 테스트
python scienceon_api_example.py

# Gemini API 테스트
python -c "from gemini_client import GeminiClient; from pathlib import Path; client = GeminiClient(Path('./configs/gemini_api_credentials.json')); print(client.generate_answer('안녕하세요!'))"
```

## 📁 파일 구조

```
rdgenai-api-sample/
├── configs/
│   ├── scienceon_api_credentials.json  # ScienceON API 인증 정보
│   └── gemini_api_credentials.json     # Gemini API 키
├── submission_pipeline.py              # 메인 파이프라인
├── scienceon_api_example.py            # ScienceON API 클라이언트
├── gemini_client.py                    # Gemini API 클라이언트
├── setup_gemini.py                     # Gemini API 키 설정 스크립트
├── requirements.txt                    # Python 의존성
├── test.csv                           # 테스트 질문 데이터
└── README.md                          # 이 파일
```

## 🔄 파이프라인 단계

1. **문서 검색**: ScienceON API로 관련 논문 검색
2. **의미 기반 필터링**: Bi-Encoder로 의미적으로 유사한 문서 선택
3. **정밀 재정렬**: Cross-Encoder로 최적의 문서 선택
4. **답변 생성**: Gemini API로 고품질 답변 생성

## 📊 출력 파일

- `submission_advanced.csv`: 최종 제출 파일
- `outputs/elapsed_times.json`: 처리 시간 기록

## ⚠️ 주의사항

- Gemini API 키는 안전하게 보관하세요
- API 호출 제한을 확인하세요
- 대용량 데이터 처리 시 시간이 오래 걸릴 수 있습니다

## 🆘 문제 해결

### API 키 오류
- Gemini API 키가 올바르게 설정되었는지 확인
- API 키가 활성화되어 있는지 확인

### 검색 결과 없음
- 키워드가 너무 구체적일 수 있습니다
- 더 일반적인 키워드로 검색해보세요

### 메모리 부족
- 더 작은 배치 크기로 실행
- CPU 모드로 전환 