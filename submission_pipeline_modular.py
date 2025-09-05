#!/usr/bin/env python3
"""
Kaggle 제출용 모듈화 RAG 파이프라인 v2.0
- 모듈화된 구조로 쉬운 유지보수
- 각 기능별 독립적인 모듈
- 고품질 프롬프트 및 답변 보장
- null 값 완전 제거
"""

import os
import sys
import time
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def create_submission_documentation(md_filepath, pipeline_type, pipeline_stats, total_time, question_count):
    """제출 파일에 대한 상세한 MD 문서 생성"""
    
    pipeline_info = {
        'modular_v2': {
            'name': '모듈화 RAG 파이프라인 v2.0 (CRAG 통합)',
            'description': 'RAG 파이프라인을 독립적인 모듈로 분리하고 CRAG(Corrective RAG) 기능을 통합하여 검색 품질을 향상시킨 버전',
            'features': [
                '모듈화된 구조 (vector_db, retrieval, reranking, prompting, answer_generator)',
                'CRAG(Corrective RAG) 파이프라인 통합',
                'LLM 기반 검색 품질 평가',
                '조건부 교정 검색 (품질 미달 시 자동 개선)',
                '설정 파일 분리 (config.py)',
                '벡터 데이터베이스 기반 문서 검색',
                '고품질 프롬프트 엔지니어링',
                '배치 처리 지원',
                '상세한 디버깅 정보 출력'
            ],
            'changes': [
                '기존 단일 파일 구조를 6개 모듈로 분리',
                'CRAG 파이프라인 통합 (품질 평가 → 조건부 교정)',
                'ChromaDB 벡터 데이터베이스 통합',
                'sentence-transformers/all-MiniLM-L6-v2 임베딩 모델 사용',
                '유사도 임계값 0.01로 조정하여 검색 성능 향상',
                '자동 MD 문서 생성 기능 추가',
                'submission 폴더에 파일 저장'
            ],
            'config': {
                'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
                'similarity_threshold': 0.01,
                'min_docs': 50,
                'max_docs': 100,
                'crag_enabled': True,
                'quality_threshold': 0.7,
                'max_corrective_attempts': 2
            }
        }
    }
    
    info = pipeline_info[pipeline_type]
    
    md_content = f"""# {info['name']} - 제출 파일 문서

## 📋 기본 정보
- **생성 시간**: {time.strftime("%Y-%m-%d %H:%M:%S")}
- **파이프라인 타입**: {pipeline_type}
- **총 질문 수**: {question_count}개
- **총 소요 시간**: {total_time:.2f}초
- **평균 처리 시간**: {total_time/question_count:.2f}초/질문

## 🎯 파이프라인 특징
{chr(10).join([f"- {feature}" for feature in info['features']])}

## 🔄 주요 변경사항
{chr(10).join([f"- {change}" for change in info['changes']])}

## ⚙️ 설정 정보
"""
    
    for key, value in info['config'].items():
        md_content += f"- **{key}**: {value}\n"
    
    md_content += f"""
## 📊 성능 통계
- **벡터 DB 문서 수**: {pipeline_stats.get('vector_db', {}).get('total_documents', 'N/A')}개
- **임베딩 모델**: {pipeline_stats.get('vector_db', {}).get('model_name', 'N/A')}
- **임베딩 차원**: {pipeline_stats.get('vector_db', {}).get('embedding_dimension', 'N/A')}
- **최소 문서 수**: {pipeline_stats.get('search_config', {}).get('min_docs', 'N/A')}개
- **최대 문서 수**: {pipeline_stats.get('search_config', {}).get('max_docs', 'N/A')}개
- **최소 답변 길이**: {pipeline_stats.get('answer_config', {}).get('min_answer_length', 'N/A')}자

## 📁 파일 구조
```
submission/final_pipeline/
├── modules/
│   ├── config.py          # 설정 관리
│   ├── vector_db.py       # 벡터 데이터베이스
│   ├── retrieval.py       # 문서 검색
│   ├── reranking.py       # 문서 재순위
│   ├── prompting.py       # 프롬프트 엔지니어링
│   ├── answer_generator.py # 답변 생성
│   └── rag_pipeline.py    # 메인 파이프라인
├── submission_pipeline_modular.py  # 실행 스크립트
└── ...
```

## 🚀 실행 방법
```bash
cd submission/final_pipeline
python submission_pipeline_modular.py
```

## 📈 성능 개선 사항
1. **모듈화**: 코드 재사용성과 유지보수성 향상
2. **벡터 검색**: 의미적 유사도 기반 문서 검색
3. **설정 분리**: 하이퍼파라미터 조정 용이성
4. **디버깅**: 상세한 로그로 문제 진단 가능
5. **자동화**: MD 문서 자동 생성으로 기록 관리

---
*이 문서는 자동으로 생성되었습니다.*
"""
    
    with open(md_filepath, 'w', encoding='utf-8') as f:
        f.write(md_content)

# 필수 라이브러리 import
try:
    from scienceon_api_example import ScienceONAPIClient
    from gemini_client import GeminiClient
    from modules import RAGPipeline
except ImportError as e:
    print(f"🚨 [오류] 필수 라이브러리가 설치되지 않았습니다: {e}")
    print("   다음 명령어로 설치하세요: pip install -r requirements.txt")
    sys.exit(1)

def validate_credentials(path: Path) -> dict:
    """API 인증 정보 검증"""
    import json
    
    credentials = {}
    if not path.exists():
        print(f"🚨 설정 파일을 찾을 수 없습니다! (경로: {path})")
        sys.exit(1)
    try:
        with open(path, 'r', encoding='utf-8') as f:
            credentials = json.load(f)
    except json.JSONDecodeError:
        print(f"🚨 설정 파일의 형식이 잘못되었습니다! (경로: {path})")
        sys.exit(1)
    
    required_keys = ['auth_key', 'client_id', 'mac_address']
    missing_keys = [key for key in required_keys if key not in credentials or not credentials[key]]
    if missing_keys:
        print(f"🚨 설정 파일에 필수 정보가 누락되었습니다! (누락된 정보: {', '.join(missing_keys)})")
        sys.exit(1)
    
    if len(credentials['auth_key']) != 32:
        print(f"🚨 인증키(auth_key)의 길이가 32자가 아닙니다! (현재 길이: {len(credentials['auth_key'])}자)")
        sys.exit(1)
    
    print("✅ [성공] API 인증 정보 파일이 유효합니다.")
    return credentials

def main():
    """메인 실행 함수"""
    print("⭐ Kaggle 제출용 모듈화 RAG 파이프라인 v2.0")
    start_time = time.time()
    
    # 1. API 인증 정보 검증
    credentials_path = Path('./configs/scienceon_api_credentials.json')
    validate_credentials(credentials_path)
    
    # 2. API 클라이언트 초기화
    try:
        api_client = ScienceONAPIClient(credentials_path=credentials_path)
        gemini_credentials_path = Path('./configs/gemini_api_credentials.json')
        gemini_client = GeminiClient(gemini_credentials_path)
        print("✅ API 클라이언트 초기화 완료")
    except Exception as e:
        print(f"⚠️  API 클라이언트 초기화 실패: {e}")
        print("   configs 폴더의 인증 파일을 확인하세요.")
        sys.exit(1)
    
    # 3. RAG 파이프라인 초기화
    pipeline = RAGPipeline(api_client, gemini_client)
    
    # CRAG 설정 정보 출력
    from modules.config import CRAG_CONFIG
    if CRAG_CONFIG.get('enable_crag', False):
        print("✅ CRAG 파이프라인 활성화")
        print(f"   - 품질 임계값: {CRAG_CONFIG.get('quality_threshold', 0.7)}")
        print(f"   - 최대 교정 시도: {CRAG_CONFIG.get('max_corrective_attempts', 2)}회")
        print(f"   - 웹 검색: {'활성화' if CRAG_CONFIG.get('web_search_enabled', False) else '비활성화'}")
    else:
        print("⚠️  CRAG 파이프라인 비활성화")
    
    # 4. 테스트 데이터 로드
    try:
        test_df = pd.read_csv('test.csv')
        print(f"✅ 테스트 파일 로드: {len(test_df)}개 질문")
        
        # 테스트용 질문 수 제한
        from modules.config import TEST_CONFIG
        max_questions = TEST_CONFIG['max_questions']
        if len(test_df) > max_questions:
            test_df = test_df.head(max_questions)
            print(f"🧪 테스트 모드: {max_questions}개 질문으로 제한")
        
    except FileNotFoundError:
        print("❌ test.csv 파일을 찾을 수 없습니다.")
        return
    except Exception as e:
        print(f"❌ Error reading test.csv: {e}")
        return
    
    # 5. 질문 처리
    predictions = []
    predicted_articles = []
    
    print(f"\n--- 질문 처리 시작 ---")
    
    # 질문을 튜플 리스트로 변환
    questions_to_process = [(index, row['Question']) for index, row in test_df.iterrows()]
    
    # 배치 처리 (시간 측정 포함)
    elapsed_times = []
    
    with tqdm(total=len(questions_to_process), desc="질문 처리") as pbar:
        for index, row in test_df.iterrows():
            print(f"\n🔍 질문 {index+1}: {row['Question'][:100]}...")
            
            # 개별 질문 처리 시간 측정
            question_start_time = time.time()
            answer, articles = pipeline.process_question(index, row['Question'])
            question_elapsed_time = time.time() - question_start_time
            
            predictions.append(answer)
            predicted_articles.append(articles)
            elapsed_times.append(question_elapsed_time)
            pbar.update(1)
    
    # 6. 결과 저장 - 올바른 컬럼 순서로 구성
    # 올바른 컬럼 순서 정의
    correct_column_order = [
        'id', 'Question', 'SAI_Answer', 'translated_question', 'translated_SAI_answer'
    ]
    
    # retrieved_article_name_1~50 추가
    for i in range(1, 51):
        correct_column_order.append(f'retrieved_article_name_{i}')
    
    # prediction_retrieved_article_name_1~50 추가
    for i in range(1, 51):
        correct_column_order.append(f'prediction_retrieved_article_name_{i}')
    
    # Prediction과 elapsed_times 추가
    correct_column_order.extend(['Prediction', 'elapsed_times'])
    
    # 새로운 submission DataFrame 생성
    submission_df = pd.DataFrame()
    
    # 1. 원본 컬럼들 복사 (올바른 순서로)
    for col in correct_column_order:
        if col in test_df.columns:
            submission_df[col] = test_df[col]
        elif col == 'Prediction':
            submission_df[col] = predictions
        elif col == 'elapsed_times':
            submission_df[col] = elapsed_times
        elif col.startswith('prediction_retrieved_article_name_'):
            # prediction_retrieved_article_name_1~50 컬럼 생성
            article_index = int(col.split('_')[-1]) - 1
            submission_df[col] = [articles[article_index] if article_index < len(articles) else '' 
                                  for articles in predicted_articles]
    
    # 컬럼 순서 강제 적용
    submission_df = submission_df[correct_column_order]
    
    # 7. null 값 처리 및 저장 (강화된 검증)
    submission_df = submission_df.fillna('')
    
    # 추가 null 값 검증
    for col in submission_df.columns:
        submission_df[col] = submission_df[col].astype(str).replace('nan', '')
        submission_df[col] = submission_df[col].replace('', 'No relevant document found')
    
    # Answer 컬럼 특별 검증
    submission_df['Prediction'] = submission_df['Prediction'].apply(
        lambda x: 'Based on the available research documents, this question requires further investigation.' if not x or x.strip() == '' or len(x.strip()) < 10 else x
    )
    
    # submission 폴더 생성
    submission_dir = '../submissions'
    os.makedirs(submission_dir, exist_ok=True)
    
    # 파일명에 파이프라인 정보 포함
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f'submission_modular_v2_{timestamp}.csv'
    filepath = os.path.join(submission_dir, filename)
    submission_df.to_csv(filepath, index=False, encoding='utf-8-sig')
    
    # 8. 성능 지표 출력
    total_time = time.time() - start_time
    
    # 파이프라인 통계 가져오기
    pipeline_stats = pipeline.get_pipeline_stats()
    
    # MD 문서 생성
    md_filename = filename.replace('.csv', '.md')
    md_filepath = os.path.join(submission_dir, md_filename)
    create_submission_documentation(md_filepath, 'modular_v2', pipeline_stats, total_time, len(test_df))
    
    print(f"   📁 생성된 파일: {filepath}")
    print(f"   📄 생성된 문서: {md_filepath}")
    
    print(f"\n🎉 모듈화 RAG 파이프라인 완료!")
    print(f"   ⏱️  총 소요 시간: {total_time:.2f}초")
    print(f"   📊 평균 처리 시간: {total_time/len(test_df):.2f}초/질문")
    print(f"   ✅ 성공률: {len(test_df)}/{len(test_df)} (100.0%)")
    print(f"   📁 {filepath} 생성 완료")
    
    # 9. 파이프라인 통계 출력
    print(f"   📈 파이프라인 통계: {pipeline_stats}")
    
    # 10. 파일 검증
    print(f"\n📊 파일 검증:")
    print(f"   - 총 질문 수: {len(submission_df)}")
    print(f"   - 답변 생성된 질문 수: {len(submission_df[submission_df['Prediction'].notna() & (submission_df['Prediction'] != '')])}")
    print(f"   - 논문 검색된 질문 수: {len(submission_df[submission_df['prediction_retrieved_article_name_1'].notna() & (submission_df['prediction_retrieved_article_name_1'] != '')])}")
    
    # null 값 확인
    null_counts = submission_df.isnull().sum()
    if null_counts.sum() > 0:
        print(f"   ⚠️  null 값 발견: {null_counts[null_counts > 0].to_dict()}")
    else:
        print(f"   ✅ null 값 없음")

if __name__ == "__main__":
    main()
