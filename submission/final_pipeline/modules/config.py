"""
설정 관리 모듈
- 모든 하이퍼파라미터와 설정값을 중앙에서 관리
"""

# 벡터 DB 설정
VECTOR_DB_CONFIG = {
    'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
    'db_path': './vector_db',
    'collection_name': 'papers',
    'similarity_threshold': 0.3,  # 개선된 유사도 임계값 (30%)
    'max_results': 100
}

# 검색 설정
SEARCH_CONFIG = {
    'min_docs': 50,
    'max_docs': 100,
    'max_retries': 5,  # 더 많은 재시도
    'api_delay': 0.3,  # 더 빠른 검색
    'batch_size': 5,
    'similarity_threshold': 0.01,  # 더 낮은 임계값으로 더 많은 결과
    'emergency_keywords': ['연구', '분석', '방법', '시스템', '기술', '개발', '최적화', '평가', '관리', '구현'],
    'hybrid_weights': {
        'keyword_weight': 0.6,  # 키워드 검색 가중치
        'vector_weight': 0.4    # 벡터 검색 가중치
    },
    'use_llm_keywords': True,   # LLM 기반 키워드 추출 사용
    'use_hybrid_search': True   # 하이브리드 검색 사용
}

# 답변 생성 설정
ANSWER_CONFIG = {
    'min_answer_length': 50,
    'max_context_docs': 8,  # 상위 3개 문서만 context에 사용 (과부하 방지)
    'max_retries': 3
}

# 프롬프트 설정
PROMPT_CONFIG = {
    'system_role': "당신은 주어진 학술 문서들을 바탕으로 질문에 대한 심층 분석 보고서를 작성하는 전문 연구원입니다.",
    'output_format': [
        "제목 (Title): 질문의 핵심 내용을 포괄하는 간결하고 전문적인 제목",
        "서론 (Introduction): 질문의 배경과 핵심 주제를 간략히 언급", 
        "본론 (Body): 참고 문서에서 찾아낸 핵심적인 사실, 데이터, 주장들을 바탕으로 구체적인 답변",
        "결론 (Conclusion): 본론의 핵심 내용을 요약하며 보고서를 마무리"
    ]
}

# 파일 설정
FILE_CONFIG = {
    'test_file': 'test.csv',
    'submission_file': 'submission.csv',
    'encoding': 'utf-8-sig',
    'filename_patterns': {
        'modular_v2': 'submission_modular_v2_{timestamp}.csv',
        'clean_v1': 'submission_clean_v1_{timestamp}.csv',
        'kure_v1': 'submission_kure_v1_{timestamp}.csv'
    }
}

# 테스트 설정
TEST_CONFIG = {
    'max_questions': 50,  # 전체 질문 테스트
    'debug_mode': True,   # 디버그 모드 활성화 (키워드 추출 과정 표시)
    'clear_vector_db': True  # 벡터 DB 초기화
}

# CRAG 설정
CRAG_CONFIG = {
    'enable_crag': True,  # CRAG 파이프라인 활성화
    'quality_threshold': 0.7,  # 품질 평가 임계값 (70% 이상이면 성공)
    'max_corrective_attempts': 2,  # 최대 교정 시도 횟수
    'web_search_enabled': True,  # 웹 검색 활성화
    'correction_prompt_template': """
    다음 검색 결과를 평가하고, 필요시 개선된 검색 키워드를 제안해주세요.
    
    원본 질문: {query}
    검색된 문서 수: {doc_count}개
    
    검색 결과 샘플:
    {sample_docs}
    
    평가 기준:
    1. 관련성: 검색 결과가 질문과 얼마나 관련이 있는가? (0-10점)
    2. 품질: 검색된 문서의 학술적 품질은 어떠한가? (0-10점)
    3. 충분성: 답변 생성에 충분한 정보가 있는가? (0-10점)
    
    평가 결과:
    - 관련성: {relevance_score}/10
    - 품질: {quality_score}/10  
    - 충분성: {sufficiency_score}/10
    - 종합 점수: {total_score}/10
    
    개선 제안:
    {improvement_suggestions}
    
    새로운 검색 키워드: {new_keywords}
    """,
    'web_search_prompt': """
    다음 질문에 대해 더 나은 검색 키워드를 찾기 위해 웹에서 관련 정보를 검색해주세요.
    
    원본 질문: {query}
    기존 검색 결과 문제점: {issues}
    
    웹 검색을 통해 다음을 찾아주세요:
    1. 관련 전문 용어나 키워드
    2. 최신 연구 동향
    3. 관련 학술 분야
    
    개선된 검색 키워드 목록:
    """
}
