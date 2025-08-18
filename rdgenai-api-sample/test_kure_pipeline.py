#!/usr/bin/env python3
"""
KURE-v1 벡터 RAG 파이프라인 테스트 스크립트
"""

import pandas as pd
from vector_rag_pipeline import KUREVectorRAGPipeline, KUREVectorDB

def test_vector_db():
    """벡터 DB 기능 테스트"""
    print("🧪 벡터 DB 테스트 시작")
    
    # 벡터 DB 초기화
    vector_db = KUREVectorDB()
    
    # 테스트 문서
    test_docs = [
        {
            'CN': 'test_001',
            'title': '인공지능과 머신러닝의 차이점',
            'abstract': '인공지능은 인간의 지능을 모방하는 기술이며, 머신러닝은 인공지능의 한 분야입니다.'
        },
        {
            'CN': 'test_002', 
            'title': '딥러닝과 신경망',
            'abstract': '딥러닝은 다층 신경망을 사용하여 복잡한 패턴을 학습하는 기술입니다.'
        },
        {
            'CN': 'test_003',
            'title': '자연어처리 기술',
            'abstract': '자연어처리는 인간의 언어를 컴퓨터가 이해하고 처리하는 기술입니다.'
        }
    ]
    
    # 문서 추가
    added_count = vector_db.add_documents(test_docs)
    print(f"   - 추가된 문서 수: {added_count}")
    
    # 검색 테스트
    query = "인공지능이란 무엇인가요?"
    results = vector_db.search_similar(query, n_results=3)
    
    print(f"   - 검색 결과 수: {len(results)}")
    for i, doc in enumerate(results):
        print(f"   - 결과 {i+1}: {doc.get('title', '')} (유사도: {doc.get('similarity_score', 0):.3f})")
    
    # 통계 확인
    stats = vector_db.get_collection_stats()
    print(f"   - 벡터 DB 통계: {stats}")
    
    print("✅ 벡터 DB 테스트 완료\n")

def test_pipeline():
    """전체 파이프라인 테스트"""
    print("🧪 전체 파이프라인 테스트 시작")
    
    # 테스트 데이터 생성
    test_data = {
        'Question': [
            '인공지능과 머신러닝의 차이점은 무엇인가요?',
            '딥러닝이란 무엇인가요?',
            '자연어처리 기술의 최신 동향은?'
        ]
    }
    
    test_df = pd.DataFrame(test_data)
    test_df.to_csv('test_sample.csv', index=False)
    
    # 파이프라인 초기화 (실제 API 키 없이 테스트)
    try:
        pipeline = KUREVectorRAGPipeline()
        
        # 문서 검색 테스트 (API 없이 벡터 DB만 테스트)
        query = "인공지능과 머신러닝의 차이점"
        documents = pipeline.vector_db.search_similar(query, n_results=5)
        
        if documents:
            # 컨텍스트 생성 테스트
            context = pipeline.create_context(documents, max_docs=3)
            print(f"   - 컨텍스트 생성 완료 (길이: {len(context)} 문자)")
            
            # Kaggle 형식 테스트
            for i, doc in enumerate(documents[:2]):
                formatted = pipeline.create_kaggle_format_article(doc, i+1)
                print(f"   - 문서 {i+1} 포맷팅: {formatted[:100]}...")
        
        print("✅ 파이프라인 테스트 완료")
        
    except Exception as e:
        print(f"   ⚠️  파이프라인 테스트 실패 (API 키 필요): {e}")
    
    print()

def test_embedding_model():
    """KURE-v1 모델 테스트"""
    print("🧪 KURE-v1 임베딩 모델 테스트 시작")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        # 모델 로드
        model = SentenceTransformer("nlpai-lab/KURE-v1")
        print(f"   - 모델 로드 완료")
        print(f"   - 임베딩 차원: {model.get_sentence_embedding_dimension()}")
        
        # 테스트 문장들
        sentences = [
            "인공지능과 머신러닝의 차이점",
            "AI와 machine learning의 차이점", 
            "딥러닝과 신경망 기술",
            "오늘 날씨가 좋네요"  # 관련 없는 문장
        ]
        
        # 임베딩 생성
        embeddings = model.encode(sentences)
        print(f"   - 임베딩 생성 완료: {embeddings.shape}")
        
        # 유사도 계산
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(embeddings)
        
        print("   - 유사도 매트릭스:")
        for i in range(len(sentences)):
            for j in range(i+1, len(sentences)):
                sim = similarities[i][j]
                print(f"     '{sentences[i][:20]}...' vs '{sentences[j][:20]}...': {sim:.3f}")
        
        print("✅ KURE-v1 모델 테스트 완료\n")
        
    except Exception as e:
        print(f"   ❌ KURE-v1 모델 테스트 실패: {e}\n")

def main():
    """메인 테스트 함수"""
    print("🚀 KURE-v1 벡터 RAG 파이프라인 테스트 시작\n")
    
    # 1. 임베딩 모델 테스트
    test_embedding_model()
    
    # 2. 벡터 DB 테스트
    test_vector_db()
    
    # 3. 전체 파이프라인 테스트
    test_pipeline()
    
    print("🎉 모든 테스트 완료!")

if __name__ == "__main__":
    main()
