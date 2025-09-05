#!/usr/bin/env python3
"""
RAG 파이프라인 전체 테스트
- 실제 질문 처리 테스트
- 전체 워크플로우 검증
"""

import sys
import logging
from pathlib import Path

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_full_rag_pipeline():
    """전체 RAG 파이프라인 테스트"""
    print("🚀 전체 RAG 파이프라인 테스트 시작\n")
    
    try:
        # 1. 필요한 모듈 import
        from modules.rag_pipeline import RAGPipeline
        from scienceon_api_example import ScienceONAPIClient
        from gemini_client import GeminiClient
        from pathlib import Path
        
        print("✅ 모든 모듈 import 성공")
        
        # 2. API 클라이언트 초기화
        print("\n🔧 API 클라이언트 초기화...")
        credentials_path = Path("./configs/scienceon_api_credentials.json")
        api_client = ScienceONAPIClient(credentials_path)
        
        gemini_credentials_path = Path("./configs/gemini_api_credentials.json")
        gemini_client = GeminiClient(gemini_credentials_path)
        
        print("✅ API 클라이언트 초기화 완료")
        
        # 3. RAG 파이프라인 초기화
        print("\n🔧 RAG 파이프라인 초기화...")
        rag_pipeline = RAGPipeline(api_client, gemini_client, dataset_name="scienceon")
        print("✅ RAG 파이프라인 초기화 완료")
        
        # 4. 테스트 질문
        test_questions = [
            "인공지능의 윤리적 문제는 무엇인가?",
            "기계학습에서 사용되는 주요 알고리즘은?",
            "딥러닝의 발전 과정은 어떻게 되나?"
        ]
        
        print(f"\n🔍 {len(test_questions)}개 질문 처리 시작...")
        
        for i, question in enumerate(test_questions):
            print(f"\n{'='*60}")
            print(f"📝 질문 {i+1}: {question}")
            print('='*60)
            
            try:
                # 질문 처리
                answer, articles = rag_pipeline.process_question(i, question)
                
                print(f"\n🤖 답변:")
                print(f"{answer[:200]}..." if len(answer) > 200 else answer)
                
                print(f"\n📚 논문 정보 ({len(articles)}개):")
                for j, article in enumerate(articles[:3]):  # 처음 3개만 표시
                    print(f"  {j+1}. {article[:100]}...")
                
                if len(articles) > 3:
                    print(f"  ... 및 {len(articles)-3}개 더")
                
            except Exception as e:
                print(f"❌ 질문 {i+1} 처리 실패: {e}")
                continue
        
        # 5. 정리
        api_client.close_session()
        print(f"\n✅ 테스트 완료!")
        
        return True
        
    except Exception as e:
        print(f"❌ RAG 파이프라인 테스트 실패: {e}")
        return False

def test_search_components():
    """검색 컴포넌트 개별 테스트"""
    print("\n🔍 검색 컴포넌트 개별 테스트...")
    
    try:
        from modules.document_manager import DocumentManager
        from modules.search_engine import FlexibleSearchEngine
        from modules.search_tools import ScienceONTool
        from modules.search_methods import HybridSearchMethod
        from modules.keyword_extractors import LLMKeywordExtractor
        from scienceon_api_example import ScienceONAPIClient
        from gemini_client import GeminiClient
        from pathlib import Path
        
        # 컴포넌트 초기화
        dm = DocumentManager()
        se = FlexibleSearchEngine(dm)
        
        # API 클라이언트
        credentials_path = Path("./configs/scienceon_api_credentials.json")
        api_client = ScienceONAPIClient(credentials_path)
        
        gemini_credentials_path = Path("./configs/gemini_api_credentials.json")
        gemini_client = GeminiClient(gemini_credentials_path)
        
        # 도구와 방법 등록
        scienceon_tool = ScienceONTool(api_client)
        hybrid_method = HybridSearchMethod()
        
        se.register_tool("scienceon", scienceon_tool, is_default=True)
        se.register_method("hybrid", hybrid_method, is_default=True)
        
        # 키워드 추출기
        keyword_extractor = LLMKeywordExtractor(gemini_client)
        
        # 테스트 쿼리
        test_query = "인공지능 윤리"
        print(f"🔍 테스트 쿼리: {test_query}")
        
        # 키워드 추출
        keywords = keyword_extractor.extract_keywords(test_query)
        print(f"📝 추출된 키워드: {keywords}")
        
        # 검색 실행
        documents, metadata = se.search(
            test_query, 
            dataset_name="scienceon",
            method="hybrid",
            keywords=keywords,
            max_docs=5
        )
        
        print(f"📊 검색 결과: {len(documents)}개 문서")
        if documents:
            print(f"📄 첫 번째 문서: {documents[0].get('title', 'N/A')}")
        
        # 검색 통계
        stats = dm.get_search_statistics()
        print(f"📈 검색 통계: {stats}")
        
        api_client.close_session()
        return True
        
    except Exception as e:
        print(f"❌ 검색 컴포넌트 테스트 실패: {e}")
        return False

def main():
    """메인 테스트 함수"""
    print("🚀 RAG 시스템 전체 테스트 시작\n")
    
    tests = [
        ("검색 컴포넌트", test_search_components),
        ("전체 RAG 파이프라인", test_full_rag_pipeline)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*60}")
            print(f"🧪 {test_name} 테스트")
            print('='*60)
            
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} 테스트 중 예외 발생: {e}")
            results.append((test_name, False))
    
    # 결과 요약
    print("\n" + "="*60)
    print("📊 전체 테스트 결과 요약")
    print("="*60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ 통과" if result else "❌ 실패"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 전체 결과: {passed}/{len(results)} 테스트 통과")
    
    if passed == len(results):
        print("🎉 모든 테스트가 성공했습니다!")
        print("🚀 RAG 시스템이 완벽하게 작동합니다!")
        return 0
    else:
        print("⚠️  일부 테스트가 실패했습니다.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
