#!/usr/bin/env python3
"""
통합 테스트 스크립트
- 새로운 구조의 RAG 시스템 테스트
"""

import sys
import logging
from pathlib import Path

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_basic_imports():
    """기본 import 테스트"""
    print("🔍 기본 import 테스트...")
    
    try:
        from modules.rag_pipeline import RAGPipeline
        from modules.document_manager import DocumentManager
        from modules.search_engine import FlexibleSearchEngine
        from modules.search_tools import ScienceONTool
        from modules.search_methods import HybridSearchMethod
        from modules.keyword_extractors import LLMKeywordExtractor
        from scienceon_api_example import ScienceONAPIClient
        
        print("✅ 모든 모듈 import 성공")
        return True
    except Exception as e:
        print(f"❌ Import 실패: {e}")
        return False

def test_document_manager():
    """DocumentManager 테스트"""
    print("\n🔍 DocumentManager 테스트...")
    
    try:
        from modules.document_manager import DocumentManager
        
        # DocumentManager 초기화
        dm = DocumentManager()
        print("✅ DocumentManager 초기화 성공")
        
        # 문서 수 조회
        doc_count = dm.get_document_count()
        print(f"📊 저장된 문서 수: {doc_count}개")
        
        # 검색 통계 조회
        stats = dm.get_search_statistics()
        print(f"📊 검색 통계: {stats}")
        
        return True
    except Exception as e:
        print(f"❌ DocumentManager 테스트 실패: {e}")
        return False

def test_search_engine():
    """SearchEngine 테스트"""
    print("\n🔍 SearchEngine 테스트...")
    
    try:
        from modules.document_manager import DocumentManager
        from modules.search_engine import FlexibleSearchEngine
        from modules.search_tools import ScienceONTool
        from modules.search_methods import HybridSearchMethod
        
        # 컴포넌트 초기화
        dm = DocumentManager()
        se = FlexibleSearchEngine(dm)
        
        # 더미 도구와 방법 등록
        class DummyTool:
            def search_documents(self, keywords, max_docs=50):
                return [{'title': 'Test Document', 'abstract': 'Test Abstract', 'CN': 'TEST001'}]
            def get_tool_name(self): return "dummy"
            def get_required_fields(self): return ['title', 'abstract', 'CN']
        
        class DummyMethod:
            def search(self, query, tools, document_manager, metadata):
                return [{'title': 'Test Document', 'abstract': 'Test Abstract', 'CN': 'TEST001'}]
            def get_method_name(self): return "dummy"
        
        # 도구와 방법 등록
        se.register_tool("dummy", DummyTool(), is_default=True)
        se.register_method("dummy", DummyMethod(), is_default=True)
        
        print("✅ SearchEngine 초기화 및 등록 성공")
        
        # 사용 가능한 도구/방법 조회
        tools = se.get_available_tools()
        methods = se.get_available_methods()
        print(f"📊 사용 가능한 도구: {tools}")
        print(f"📊 사용 가능한 방법: {methods}")
        
        return True
    except Exception as e:
        print(f"❌ SearchEngine 테스트 실패: {e}")
        return False

def test_api_client():
    """API 클라이언트 테스트"""
    print("\n🔍 API 클라이언트 테스트...")
    
    try:
        from scienceon_api_example import ScienceONAPIClient
        from pathlib import Path
        
        # 인증 파일 경로 확인
        credentials_path = Path("./configs/scienceon_api_credentials.json")
        if not credentials_path.exists():
            print("⚠️  인증 파일이 없습니다. API 테스트를 건너뜁니다.")
            return True
        
        # API 클라이언트 초기화
        client = ScienceONAPIClient(credentials_path)
        print("✅ ScienceON API 클라이언트 초기화 성공")
        
        # 간단한 검색 테스트 (실제 API 호출)
        print("🔍 API 검색 테스트...")
        results = client.search_articles("인공지능", row_count=3)
        print(f"📊 검색 결과: {len(results)}개 문서")
        
        if results:
            print(f"📄 첫 번째 문서: {results[0].get('title', 'N/A')}")
        
        client.close_session()
        return True
    except Exception as e:
        print(f"❌ API 클라이언트 테스트 실패: {e}")
        return False

def main():
    """메인 테스트 함수"""
    print("🚀 RAG 시스템 통합 테스트 시작\n")
    
    tests = [
        ("기본 Import", test_basic_imports),
        ("DocumentManager", test_document_manager),
        ("SearchEngine", test_search_engine),
        ("API 클라이언트", test_api_client)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} 테스트 중 예외 발생: {e}")
            results.append((test_name, False))
    
    # 결과 요약
    print("\n" + "="*50)
    print("📊 테스트 결과 요약")
    print("="*50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ 통과" if result else "❌ 실패"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 전체 결과: {passed}/{len(results)} 테스트 통과")
    
    if passed == len(results):
        print("🎉 모든 테스트가 성공했습니다!")
        return 0
    else:
        print("⚠️  일부 테스트가 실패했습니다.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
