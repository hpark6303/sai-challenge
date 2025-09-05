#!/usr/bin/env python3
"""
간단한 RAG 테스트
"""

import sys
import logging
from pathlib import Path

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_simple_rag():
    """간단한 RAG 테스트"""
    print("🚀 간단한 RAG 테스트 시작\n")
    
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
        
        # 4. 단일 질문 테스트
        test_question = "인공지능의 윤리적 문제는 무엇인가?"
        print(f"\n🔍 테스트 질문: {test_question}")
        
        # 질문 처리
        answer, articles = rag_pipeline.process_question(0, test_question)
        
        print(f"\n🤖 답변:")
        print(f"{answer[:300]}..." if len(answer) > 300 else answer)
        
        print(f"\n📚 논문 정보 ({len(articles)}개):")
        for i, article in enumerate(articles[:3]):  # 처음 3개만 표시
            print(f"  {i+1}. {article[:100]}...")
        
        if len(articles) > 3:
            print(f"  ... 및 {len(articles)-3}개 더")
        
        # 5. 정리
        api_client.close_session()
        print(f"\n✅ 테스트 완료!")
        
        return True
        
    except Exception as e:
        print(f"❌ RAG 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_simple_rag()
    sys.exit(0 if success else 1)
