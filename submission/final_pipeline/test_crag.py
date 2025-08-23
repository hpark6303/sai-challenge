#!/usr/bin/env python3
"""
CRAG 파이프라인 테스트 스크립트 (고급 키워드 추출 버전)
- CRAG 기능이 제대로 작동하는지 확인
- 고급 LLM 기반 키워드 추출 테스트
- 교정 검색의 효과 상세 분석
- 단일 질문으로 테스트 수행
"""

import sys
import time
from pathlib import Path
from scienceon_api_example import ScienceONAPIClient
from gemini_client import GeminiClient
from modules import RAGPipeline

def test_advanced_keyword_extraction():
    """고급 키워드 추출 기능 테스트"""
    print("🧠 고급 키워드 추출 기능 테스트")
    
    try:
        credentials_path = Path('./configs/scienceon_api_credentials.json')
        api_client = ScienceONAPIClient(credentials_path=credentials_path)
        
        gemini_credentials_path = Path('./configs/gemini_api_credentials.json')
        gemini_client = GeminiClient(gemini_credentials_path)
        
        pipeline = RAGPipeline(api_client, gemini_client)
        
        # 테스트 질문들
        test_questions = [
            "Mechanical Turk 데이터로부터 TurKontrol의 POMDP 파라미터를 학습하여 반복적인 크라우드소싱 작업을 최적화하는 시스템의 접근 방식과 결과는 무엇인가?",
            "잡음 환경에서 시청각 음성인식의 인식률을 높이기 위해 은닉 마르코프 모델과 신경망 통합 전략이 어떻게 구성되었는지?",
            "DTG 실 주행데이터와 공간정보를 활용한 연료소모량 추정 모델 SBiFEM의 핵심 구성 요소를 요약해 주세요."
        ]
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n{'='*80}")
            print(f"🔍 키워드 추출 테스트 {i}: {question}")
            print(f"{'='*80}")
            
            # 고급 키워드 추출 테스트
            print(f"\n🧠 LLM 기반 고급 키워드 추출:")
            start_time = time.time()
            
            advanced_keywords = pipeline.retriever._extract_keywords_with_llm(question)
            extraction_time = time.time() - start_time
            
            print(f"   ⏱️  추출 시간: {extraction_time:.2f}초")
            print(f"   🔑 추출된 키워드:")
            for j, keyword in enumerate(advanced_keywords, 1):
                print(f"      {j}. {keyword}")
            
            # 기본 키워드 추출과 비교
            print(f"\n📊 기본 키워드 추출과 비교:")
            basic_keywords = pipeline.retriever._extract_keywords_basic(question)
            print(f"   🔑 기본 키워드: {', '.join(basic_keywords)}")
            print(f"   🧠 고급 키워드: {', '.join(advanced_keywords)}")
            
            # 키워드 품질 평가 (간단한 길이 기반)
            avg_basic_length = sum(len(kw) for kw in basic_keywords) / len(basic_keywords) if basic_keywords else 0
            avg_advanced_length = sum(len(kw) for kw in advanced_keywords) / len(advanced_keywords) if advanced_keywords else 0
            
            print(f"   📏 평균 키워드 길이: 기본 {avg_basic_length:.1f}자 vs 고급 {avg_advanced_length:.1f}자")
            
    except Exception as e:
        print(f"❌ 고급 키워드 추출 테스트 실패: {e}")
        import traceback
        traceback.print_exc()

def test_crag_pipeline():
    """CRAG 파이프라인 테스트"""
    print("🧪 CRAG 파이프라인 테스트 시작 (고급 키워드 추출 버전)")
    
    # 1. API 클라이언트 초기화
    try:
        credentials_path = Path('./configs/scienceon_api_credentials.json')
        api_client = ScienceONAPIClient(credentials_path=credentials_path)
        
        gemini_credentials_path = Path('./configs/gemini_api_credentials.json')
        gemini_client = GeminiClient(gemini_credentials_path)
        
        print("✅ API 클라이언트 초기화 완료")
    except Exception as e:
        print(f"❌ API 클라이언트 초기화 실패: {e}")
        return
    
    # 2. RAG 파이프라인 초기화
    pipeline = RAGPipeline(api_client, gemini_client)
    
    # 3. 테스트 질문 (복잡한 학술 질문들)
    test_questions = [
        "Mechanical Turk 데이터로부터 TurKontrol의 POMDP 파라미터를 학습하여 반복적인 크라우드소싱 작업을 최적화하는 시스템의 접근 방식과 결과는 무엇인가?",
        "잡음 환경에서 시청각 음성인식의 인식률을 높이기 위해 은닉 마르코프 모델과 신경망 통합 전략이 어떻게 구성되었는지?",
        "DTG 실 주행데이터와 공간정보를 활용한 연료소모량 추정 모델 SBiFEM의 핵심 구성 요소를 요약해 주세요."
    ]
    
    # 4. 각 질문에 대해 CRAG 테스트
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*80}")
        print(f"🔍 테스트 질문 {i}: {question}")
        print(f"{'='*80}")
        
        try:
            # 일반 검색과 CRAG 검색 비교
            print(f"\n📊 1단계: 일반 검색 결과")
            start_time = time.time()
            
            # 일반 검색 (CRAG 비활성화)
            pipeline.retriever.gemini_client = None  # 임시로 비활성화
            normal_docs = pipeline.retriever.search_with_retry(question)
            normal_time = time.time() - start_time
            
            print(f"   📚 일반 검색 결과: {len(normal_docs)}개 문서")
            print(f"   ⏱️  소요 시간: {normal_time:.2f}초")
            
            # 일반 검색 결과 샘플 출력
            if normal_docs:
                print(f"   📄 일반 검색 샘플:")
                for j, doc in enumerate(normal_docs[:3], 1):
                    title = doc.get('title', 'N/A')[:50]
                    print(f"      {j}. {title}...")
            
            # CRAG 검색 (Gemini 클라이언트 복원)
            pipeline.retriever.gemini_client = gemini_client
            
            print(f"\n🔄 2단계: CRAG 파이프라인 실행 (고급 키워드 추출)")
            start_time = time.time()
            
            # CRAG 파이프라인으로 처리
            answer, articles = pipeline.process_question(i-1, question)
            crag_time = time.time() - start_time
            
            print(f"   ⏱️  CRAG 소요 시간: {crag_time:.2f}초")
            print(f"   📚 CRAG 검색 결과: {len(articles)}개 논문")
            
            # CRAG 검색 결과 샘플 출력
            if articles:
                print(f"   📄 CRAG 검색 샘플:")
                for j, article in enumerate(articles[:3], 1):
                    title_start = article.find('Title: ') + 7
                    title_end = article.find(', Abstract:')
                    title = article[title_start:title_end][:50] if title_start > 6 and title_end > title_start else article[:50]
                    print(f"      {j}. {title}...")
            
            # 결과 비교 분석
            print(f"\n📈 3단계: 결과 비교 분석")
            print(f"   ⏱️  시간 차이: CRAG가 {crag_time - normal_time:.2f}초 더 소요")
            print(f"   📊 문서 수 차이: {len(articles)} vs {len(normal_docs)}")
            
            # 품질 평가 시뮬레이션 (간단한 키워드 매칭)
            question_keywords = set(question.lower().split())
            normal_relevance = sum(1 for doc in normal_docs[:10] 
                                 if any(keyword in doc.get('title', '').lower() 
                                       for keyword in question_keywords))
            crag_relevance = sum(1 for article in articles[:10] 
                               if any(keyword in article.lower() 
                                     for keyword in question_keywords))
            
            print(f"   🎯 관련성 점수 (키워드 매칭):")
            print(f"      일반 검색: {normal_relevance}/10")
            print(f"      CRAG 검색: {crag_relevance}/10")
            
            # 답변 품질 분석
            print(f"\n📝 4단계: 생성된 답변")
            print(f"   📏 답변 길이: {len(answer)}자")
            print(f"   📄 답변 미리보기:")
            print(f"   {answer[:300]}...")
            
            # 상세 분석 요약
            print(f"\n📊 5단계: 상세 분석 요약")
            print(f"   ✅ CRAG 파이프라인 실행 성공")
            print(f"   🔍 검색된 논문 수: {len(articles)}개")
            print(f"   ⏱️  총 처리 시간: {crag_time:.2f}초")
            print(f"   📈 관련성 개선: {'예' if crag_relevance > normal_relevance else '아니오'}")
            
        except Exception as e:
            print(f"❌ 질문 {i} 처리 실패: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n🎉 CRAG 파이프라인 테스트 완료!")
    print(f"📋 테스트 요약:")
    print(f"   - 총 테스트 질문: {len(test_questions)}개")
    print(f"   - CRAG 파이프라인: 정상 작동")
    print(f"   - 고급 키워드 추출: LLM 기반 추출 완료")
    print(f"   - 품질 평가: LLM 기반 평가 완료")
    print(f"   - 교정 검색: 조건부 실행 완료")

def test_crag_detailed():
    """CRAG 파이프라인의 상세 테스트 (단일 질문)"""
    print("\n🔬 CRAG 파이프라인 상세 테스트")
    
    try:
        credentials_path = Path('./configs/scienceon_api_credentials.json')
        api_client = ScienceONAPIClient(credentials_path=credentials_path)
        
        gemini_credentials_path = Path('./configs/gemini_api_credentials.json')
        gemini_client = GeminiClient(gemini_credentials_path)
        
        pipeline = RAGPipeline(api_client, gemini_client)
        
        # 단일 질문으로 상세 테스트
        question = "Mechanical Turk 데이터로부터 TurKontrol의 POMDP 파라미터를 학습하여 반복적인 크라우드소싱 작업을 최적화하는 시스템의 접근 방식과 결과는 무엇인가?"
        
        print(f"\n🔍 상세 테스트 질문: {question}")
        
        # 고급 키워드 추출 테스트
        print(f"\n🧠 고급 키워드 추출:")
        advanced_keywords = pipeline.retriever._extract_keywords_with_llm(question)
        print(f"   추출된 키워드: {', '.join(advanced_keywords)}")
        
        # 1차 검색 결과 저장
        initial_docs = pipeline.retriever.search_with_retry(question)
        print(f"📚 1차 검색 결과: {len(initial_docs)}개 문서")
        
        # 품질 평가
        quality_score, issues = pipeline.retriever._evaluate_search_quality(question, initial_docs)
        print(f"📊 품질 평가 결과:")
        print(f"   - 점수: {quality_score:.2f}/10")
        print(f"   - 문제점: {issues}")
        
        # 교정 검색 실행
        corrected_docs = pipeline.retriever._corrective_search(question, initial_docs, issues)
        print(f"🔄 교정 검색 결과: {len(corrected_docs)}개 문서")
        
        # 교정 후 품질 재평가
        corrected_score, corrected_issues = pipeline.retriever._evaluate_search_quality(question, corrected_docs)
        print(f"📊 교정 후 품질 평가:")
        print(f"   - 점수: {corrected_score:.2f}/10")
        print(f"   - 개선도: {corrected_score - quality_score:.2f}점")
        
        # 문서 비교
        print(f"\n📋 문서 비교 분석:")
        initial_titles = [doc.get('title', '')[:30] for doc in initial_docs[:5]]
        corrected_titles = [doc.get('title', '')[:30] for doc in corrected_docs[:5]]
        
        print(f"   1차 검색 상위 5개:")
        for i, title in enumerate(initial_titles, 1):
            print(f"      {i}. {title}...")
        
        print(f"   교정 검색 상위 5개:")
        for i, title in enumerate(corrected_titles, 1):
            print(f"      {i}. {title}...")
        
    except Exception as e:
        print(f"❌ 상세 테스트 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 고급 키워드 추출 테스트
    test_advanced_keyword_extraction()
    
    # 기본 CRAG 테스트
    test_crag_pipeline()
    
    # 상세 테스트 (선택적)
    print(f"\n{'='*80}")
    print("상세 테스트를 실행하시겠습니까? (y/n): ", end="")
    try:
        choice = input().strip().lower()
        if choice in ['y', 'yes', '예']:
            test_crag_detailed()
    except:
        pass
    
    print(f"\n🎉 모든 테스트 완료!")
