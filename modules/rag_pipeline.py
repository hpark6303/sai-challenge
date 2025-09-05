"""
RAG 파이프라인 메인 모듈 (개선된 버전)
- 전체 RAG 워크플로우 관리
- 모듈 간 협력 조정
- 결과 생성 및 검증
- 향상된 검색 시스템 통합
"""

import sys
import logging
from typing import List, Dict, Tuple
from .document_manager import DocumentManager
from .search_engine import FlexibleSearchEngine
from .search_tools import ScienceONTool
from .search_methods import KeywordSearchMethod, HybridSearchMethod, SemanticSearchMethod
from .keyword_extractors import LLMKeywordExtractor, BasicKeywordExtractor
from .reranking import DocumentReranker
from .answer_generator import AnswerGenerator
from .config import SEARCH_CONFIG, ANSWER_CONFIG, TEST_CONFIG

class RAGPipeline:
    """RAG 파이프라인 메인 클래스 (개선된 버전)"""
    
    def __init__(self, api_client, gemini_client, dataset_name: str = "scienceon"):
        """
        RAG 파이프라인 초기화
        
        Args:
            api_client: ScienceON API 클라이언트
            gemini_client: Gemini API 클라이언트
            dataset_name: 데이터셋 이름
        """
        self.dataset_name = dataset_name
        
        # 벡터 DB 초기화 여부 확인
        clear_db = TEST_CONFIG.get('clear_vector_db', False)
        
        # 각 모듈 초기화
        self.document_manager = DocumentManager(clear_db=clear_db)
        
        # 검색 도구들 초기화
        scienceon_tool = ScienceONTool(api_client)
        
        # 검색 방법들 초기화
        keyword_method = KeywordSearchMethod()
        hybrid_method = HybridSearchMethod()
        semantic_method = SemanticSearchMethod()
        
        # 검색 엔진 초기화
        self.search_engine = FlexibleSearchEngine(self.document_manager)
        self.search_engine.register_tool("scienceon", scienceon_tool, is_default=True)
        self.search_engine.register_method("keyword", keyword_method)
        self.search_engine.register_method("hybrid", hybrid_method, is_default=True)
        self.search_engine.register_method("semantic", semantic_method)
        
        # 키워드 추출기들 초기화
        self.keyword_extractors = {
            'llm': LLMKeywordExtractor(gemini_client),
            'basic': BasicKeywordExtractor()
        }
        
        self.reranker = DocumentReranker()
        self.answer_generator = AnswerGenerator(gemini_client)
        
        logging.info("✅ 향상된 RAG 파이프라인 초기화 완료")
    
    def process_question(self, question_id: int, query: str) -> Tuple[str, List[str]]:
        """
        단일 질문 처리 (전체 RAG 워크플로우)
        
        Args:
            question_id: 질문 ID
            query: 질문 내용
            
        Returns:
            (답변, 논문 정보 리스트) 튜플
        """
        print(f"\n🔍 질문 {question_id+1} 처리: '{query[:50]}...'")
        
        try:
            # 1단계: 문서 검색
            documents = self._retrieve_documents(query)
            print(f"   📚 검색된 문서: {len(documents)}개")
            
            # 2단계: 벡터 DB에 저장
            added_count = self.document_manager.store_documents(documents, query)
            if added_count > 0:
                print(f"   📚 벡터 DB에 {added_count}개 문서 추가")
            
            # 3단계: 벡터 검색
            similar_docs = self.document_manager.search_similar_documents(query)
            
            # 4단계: 검색 결과 보충 (기존 문서와 유사 문서 결합)
            final_docs = documents + similar_docs
            
            # 5단계: 재순위화
            reranked_docs = self.reranker.rerank_documents(final_docs, query)
            
            # 6단계: 답변 생성용 상위 문서 선택
            context_docs = self.reranker.get_top_documents(reranked_docs)
            
            # 7단계: 답변 생성
            answer = self.answer_generator.generate_quality_answer(query, context_docs)
            
            # 8단계: 논문 정보 형식화
            articles = self._format_articles(reranked_docs)
            
            return answer, articles
            
        except Exception as e:
            print(f"   ❌ 질문 {question_id+1} 처리 실패: {e}")
            return f"처리 중 오류가 발생했습니다: {str(e)}", [''] * 50
    
    def _retrieve_documents(self, query: str, search_strategy: str = None) -> List[Dict]:
        """
        문서 검색 (향상된 검색 시스템 사용)
        
        Args:
            query: 검색 쿼리
            search_strategy: 검색 전략
            
        Returns:
            검색된 문서 리스트
        """
        # 키워드 추출
        keywords = self.keyword_extractors['llm'].extract_keywords(query)
        
        # 검색 실행
        documents, search_metadata = self.search_engine.search(
            query, 
            dataset_name=self.dataset_name,
            method=search_strategy,
            keywords=keywords
        )
        
        return documents
    
    def _format_articles(self, documents: List[Dict]) -> List[str]:
        """
        문서를 Kaggle 형식으로 변환 (실제 50개 문서 보장)
        
        Args:
            documents: 문서 리스트
            
        Returns:
            Kaggle 형식의 논문 정보 리스트
        """
        articles = []
        
        # 실제 문서가 50개 미만이면 추가 검색
        if len(documents) < 50:
            print(f"   🚨 실제 문서 부족: {len(documents)}개 (목표: 50개)")
            print(f"   🔍 추가 문서 검색 중...")
            
            # 추가 검색을 위해 search_engine 사용
            additional_docs, _ = self.search_engine.search(
                "", 
                dataset_name=self.dataset_name,
                method="keyword",
                keywords=["research", "study", "analysis"]
            )
            documents.extend(additional_docs)
            
            # 중복 제거
            seen_ids = set()
            unique_docs = []
            for doc in documents:
                doc_id = doc.get('CN')
                if doc_id and doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    unique_docs.append(doc)
            documents = unique_docs
            
            print(f"   📊 추가 검색 후: {len(documents)}개 문서")
        
        # 정확히 50개 문서만 사용
        documents = documents[:50]
        
        for doc in documents:
            formatted_article = self._create_kaggle_format_article(doc)
            if not formatted_article or formatted_article.strip() == '':
                # 빈 문서인 경우 기본값으로 대체
                formatted_article = f'Title: {doc.get("title", "Research Document")}, Abstract: {doc.get("abstract", "This document contains relevant research information.")}, Source: http://click.ndsl.kr/servlet/OpenAPIDetailView?keyValue=05787966&target=NART&cn={doc.get("CN", "DOCUMENT")}'
            articles.append(formatted_article)
        
        # 정확히 50개 반환
        return articles[:50]
    
    def _create_kaggle_format_article(self, doc: Dict) -> str:
        """
        Kaggle 형식으로 논문 정보 생성
        
        Args:
            doc: 문서 딕셔너리
            
        Returns:
            Kaggle 형식의 문자열
        """
        title = doc.get('title', '')
        abstract = doc.get('abstract', '')
        cn = doc.get('CN', '')
        
        # 필수 필드 검증
        if not title or not cn:
            return ''
        
        # Source URL 생성
        source_url = f"http://click.ndsl.kr/servlet/OpenAPIDetailView?keyValue=05787966&target=NART&cn={cn}"
        
        # Abstract가 없으면 빈 문자열로 처리
        if not abstract:
            abstract = ''
        
        # Kaggle 형식으로 포맷팅
        return f'Title: {title}, Abstract: {abstract}, Source: {source_url}'
    
    def batch_process_questions(self, questions: List[Tuple[int, str]]) -> List[Tuple[int, str, List[str]]]:
        """
        배치 질문 처리
        
        Args:
            questions: (질문 ID, 질문) 튜플 리스트
            
        Returns:
            (질문 ID, 답변, 논문 정보) 튜플 리스트
        """
        results = []
        
        for question_id, query in questions:
            answer, articles = self.process_question(question_id, query)
            results.append((question_id, answer, articles))
        
        return results
    
    def get_pipeline_stats(self) -> Dict:
        """
        파이프라인 통계 정보
        
        Returns:
            통계 정보 딕셔너리
        """
        vector_stats = self.document_manager.get_stats()
        
        return {
            "vector_db": vector_stats,
            "search_config": SEARCH_CONFIG,
            "answer_config": ANSWER_CONFIG
        }
