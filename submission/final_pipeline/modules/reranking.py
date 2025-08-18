"""
고급 재순위화 모듈 (대회 핵심 요구사항)
- 질문과 문서 간의 관련성 점수 계산
- 다중 기준 재순위화 (TF-IDF, 키워드 매칭, 품질, 컨텍스트)
- 다양성 기반 필터링
- 도메인별 최적화
"""

from typing import List, Dict, Tuple
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .config import ANSWER_CONFIG

class DocumentReranker:
    """고급 문서 재순위화기 (대회 핵심 요구사항)"""
    
    def __init__(self):
        """재순위화기 초기화"""
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
    
    def rerank_documents(self, documents: List[Dict], query: str, top_k: int = 50) -> List[Dict]:
        """
        고급 문서 재순위화 (대회 핵심 요구사항)
        
        Args:
            documents: 원본 문서 리스트
            query: 검색 쿼리
            top_k: 반환할 상위 문서 수
            
        Returns:
            재순위화된 문서 리스트
        """
        if not documents:
            return documents
        
        print(f"   🔄 고급 문서 재순위화 시작: {len(documents)}개 문서")
        
        # 1. 다중 기준 관련성 점수 계산
        scored_docs = []
        for doc in documents:
            relevance_score = self._calculate_relevance_score(query, doc)
            doc_with_score = doc.copy()
            doc_with_score['_relevance_score'] = relevance_score
            scored_docs.append(doc_with_score)
        
        # 2. 관련성 점수로 정렬
        reranked_docs = sorted(scored_docs, key=lambda x: x['_relevance_score'], reverse=True)
        
        # 3. 다양성 기반 필터링
        diverse_docs = self.filter_by_diversity(reranked_docs[:top_k*2])  # 2배로 확장 후 필터링
        
        print(f"   ✅ 고급 재순위화 완료: 상위 {len(diverse_docs)}개 선택")
        
        return diverse_docs[:top_k]
    
    def _calculate_relevance_score(self, query: str, document: Dict) -> float:
        """
        질문과 문서 간의 관련성 점수 계산 (다중 기준)
        
        Args:
            query: 질문
            document: 문서
            
        Returns:
            관련성 점수 (0.0 ~ 1.0)
        """
        
        title = document.get('title', '')
        abstract = document.get('abstract', '')
        
        # 1. TF-IDF 기반 유사도 (30%)
        tfidf_score = self._calculate_tfidf_similarity(query, title + " " + abstract)
        
        # 2. 키워드 매칭 점수 (25%)
        keyword_score = self._calculate_keyword_matching(query, title, abstract)
        
        # 3. 제목 관련성 점수 (20%)
        title_score = self._calculate_title_relevance(query, title)
        
        # 4. 문서 품질 점수 (15%)
        quality_score = self._calculate_document_quality(document)
        
        # 5. 컨텍스트 일관성 점수 (10%)
        context_score = self._calculate_context_consistency(query, abstract)
        
        # 가중 평균 계산
        final_score = (
            tfidf_score * 0.3 +
            keyword_score * 0.25 +
            title_score * 0.2 +
            quality_score * 0.15 +
            context_score * 0.1
        )
        
        return final_score
    
    def _calculate_tfidf_similarity(self, query: str, text: str) -> float:
        """TF-IDF 기반 유사도 계산"""
        try:
            # TF-IDF 벡터화
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([query, text])
            
            # 코사인 유사도 계산
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            return float(similarity)
        except:
            return 0.0
    
    def _calculate_keyword_matching(self, query: str, title: str, abstract: str) -> float:
        """키워드 매칭 점수 계산"""
        
        # 질문에서 키워드 추출
        query_keywords = self._extract_keywords(query)
        
        # 제목과 초록에서 키워드 매칭
        title_matches = sum(1 for keyword in query_keywords if keyword.lower() in title.lower())
        abstract_matches = sum(1 for keyword in query_keywords if keyword.lower() in abstract.lower())
        
        # 가중 점수 계산
        title_score = title_matches / len(query_keywords) if query_keywords else 0
        abstract_score = abstract_matches / len(query_keywords) if query_keywords else 0
        
        # 제목 매칭에 더 높은 가중치
        return title_score * 0.7 + abstract_score * 0.3
    
    def _calculate_title_relevance(self, query: str, title: str) -> float:
        """제목 관련성 점수 계산"""
        
        # 질문의 핵심 개념 추출
        query_concepts = self._extract_concepts(query)
        title_concepts = self._extract_concepts(title)
        
        # 개념 매칭 계산
        matches = len(set(query_concepts) & set(title_concepts))
        total = len(set(query_concepts) | set(title_concepts))
        
        return matches / total if total > 0 else 0.0
    
    def _extract_keywords(self, text: str) -> List[str]:
        """텍스트에서 키워드 추출"""
        # 간단한 키워드 추출
        words = re.findall(r'\b\w+\b', text.lower())
        
        # 불용어 제거
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'how', 'what', 'why', 'when', 'where', 'which', 'who',
            'can', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might'
        }
        
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords[:10]  # 상위 10개만
    
    def _extract_concepts(self, text: str) -> List[str]:
        """텍스트에서 핵심 개념 추출"""
        # 더 긴 단어들을 개념으로 간주
        words = re.findall(r'\b\w+\b', text.lower())
        concepts = [word for word in words if len(word) > 5]
        
        return concepts[:5]  # 상위 5개만
    
    def _apply_quality_scores(self, documents: List[Dict], query: str) -> List[Dict]:
        """품질 점수 계산 및 적용"""
        for doc in documents:
            quality_score = self._calculate_quality_score(doc, query)
            doc['quality_score'] = quality_score
            
            # 종합 점수 계산 (유사도 70% + 품질 30%)
            similarity_score = doc.get('similarity_score', 0)
            doc['final_score'] = similarity_score * 0.7 + quality_score * 0.3
        
        return documents
    
    def _calculate_document_quality(self, document: Dict) -> float:
        """문서 품질 점수 계산 (개선된 버전)"""
        title = document.get('title', '')
        abstract = document.get('abstract', '')
        
        quality_score = 0.0
        
        # 1. 제목 길이 점수
        if 10 <= len(title) <= 200:
            quality_score += 0.3
        elif len(title) > 200:
            quality_score += 0.1
        
        # 2. 초록 길이 점수
        if 50 <= len(abstract) <= 2000:
            quality_score += 0.4
        elif len(abstract) > 2000:
            quality_score += 0.2
        
        # 3. 제목 품질 점수 (질문어 제외)
        if not any(title.lower().startswith(word) for word in ['how', 'what', 'why', 'when', 'where']):
            quality_score += 0.3
        
        return min(quality_score, 1.0)
    
    def _calculate_context_consistency(self, query: str, abstract: str) -> float:
        """컨텍스트 일관성 점수 계산"""
        
        # 질문의 도메인 추정
        query_domain = self._estimate_domain(query)
        abstract_domain = self._estimate_domain(abstract)
        
        # 도메인 일치도 계산
        if query_domain == abstract_domain:
            return 1.0
        elif query_domain and abstract_domain:
            # 부분 일치
            common_terms = set(query_domain.split()) & set(abstract_domain.split())
            return len(common_terms) / max(len(query_domain.split()), len(abstract_domain.split()))
        
        return 0.5  # 기본값
    
    def _estimate_domain(self, text: str) -> str:
        """텍스트의 도메인 추정"""
        text_lower = text.lower()
        
        # 도메인별 키워드 매칭
        domains = {
            'computer_science': ['algorithm', 'neural', 'network', 'machine', 'learning', 'artificial', 'intelligence'],
            'mathematics': ['mathematics', 'mathematical', 'equation', 'theorem', 'proof', 'calculation'],
            'medicine': ['medical', 'clinical', 'patient', 'treatment', 'diagnosis', 'disease'],
            'engineering': ['engineering', 'system', 'design', 'technology', 'implementation'],
            'business': ['business', 'management', 'corporate', 'strategy', 'organization'],
            'sustainability': ['sustainability', 'environmental', 'green', 'eco', 'climate']
        }
        
        for domain, keywords in domains.items():
            if any(keyword in text_lower for keyword in keywords):
                return domain
        
        return 'general'
    
    def filter_by_diversity(self, documents: List[Dict], max_similar: float = 0.8) -> List[Dict]:
        """
        다양성 기반 필터링 (중복 제거)
        
        Args:
            documents: 재순위화된 문서 리스트
            max_similar: 최대 유사도 임계값
            
        Returns:
            다양성이 보장된 문서 리스트
        """
        
        if not documents:
            return []
        
        diverse_docs = [documents[0]]  # 첫 번째 문서는 항상 포함
        
        for doc in documents[1:]:
            # 기존 문서들과의 유사도 계산
            max_similarity = 0.0
            
            for existing_doc in diverse_docs:
                similarity = self._calculate_document_similarity(doc, existing_doc)
                max_similarity = max(max_similarity, similarity)
            
            # 임계값보다 낮으면 추가
            if max_similarity < max_similar:
                diverse_docs.append(doc)
        
        print(f"   🌈 다양성 필터링: {len(documents)}개 → {len(diverse_docs)}개")
        
        return diverse_docs
    
    def _calculate_document_similarity(self, doc1: Dict, doc2: Dict) -> float:
        """두 문서 간의 유사도 계산"""
        
        text1 = doc1.get('title', '') + ' ' + doc1.get('abstract', '')
        text2 = doc2.get('title', '') + ' ' + doc2.get('abstract', '')
        
        return self._calculate_tfidf_similarity(text1, text2)
    
    def _final_ranking(self, documents: List[Dict]) -> List[Dict]:
        """최종 순위 조정"""
        # 종합 점수로 정렬
        documents = sorted(documents, key=lambda x: x.get('final_score', 0), reverse=True)
        
        # 순위 업데이트
        for i, doc in enumerate(documents):
            doc['final_rank'] = i + 1
        
        return documents
    
    def filter_by_quality(self, documents: List[Dict], min_quality: float = 0.3) -> List[Dict]:
        """
        품질 기준으로 필터링
        
        Args:
            documents: 문서 리스트
            min_quality: 최소 품질 점수
            
        Returns:
            필터링된 문서 리스트
        """
        return [doc for doc in documents if doc.get('quality_score', 0) >= min_quality]
    
    def get_top_documents(self, documents: List[Dict], top_k: int = ANSWER_CONFIG['max_context_docs']) -> List[Dict]:
        """
        상위 k개 문서 선택
        
        Args:
            documents: 문서 리스트
            top_k: 선택할 문서 수
            
        Returns:
            상위 k개 문서
        """
        return documents[:top_k]
    
    def create_context_from_documents(self, documents: List[Dict]) -> str:
        """
        문서들로부터 컨텍스트 생성
        
        Args:
            documents: 문서 리스트
            
        Returns:
            생성된 컨텍스트
        """
        if not documents:
            return ""
        
        context_parts = []
        for i, doc in enumerate(documents):
            title = doc.get('title', '')
            abstract = doc.get('abstract', '')
            similarity = doc.get('similarity_score', 0)
            
            context = f"[문서 {i+1}] (유사도: {similarity:.3f})\n"
            context += f"제목: {title}\n"
            context += f"초록: {abstract}\n"
            context_parts.append(context)
        
        return "\n".join(context_parts)
