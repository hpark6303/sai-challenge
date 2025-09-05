"""
의미적 검색 방법
- 벡터 기반 의미적 유사도 검색
"""

import logging
from typing import List, Dict, Any
from .base_method import SearchMethod

class SemanticSearchMethod(SearchMethod):
    """의미적 검색 방법"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        의미적 검색 방법 초기화
        
        Args:
            config: 방법 설정
        """
        self.config = config or {
            'max_docs': 50,
            'similarity_threshold': 0.3,
            'use_reranking': True
        }
        self.method_name = "semantic_search"
    
    def search(self, query: str, tools: Dict[str, Any], 
               document_manager: Any, metadata: Dict[str, Any]) -> List[Dict]:
        """
        의미적 검색 실행
        
        Args:
            query: 검색 쿼리
            tools: 사용 가능한 검색 도구들
            document_manager: 문서 관리자
            metadata: 검색 메타데이터
            
        Returns:
            검색된 문서 리스트
        """
        max_docs = metadata.get('max_docs', self.config['max_docs'])
        similarity_threshold = metadata.get('similarity_threshold', self.config['similarity_threshold'])
        
        # 벡터 기반 의미적 검색
        documents = document_manager.search_similar_documents(
            query, max_docs, similarity_threshold
        )
        
        # 재순위화 (옵션)
        if self.config.get('use_reranking', True):
            documents = self._rerank_documents(documents, query)
        
        # 문서 저장
        if documents:
            document_manager.store_documents(documents, query, metadata)
        
        logging.info(f"의미적 검색 완료: {len(documents)}개 문서")
        return documents
    
    def _rerank_documents(self, documents: List[Dict], query: str) -> List[Dict]:
        """문서 재순위화"""
        # TODO: 고급 재순위화 로직 구현
        # 현재는 단순히 원본 순서 유지
        return documents
    
    def get_method_name(self) -> str:
        return self.method_name
