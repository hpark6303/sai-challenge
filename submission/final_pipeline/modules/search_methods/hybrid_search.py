"""
하이브리드 검색 방법
- 키워드 검색 + 벡터 검색 결합
"""

import logging
from typing import List, Dict, Any
from .base_method import SearchMethod

class HybridSearchMethod(SearchMethod):
    """하이브리드 검색 방법"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        하이브리드 검색 방법 초기화
        
        Args:
            config: 방법 설정
        """
        self.config = config or {
            'max_docs': 50,
            'keyword_weight': 0.6,
            'vector_weight': 0.4,
            'similarity_threshold': 0.3
        }
        self.method_name = "hybrid_search"
    
    def search(self, query: str, tools: Dict[str, Any], 
               document_manager: Any, metadata: Dict[str, Any]) -> List[Dict]:
        """
        하이브리드 검색 실행
        
        Args:
            query: 검색 쿼리
            tools: 사용 가능한 검색 도구들
            document_manager: 문서 관리자
            metadata: 검색 메타데이터
            
        Returns:
            검색된 문서 리스트
        """
        all_documents = []
        
        # 1. 키워드 검색
        keyword_docs = self._keyword_search(query, tools, document_manager, metadata)
        all_documents.extend(keyword_docs)
        
        # 2. 벡터 검색
        vector_docs = self._vector_search(query, document_manager, metadata)
        all_documents.extend(vector_docs)
        
        # 3. 결과 병합 및 중복 제거
        merged_docs = self._merge_documents(all_documents)
        
        # 4. 문서 저장
        if merged_docs:
            document_manager.store_documents(merged_docs, query, metadata)
        
        logging.info(f"하이브리드 검색 완료: {len(merged_docs)}개 문서")
        return merged_docs
    
    def _keyword_search(self, query: str, tools: Dict[str, Any], 
                       document_manager: Any, metadata: Dict[str, Any]) -> List[Dict]:
        """키워드 검색 실행"""
        keywords = metadata.get('keywords', [])
        if not keywords:
            return []
        
        tool_name = metadata.get('tool', 'scienceon')
        if tool_name not in tools:
            return []
        
        tool = tools[tool_name]
        max_docs = int(metadata.get('max_docs', self.config['max_docs']) * self.config['keyword_weight'])
        
        return tool.search_documents(keywords, max_docs)
    
    def _vector_search(self, query: str, document_manager: Any, 
                      metadata: Dict[str, Any]) -> List[Dict]:
        """벡터 검색 실행"""
        max_docs = int(metadata.get('max_docs', self.config['max_docs']) * self.config['vector_weight'])
        similarity_threshold = metadata.get('similarity_threshold', self.config['similarity_threshold'])
        
        return document_manager.search_similar_documents(
            query, max_docs, similarity_threshold
        )
    
    def _merge_documents(self, documents: List[Dict]) -> List[Dict]:
        """문서 병합 및 중복 제거"""
        seen_ids = set()
        merged_docs = []
        
        for doc in documents:
            doc_id = doc.get('CN') or doc.get('id')
            if doc_id and doc_id not in seen_ids:
                seen_ids.add(doc_id)
                merged_docs.append(doc)
        
        return merged_docs
    
    def get_method_name(self) -> str:
        return self.method_name
