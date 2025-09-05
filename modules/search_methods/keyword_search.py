"""
키워드 기반 검색 방법
- 키워드 추출 후 도구를 통한 검색
"""

import logging
from typing import List, Dict, Any
from .base_method import SearchMethod

class KeywordSearchMethod(SearchMethod):
    """키워드 기반 검색 방법"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        키워드 검색 방법 초기화
        
        Args:
            config: 방법 설정
        """
        self.config = config or {
            'max_docs': 50,
            'use_retry': True,
            'max_retries': 3
        }
        self.method_name = "keyword_search"
    
    def search(self, query: str, tools: Dict[str, Any], 
               document_manager: Any, metadata: Dict[str, Any]) -> List[Dict]:
        """
        키워드 기반 검색 실행
        
        Args:
            query: 검색 쿼리
            tools: 사용 가능한 검색 도구들
            document_manager: 문서 관리자
            metadata: 검색 메타데이터
            
        Returns:
            검색된 문서 리스트
        """
        # 키워드 추출
        keywords = metadata.get('keywords', [])
        if not keywords:
            logging.warning("키워드가 제공되지 않았습니다.")
            return []
        
        # 사용할 도구 선택
        tool_name = metadata.get('tool', 'scienceon')
        if tool_name not in tools:
            logging.error(f"도구 '{tool_name}'를 찾을 수 없습니다.")
            return []
        
        tool = tools[tool_name]
        max_docs = metadata.get('max_docs', self.config['max_docs'])
        
        # 도구를 통한 검색
        documents = tool.search_documents(keywords, max_docs)
        
        # 문서 저장
        if documents:
            document_manager.store_documents(documents, query, metadata)
        
        logging.info(f"키워드 검색 완료: {len(documents)}개 문서")
        return documents
    
    def get_method_name(self) -> str:
        return self.method_name
