"""
ArXiv API 검색 도구 (미래 확장용)
- ArXiv API를 통한 논문 검색
"""

import logging
from typing import List, Dict, Any
from .base_tool import SearchTool

class ArxivTool(SearchTool):
    """ArXiv API 검색 도구"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        ArXiv 도구 초기화
        
        Args:
            config: 도구 설정
        """
        self.config = config or {
            'max_docs': 50,
            'required_fields': ['title', 'abstract', 'authors', 'id']
        }
        self.tool_name = "arxiv"
    
    def search_documents(self, keywords: List[str], max_docs: int = 50) -> List[Dict]:
        """
        ArXiv API를 통한 문서 검색 (구현 예정)
        
        Args:
            keywords: 검색 키워드 리스트
            max_docs: 최대 문서 수
            
        Returns:
            검색된 문서 리스트
        """
        # TODO: ArXiv API 구현
        logging.warning("ArXiv 도구는 아직 구현되지 않았습니다.")
        return []
    
    def get_tool_name(self) -> str:
        return self.tool_name
    
    def get_required_fields(self) -> List[str]:
        return self.config['required_fields']
