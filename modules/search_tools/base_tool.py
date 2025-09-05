"""
검색 도구 기본 클래스
- 모든 검색 도구가 구현해야 하는 인터페이스
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any

class SearchTool(ABC):
    """검색 도구 인터페이스"""
    
    @abstractmethod
    def search_documents(self, keywords: List[str], max_docs: int = 50) -> List[Dict]:
        """
        문서 검색
        
        Args:
            keywords: 검색 키워드 리스트
            max_docs: 최대 문서 수
            
        Returns:
            검색된 문서 리스트
        """
        pass
    
    @abstractmethod
    def get_tool_name(self) -> str:
        """도구 이름 반환"""
        pass
    
    @abstractmethod
    def get_required_fields(self) -> List[str]:
        """필요한 필드 목록 반환"""
        pass
