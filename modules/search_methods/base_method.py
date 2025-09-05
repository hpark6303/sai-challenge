"""
검색 방법 기본 클래스
- 모든 검색 방법이 구현해야 하는 인터페이스
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any

class SearchMethod(ABC):
    """검색 방법 인터페이스"""
    
    @abstractmethod
    def search(self, query: str, tools: Dict[str, Any], 
               document_manager: Any, metadata: Dict[str, Any]) -> List[Dict]:
        """
        검색 실행
        
        Args:
            query: 검색 쿼리
            tools: 사용 가능한 검색 도구들
            document_manager: 문서 관리자
            metadata: 검색 메타데이터
            
        Returns:
            검색된 문서 리스트
        """
        pass
    
    @abstractmethod
    def get_method_name(self) -> str:
        """방법 이름 반환"""
        pass
