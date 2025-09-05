"""
키워드 추출기 기본 클래스
- 모든 키워드 추출기가 구현해야 하는 인터페이스
"""

from abc import ABC, abstractmethod
from typing import List

class KeywordExtractor(ABC):
    """키워드 추출기 인터페이스"""
    
    @abstractmethod
    def extract_keywords(self, query: str, **kwargs) -> List[str]:
        """
        키워드 추출
        
        Args:
            query: 검색 쿼리
            **kwargs: 추가 옵션
            
        Returns:
            추출된 키워드 리스트
        """
        pass
    
    @abstractmethod
    def get_extractor_name(self) -> str:
        """추출기 이름 반환"""
        pass
