"""
기본 키워드 추출기
- 기존 retrieval.py의 _extract_keywords_basic 로직을 정리
"""

import re
import logging
from typing import List
from konlpy.tag import Okt
from .base_extractor import KeywordExtractor

class BasicKeywordExtractor(KeywordExtractor):
    """기본 키워드 추출기 (한국어/영어 지원)"""
    
    def __init__(self, config: dict = None):
        """
        기본 키워드 추출기 초기화
        
        Args:
            config: 추출 설정
        """
        self.config = config or {
            'max_keywords': 8,
            'min_keyword_length': 2,
            'use_technical_terms': True,
            'use_special_terms': True
        }
        self.extractor_name = "basic_extractor"
        self.okt = Okt()
        
        # 영어 불용어 정의
        self.english_stop_words = {
            'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'how', 'what', 'why', 'when', 'where', 'which', 'who', 'whom', 'whose',
            'can', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'this', 'that', 'these', 'those', 'it', 'its', 'they', 'them', 'their',
            'we', 'you', 'he', 'she', 'his', 'her', 'our', 'your', 'my', 'me', 'i'
        }
        
        # 전문 용어 정의
        self.technical_terms = {
            'neural', 'artificial', 'machine', 'learning', 'deep', 'network', 
            'algorithm', 'model', 'system', 'method', 'approach', 'technique',
            'sustainability', 'corporate', 'culture', 'development', 'management',
            'analysis', 'research', 'study', 'framework', 'architecture',
            '인공지능', '머신러닝', '딥러닝', '알고리즘', '시스템', '분석', '연구'
        }
    
    def extract_keywords(self, query: str, **kwargs) -> List[str]:
        """
        기본 키워드 추출
        
        Args:
            query: 검색 쿼리
            **kwargs: 추가 옵션
            
        Returns:
            추출된 키워드 리스트
        """
        if self._is_korean(query):
            keywords = self._extract_korean_keywords(query)
        else:
            keywords = self._extract_english_keywords(query)
        
        # 특수 용어 추가
        if self.config.get('use_special_terms', True):
            special_terms = self._extract_special_terms(query)
            keywords.extend(special_terms)
        
        # 중복 제거 및 정렬
        keywords = list(set(keywords))
        keywords.sort(key=len, reverse=True)  # 긴 키워드 우선
        
        logging.info(f"기본 키워드 추출: {', '.join(keywords)}")
        return keywords[:self.config['max_keywords']]
    
    def _extract_korean_keywords(self, query: str) -> List[str]:
        """한국어 키워드 추출"""
        # 명사 추출
        nouns = self.okt.nouns(query)
        keywords = [noun for noun in nouns if len(noun) > 1]
        
        # 전문 용어 보존
        if self.config.get('use_technical_terms', True):
            for term in self.technical_terms:
                if term in query and term not in keywords:
                    keywords.append(term)
        
        return keywords
    
    def _extract_english_keywords(self, query: str) -> List[str]:
        """영어 키워드 추출"""
        # 단어 분리
        words = re.findall(r'\w+', query.lower())
        
        # 불용어 제거
        filtered_words = [word for word in words 
                         if word not in self.english_stop_words and len(word) > 2]
        
        # 전문 용어와 일반 용어 분리
        technical_terms = []
        general_terms = []
        
        for word in filtered_words:
            if (len(word) > 6 or word in self.technical_terms):
                technical_terms.append(word)
            else:
                general_terms.append(word)
        
        # 전문 용어를 먼저, 그 다음 일반 용어
        return technical_terms + general_terms
    
    def _extract_special_terms(self, query: str) -> List[str]:
        """특수 용어 추출"""
        special_terms = []
        
        # 약어 패턴 (대문자 2개 이상)
        abbreviations = re.findall(r'\b[A-Z]{2,}\b', query)
        special_terms.extend(abbreviations)
        
        # 하이픈이 있는 복합어
        hyphenated = re.findall(r'\b\w+-\w+\b', query)
        special_terms.extend(hyphenated)
        
        # 언더스코어가 있는 용어
        underscored = re.findall(r'\b\w+_\w+\b', query)
        special_terms.extend(underscored)
        
        return special_terms
    
    def _is_korean(self, text: str) -> bool:
        """한국어 텍스트 감지"""
        return bool(re.search('[가-힣]', text))
    
    def get_extractor_name(self) -> str:
        return self.extractor_name
    
    def get_config(self) -> dict:
        return self.config.copy()
    
    def update_config(self, new_config: dict):
        """설정 업데이트"""
        self.config.update(new_config)
        logging.info(f"기본 추출기 설정 업데이트: {new_config}")
