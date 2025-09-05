"""
도메인별 키워드 추출기
- 특정 분야에 특화된 키워드 추출
"""

import logging
from typing import List, Dict
from .base_extractor import KeywordExtractor

class DomainKeywordExtractor(KeywordExtractor):
    """도메인별 키워드 추출기"""
    
    def __init__(self, config: dict = None):
        """
        도메인 키워드 추출기 초기화
        
        Args:
            config: 추출 설정
        """
        self.config = config or {
            'max_keywords': 10,
            'domain': 'general',
            'use_synonyms': True
        }
        self.extractor_name = "domain_extractor"
        
        # 도메인별 전문 용어 사전
        self.domain_terms = {
            'ai': {
                'keywords': ['인공지능', 'AI', '머신러닝', '딥러닝', '신경망', '알고리즘'],
                'synonyms': {
                    '인공지능': ['AI', 'artificial intelligence', '머신러닝'],
                    '머신러닝': ['machine learning', 'ML', '딥러닝'],
                    '딥러닝': ['deep learning', 'neural network', '신경망']
                }
            },
            'data': {
                'keywords': ['데이터', '분석', '마이닝', '처리', '베이스'],
                'synonyms': {
                    '데이터': ['data', 'information', 'dataset'],
                    '분석': ['analysis', 'analytics', 'processing'],
                    '마이닝': ['mining', 'extraction', 'discovery']
                }
            },
            'system': {
                'keywords': ['시스템', '구현', '설계', '아키텍처', '플랫폼'],
                'synonyms': {
                    '시스템': ['system', 'framework', 'platform'],
                    '구현': ['implementation', 'development', 'deployment'],
                    '설계': ['design', 'architecture', 'structure']
                }
            }
        }
    
    def extract_keywords(self, query: str, **kwargs) -> List[str]:
        """
        도메인별 키워드 추출
        
        Args:
            query: 검색 쿼리
            **kwargs: 추가 옵션 (domain 등)
            
        Returns:
            추출된 키워드 리스트
        """
        domain = kwargs.get('domain', self.config['domain'])
        keywords = []
        
        # 도메인별 전문 용어 추출
        if domain in self.domain_terms:
            domain_data = self.domain_terms[domain]
            
            # 직접 매칭되는 키워드
            for term in domain_data['keywords']:
                if term.lower() in query.lower():
                    keywords.append(term)
            
            # 동의어 확장
            if self.config.get('use_synonyms', True):
                synonyms = self._extract_synonyms(query, domain_data.get('synonyms', {}))
                keywords.extend(synonyms)
        
        # 일반 키워드도 추출
        general_keywords = self._extract_general_keywords(query)
        keywords.extend(general_keywords)
        
        # 중복 제거
        keywords = list(set(keywords))
        
        logging.info(f"도메인 키워드 추출 ({domain}): {', '.join(keywords)}")
        return keywords[:self.config['max_keywords']]
    
    def _extract_synonyms(self, query: str, synonyms_dict: Dict[str, List[str]]) -> List[str]:
        """동의어 추출"""
        found_synonyms = []
        
        for main_term, synonyms in synonyms_dict.items():
            if main_term.lower() in query.lower():
                found_synonyms.extend(synonyms)
        
        return found_synonyms
    
    def _extract_general_keywords(self, query: str) -> List[str]:
        """일반 키워드 추출"""
        # 간단한 단어 분리
        words = query.split()
        keywords = []
        
        for word in words:
            # 길이가 3 이상인 단어만
            if len(word) > 3:
                # 특수 문자 제거
                clean_word = ''.join(c for c in word if c.isalnum())
                if clean_word:
                    keywords.append(clean_word)
        
        return keywords
    
    def get_extractor_name(self) -> str:
        return self.extractor_name
    
    def get_config(self) -> dict:
        return self.config.copy()
    
    def update_config(self, new_config: dict):
        """설정 업데이트"""
        self.config.update(new_config)
        logging.info(f"도메인 추출기 설정 업데이트: {new_config}")
    
    def add_domain_terms(self, domain: str, terms: Dict[str, List[str]]):
        """새로운 도메인 용어 추가"""
        self.domain_terms[domain] = terms
        logging.info(f"도메인 '{domain}' 용어 추가")
