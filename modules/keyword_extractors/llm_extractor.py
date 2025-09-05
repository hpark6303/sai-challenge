"""
LLM 기반 키워드 추출기
- 기존 retrieval.py의 _extract_keywords_with_llm 로직을 정리
"""

import re
import logging
from typing import List, Optional
from .base_extractor import KeywordExtractor

class LLMKeywordExtractor(KeywordExtractor):
    """LLM 기반 키워드 추출기"""
    
    def __init__(self, gemini_client, config: dict = None):
        """
        LLM 키워드 추출기 초기화
        
        Args:
            gemini_client: Gemini API 클라이언트
            config: 추출 설정
        """
        self.gemini_client = gemini_client
        self.config = config or {
            'max_keywords': 5,
            'min_keyword_length': 2,
            'use_advanced_prompt': True
        }
        self.extractor_name = "llm_extractor"
    
    def extract_keywords(self, query: str, **kwargs) -> List[str]:
        """
        LLM을 사용한 키워드 추출
        
        Args:
            query: 검색 쿼리
            **kwargs: 추가 옵션
            
        Returns:
            추출된 키워드 리스트
        """
        if not self.gemini_client:
            logging.warning("Gemini 클라이언트가 없습니다.")
            return []
        
        try:
            # 고급 키워드 생성 프롬프트 사용
            prompt = self._create_keyword_generation_prompt(query)
            
            response = self.gemini_client.generate_answer(prompt)
            
            # 응답에서 키워드 추출
            keywords = self._parse_llm_response(response)
            
            logging.info(f"LLM 키워드 추출: {', '.join(keywords)}")
            return keywords
            
        except Exception as e:
            logging.error(f"LLM 키워드 추출 실패: {e}")
            return []
    
    def _create_keyword_generation_prompt(self, query: str) -> str:
        """키워드 생성 프롬프트 생성"""
        if self.config.get('use_advanced_prompt', True):
            return self._create_advanced_prompt(query)
        else:
            return self._create_simple_prompt(query)
    
    def _create_advanced_prompt(self, query: str) -> str:
        """고급 키워드 생성 프롬프트"""
        return f"""
당신은 한국 학술 연구 데이터베이스 'ScienceOn'의 검색 성능을 극대화하는 전문가입니다.
사용자의 질문을 분석하여 ScienceOn API에서 효과적으로 검색할 수 있는 **작은 단위의 키워드들**을 생성하세요.

질문: "{query}"

요구사항:
1. **학술적 정확성**: 연구 분야의 전문 용어와 개념을 정확히 반영
2. **즉시 검색 가능**: ScienceOn API에서 바로 검색할 수 있는 형태
3. **다양성**: 동일한 개념의 다양한 표현 방식 포함
4. **구체성**: 너무 일반적이지 않은 구체적인 키워드

키워드 생성 규칙:
- 각 키워드는 한 줄에 하나씩
- 불필요한 기호나 설명 없이 키워드만
- 최대 {self.config['max_keywords']}개
- 한국어와 영어 모두 가능

키워드:
"""
    
    def _create_simple_prompt(self, query: str) -> str:
        """간단한 키워드 생성 프롬프트"""
        return f"""
다음 질문에서 검색에 유용한 키워드 {self.config['max_keywords']}개를 추출해주세요.

질문: "{query}"

키워드 (한 줄에 하나씩):
"""
    
    def _parse_llm_response(self, response: str) -> List[str]:
        """LLM 응답에서 키워드 파싱"""
        keywords = []
        
        for line in response.strip().split('\n'):
            line = line.strip()
            
            # 빈 줄이나 주석 제외
            if not line or line.startswith('#') or line.startswith('-'):
                continue
            
            # 불필요한 문자 제거
            clean_keyword = self._clean_keyword(line)
            
            # 유효한 키워드인지 확인
            if self._is_valid_keyword(clean_keyword):
                keywords.append(clean_keyword)
        
        return keywords[:self.config['max_keywords']]
    
    def _clean_keyword(self, keyword: str) -> str:
        """키워드 정리"""
        # 불필요한 문자 제거
        clean = keyword.replace('*', '').replace('-', '').replace('•', '').strip()
        
        # 숫자로 시작하는 경우 제거
        if clean and clean[0].isdigit():
            clean = clean.split('.', 1)[-1].strip()
        
        return clean
    
    def _is_valid_keyword(self, keyword: str) -> bool:
        """유효한 키워드인지 확인"""
        if not keyword:
            return False
        
        # 최소 길이 확인
        if len(keyword) < self.config['min_keyword_length']:
            return False
        
        # 너무 긴 키워드 제외
        if len(keyword) > 50:
            return False
        
        # 특수 문자만 있는 경우 제외
        if not re.search(r'[가-힣a-zA-Z]', keyword):
            return False
        
        return True
    
    def get_extractor_name(self) -> str:
        return self.extractor_name
    
    def get_config(self) -> dict:
        return self.config.copy()
    
    def update_config(self, new_config: dict):
        """설정 업데이트"""
        self.config.update(new_config)
        logging.info(f"LLM 추출기 설정 업데이트: {new_config}")
