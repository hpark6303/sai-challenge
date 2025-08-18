"""
프롬프트 엔지니어링 모듈
- 고품질 프롬프트 생성
- 언어별 프롬프트 최적화
- 답변 품질 보장
"""

import re
from typing import List, Dict
from .config import PROMPT_CONFIG, ANSWER_CONFIG

class PromptEngineer:
    """프롬프트 엔지니어"""
    
    def __init__(self):
        """프롬프트 엔지니어 초기화"""
        pass
    
    def create_final_prompt(self, query: str, context: str, language: str) -> str:
        """
        최종 답변 생성을 위한 프롬프트 (고품질 버전)
        
        Args:
            query: 사용자 질문
            context: 참고 문서 컨텍스트
            language: 언어 ('ko' 또는 'en')
            
        Returns:
            생성된 프롬프트
        """
        language_instruction = "한국어로" if language == "ko" else "영어로 (in English)"
        
        return f"""{PROMPT_CONFIG['system_role']}

### 참고 문서 (Context):
{context}

### 과제 (Task):
'참고 문서'의 내용을 완벽하게 숙지한 후, 아래 '출력 형식'에 맞춰 '원본 질문'에 대한 분석 보고서를 작성하세요.

### 원본 질문 (Original Question):
{query}

### 핵심 지침 (Core Directives):
1. **언어 준수**: 원본 질문이 '{language_instruction}'로 작성되었으므로, 최종 보고서 전체를 반드시 **{language_instruction}**로 작성해야 합니다.
2. **전문가의 자세**: 당신은 이 주제의 전문가입니다. "정보가 부족하다", "~일 것으로 추정된다"와 같은 불확실한 표현을 절대 사용하지 마세요.
3. **사실 기반 종합**: 여러 문서에 흩어져 있는 정보를 논리적으로 연결하고 종합하여 하나의 완성된 글로 재구성하세요.
4. **엄격한 출처 표기**: 보고서의 모든 문장은 반드시 '참고 문서'에 명시된 사실에 기반해야 합니다.

### 출력 형식 (Output Format):
{self._format_output_instructions()}

---
### 최종 보고서:
"""
    
    def _format_output_instructions(self) -> str:
        """출력 형식 지침 포맷팅"""
        formatted = []
        for i, instruction in enumerate(PROMPT_CONFIG['output_format'], 1):
            formatted.append(f"{i}. **{instruction}**")
        return "\n".join(formatted)
    
    def create_simple_prompt(self, query: str, context: str) -> str:
        """
        간단한 프롬프트 (fallback용)
        
        Args:
            query: 사용자 질문
            context: 참고 문서 컨텍스트
            
        Returns:
            간단한 프롬프트
        """
        return f"""다음 질문에 대해 제공된 논문 정보를 바탕으로 정확하고 상세한 답변을 작성하세요.

질문: {query}

참고 논문 정보:
{context}

답변 (최소 {ANSWER_CONFIG['min_answer_length']}자 이상):
"""
    
    def create_quality_check_prompt(self, answer: str, query: str) -> str:
        """
        답변 품질 검증용 프롬프트
        
        Args:
            answer: 생성된 답변
            query: 원본 질문
            
        Returns:
            품질 검증 프롬프트
        """
        return f"""다음 답변이 질문에 적절하게 답변하고 있는지 평가해주세요.

질문: {query}

답변: {answer}

평가 기준:
1. 답변이 질문의 핵심을 다루고 있는가?
2. 답변이 충분히 상세한가? (최소 {ANSWER_CONFIG['min_answer_length']}자)
3. 답변이 논리적으로 구성되어 있는가?

평가 결과 (적절함/부적절함):
"""
    
    def detect_language(self, text: str) -> str:
        """
        텍스트 언어 감지
        
        Args:
            text: 감지할 텍스트
            
        Returns:
            언어 코드 ('ko' 또는 'en')
        """
        korean_chars = len(re.findall('[가-힣]', text))
        total_chars = len(re.findall('[a-zA-Z가-힣]', text))
        
        if total_chars == 0:
            return 'en'  # 기본값
        
        korean_ratio = korean_chars / total_chars
        return 'ko' if korean_ratio > 0.3 else 'en'
    
    def enhance_context(self, context: str, query: str) -> str:
        """
        컨텍스트 강화
        
        Args:
            context: 원본 컨텍스트
            query: 사용자 질문
            
        Returns:
            강화된 컨텍스트
        """
        if not context:
            return context
        
        # 질문 키워드 강조
        query_keywords = re.findall(r'\w+', query.lower())
        
        enhanced_context = context
        for keyword in query_keywords[:5]:  # 상위 5개 키워드만
            if len(keyword) > 2:
                # 키워드가 포함된 문장 강조
                enhanced_context = enhanced_context.replace(
                    keyword, f"**{keyword}**"
                )
        
        return enhanced_context
    
    def create_fallback_prompt(self, query: str, documents: List[Dict]) -> str:
        """
        Fallback 답변 생성용 프롬프트
        
        Args:
            query: 사용자 질문
            documents: 참고 문서 리스트
            
        Returns:
            Fallback 프롬프트
        """
        if not documents:
            return f"질문 '{query}'에 대한 충분한 정보를 찾을 수 없습니다."
        
        # 문서에서 핵심 정보 추출
        titles = [doc.get('title', '') for doc in documents if doc.get('title')]
        
        language = self.detect_language(query)
        
        if language == "ko":
            prompt = f"제공된 문서들을 바탕으로 '{query}'에 대한 분석을 수행했습니다.\n\n"
            prompt += "주요 참고 문서:\n"
            for i, title in enumerate(titles[:3], 1):
                prompt += f"{i}. {title}\n"
            prompt += f"\n이 문서들은 질문과 관련된 유용한 정보를 제공합니다. 상세한 내용은 참고 문서를 확인하시기 바랍니다."
        else:
            prompt = f"Based on the provided documents, I have analyzed '{query}'.\n\n"
            prompt += "Key reference documents:\n"
            for i, title in enumerate(titles[:3], 1):
                prompt += f"{i}. {title}\n"
            prompt += f"\nThese documents provide useful information related to the question. Please refer to the documents for detailed content."
        
        return prompt
