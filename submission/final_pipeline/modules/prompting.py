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
1. 답변의 정확성 (0-10점)
2. 답변의 완성도 (0-10점)
3. 질문과의 관련성 (0-10점)

평가 결과:
"""

    def create_advanced_keyword_generation_prompt(self, question: str) -> str:
        """
        단계별 사고(Step-by-Step)를 통해 ScienceOn API에 최적화된
        고품질 검색 키워드를 생성하는 프롬프트.
        """
        return f"""
# ROLE & GOAL
당신은 한국 학술 연구 데이터베이스 'ScienceOn'의 검색 성능을 극대화하는 임무를 맡은 최고 수준의 프롬프트 엔지니어이자 연구원입니다. 당신의 목표는 사용자의 복잡한 질문을 분석하여, 관련성이 가장 높은 핵심 논문을 찾아낼 수 있는 정교하고 다각적인 검색 키워드 셋을 생성하는 것입니다.

# CONTEXT
ScienceOn은 한국어로 된 학술 용어와 핵심적인 영문 약어(Acronym)에 가장 잘 반응하는 전문적인 데이터베이스입니다. "결과", "연구", "방법"과 같은 일반적인 단어는 검색에 도움이 되지 않습니다.

# STEP-BY-STEP PROCESS
당신은 다음의 4단계 사고 과정을 반드시 순서대로 따라야 합니다.

**Step 1: 질문 분해 (Deconstruct the Question)**
- 먼저, 사용자의 질문을 의미론적으로 가장 작은 핵심 구성 요소로 분해합니다.
- 질문의 주제(Topic), 범위(Scope), 관점(Perspective), 구체적인 대상(Object)은 무엇인지 명확히 식별하세요.

**Step 2: 핵심 개념 식별 (Identify Core Concepts)**
- 분해된 구성 요소를 바탕으로, 이 질문의 학술적 핵심이 되는 개념들을 모두 나열하세요.
- 한국어 학술 용어, 일반적으로 통용되는 영문 명칭 및 약어(Acronym)를 모두 고려해야 합니다.

**Step 3: 키워드 확장 및 브레인스토밍 (Expand & Brainstorm Keywords)**
- 식별된 핵심 개념들을 바탕으로, 검색에 사용할 수 있는 잠재적 키워드 목록을 만듭니다.
- 동의어, 유의어, 상위 개념, 하위 개념 등을 모두 고려하여 풍부한 키워드 풀을 구성하세요.
- 예를 들어, '인공지능'이라면 '딥러닝', '머신러닝', '자연어 처리', 'LLM' 등으로 확장할 수 있습니다.

**Step 4: 최종 키워드 선택 및 정제 (Select & Refine Final Keywords)**
- 브레인스토밍된 키워드들 중에서, ScienceOn API 검색에 가장 효과적일 것이라고 판단되는 **최상위 키워드 3개에서 5개**를 신중하게 선택하세요.
- 선택된 키워드는 너무 광범위하지도, 너무 협소하지도 않아야 합니다.
- 최종 키워드들은 반드시 줄바꿈(new line)으로만 구분하여 다른 설명 없이 출력하세요.

# EXAMPLE OF YOUR THOUGHT PROCESS
---
[사용자 질문 예시]
"컴퓨터과학자들이 제안한 인공지능 정의에 내재된 지능, 뇌, 그리고 컴퓨터 모의 사이의 논쟁적 쟁점을 어떻게 요약할 수 있나요?"

[당신의 사고 과정 예시]
- **Step 1: 질문 분해**
  - 주제: 인공지능(AI)의 정의
  - 범위: 컴퓨터과학자들의 제안
  - 관점: 지능, 뇌, 컴퓨터 시뮬레이션 간의 관계 및 논쟁
  - 대상: 논쟁적 쟁점
- **Step 2: 핵심 개념 식별**
  - 인공지능 정의 (Definition of Artificial Intelligence)
  - 인지과학 (Cognitive Science)
  - 계산주의 마음 이론 (Computational Theory of Mind)
  - 튜링 테스트 (Turing Test)
  - 중국어 방 논변 (Chinese Room Argument)
- **Step 3: 키워드 확장 및 브레인스토밍**
  - 인공지능, AI, 지능의 정의, 강한 AI, 약한 AI, 튜링 테스트, 존 설, 인지 모의, 뇌-컴퓨터 인터페이스, 상징주의, 연결주의, 인공 의식...
- **Step 4: 최종 키워드 선택 및 정제**
  (아래 내용이 최종 출력이 됨)
  인공지능 정의 논쟁
  계산주의 마음 이론
  튜링 테스트 비판
  인지과학과 AI
---

# EXECUTION
이제 아래 '사용자 질문'에 대해 위의 4단계 사고 과정을 적용하여 최적의 검색 키워드를 생성하세요.

# USER QUESTION:
{question}

# FINAL SEARCH KEYWORDS:
"""

    def create_bilingual_keyword_prompt(self, question: str, target_language: str = "ko") -> str:
        """
        쌍방 언어 키워드 생성을 위한 프롬프트
        
        Args:
            question: 원본 질문
            target_language: 대상 언어 ('ko' 또는 'en')
            
        Returns:
            쌍방 언어 키워드 생성 프롬프트
        """
        language_instruction = "한국어" if target_language == "ko" else "영어"
        
        return f"""
당신은 다국어 학술 검색 전문가입니다. 주어진 질문을 분석하여 {language_instruction}로 된 고품질 검색 키워드를 생성해주세요.

# 질문 분석 및 키워드 생성 가이드라인:

1. **질문의 핵심 주제 파악**: 질문에서 가장 중요한 학술적 개념을 식별하세요.
2. **전문 용어 추출**: 해당 분야의 전문적인 용어와 개념을 추출하세요.
3. **동의어 및 관련어 확장**: 핵심 개념의 동의어, 유의어, 상위/하위 개념을 포함하세요.
4. **학술적 표현 사용**: 일반적인 단어보다는 학술 논문에서 사용되는 전문적인 표현을 선호하세요.

# 원본 질문:
{question}

# {language_instruction} 검색 키워드 (3-5개, 줄바꿈으로 구분):
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
